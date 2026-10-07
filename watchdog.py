"""Runs train_planestrike.py and restarts it on OOM, crash or hang.

OOM on this TPU VM often does not kill the job: the failing thread logs
RESOURCE_EXHAUSTED and the process (and its vLLM EngineCore child, which keeps
/dev/vfio open) just sits there. The watchdog therefore:
  * starts each attempt in its own session, so the whole process group
    (including vLLM children) can be killed;
  * treats an attempt as failed if it exits non-zero (42 = OOM from the
    in-process excepthook), if an OOM pattern appears in its log, or if
    neither the log nor games.jsonl grows for --stall_minutes;
  * kills the process group, waits until no process of that session holds
    the TPU, and restarts. Training resumes from the latest checkpoint;
  * after an OOM, lowers --vllm_hbm_utilization by --oom_hbm_step (down to
    --min_hbm_utilization), leaving more HBM for the trainer.

Usage:
  python watchdog.py --run_dir runs/grpo -- [train_planestrike.py args]
Stop with `kill <watchdog_pid>` (in <run_dir>/watchdog_status.json); the
watchdog then kills the whole training session, vLLM included.
"""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OOM_EXIT_CODE = 42
OOM_RE = re.compile(
    r"RESOURCE_EXHAUSTED|Out of memory while trying|Ran out of memory|"
    r"OutOfMemoryError|Attempting to allocate .* That was not possible|"
    r"exiting with code 42"
)

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--max_attempts", type=int, default=30)
parser.add_argument("--stall_minutes", type=float, default=45)
parser.add_argument("--startup_stall_minutes", type=float, default=40,
                    help="Allowed silence before the first game (model load, compile).")
parser.add_argument("--vllm_hbm_utilization", type=float, default=0.23)
parser.add_argument("--oom_hbm_step", type=float, default=0.02)
parser.add_argument("--min_hbm_utilization", type=float, default=0.18)
args, train_args = parser.parse_known_args()
if train_args and train_args[0] == "--":
  train_args = train_args[1:]

run_dir = os.path.abspath(args.run_dir)
log_dir = os.path.join(run_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
games_path = os.path.join(run_dir, "games.jsonl")
status_path = os.path.join(run_dir, "watchdog_status.json")


def log(msg):
  line = f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
  print(line, flush=True)
  with open(os.path.join(log_dir, "watchdog.log"), "a") as f:
    f.write(line + "\n")


def size(path):
  try:
    return os.path.getsize(path)
  except OSError:
    return 0


def tpu_holders():
  """PIDs with /dev/vfio/* open."""
  pids = []
  for fd_dir in glob.glob("/proc/[0-9]*/fd"):
    try:
      for fd in os.listdir(fd_dir):
        if os.readlink(os.path.join(fd_dir, fd)).startswith("/dev/vfio/"):
          pids.append(int(fd_dir.split("/")[2]))
          break
    except OSError:
      continue
  return pids


def session_pids(sid):
  pids = []
  for p in glob.glob("/proc/[0-9]*"):
    try:
      pid = int(p.split("/")[2])
      if os.getsid(pid) == sid:
        pids.append(pid)
    except (OSError, ValueError):
      continue
  return pids


def kill_session(sid):
  """SIGTERM then SIGKILL every process in the attempt's session."""
  for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
    pids = session_pids(sid)
    if not pids:
      return
    log(f"sending {sig.name} to session {sid}: {pids}")
    for pid in pids:
      try:
        os.kill(pid, sig)
      except OSError:
        pass
    deadline = time.time() + wait
    while time.time() < deadline and session_pids(sid):
      time.sleep(1)


def wait_for_tpu(timeout=300):
  deadline = time.time() + timeout
  while time.time() < deadline:
    holders = tpu_holders()
    if not holders:
      return True
    time.sleep(5)
  log(f"TPU still held by {tpu_holders()} after {timeout}s")
  return False


def write_status(**kw):
  # Never overwrite the status of a different (newer) watchdog on this run dir.
  try:
    with open(status_path) as f:
      owner = json.load(f).get("watchdog_pid")
    if owner not in (None, os.getpid()) and os.path.exists(f"/proc/{owner}"):
      return
  except (OSError, ValueError):
    pass
  with open(status_path, "w") as f:
    json.dump({"watchdog_pid": os.getpid(), **kw}, f, indent=1)


current_sid = None


def on_sigterm(signum, frame):
  del frame
  log(f"received signal {signum}; stopping training session {current_sid}")
  if current_sid is not None:
    kill_session(current_sid)
  write_status(state="stopped")
  sys.exit(0)


signal.signal(signal.SIGTERM, on_sigterm)
signal.signal(signal.SIGINT, on_sigterm)

hbm = args.vllm_hbm_utilization
attempt = len(glob.glob(os.path.join(log_dir, "train_attempt*.log")))
consecutive_failures = 0
repeats, last_signature = 0, None
while attempt < args.max_attempts:
  attempt += 1
  holders = tpu_holders()
  if holders:
    log(f"TPU busy before attempt {attempt} (pids {holders}); waiting")
    if not wait_for_tpu():
      log("giving up: TPU is held by a process this watchdog did not start")
      sys.exit(2)
  attempt_log = os.path.join(log_dir, f"train_attempt{attempt}.log")
  cmd = [sys.executable, "-u", os.path.join(HERE, "train_planestrike.py"),
         "--run_dir", run_dir, "--vllm_hbm_utilization", f"{hbm:.3f}", *train_args]
  log(f"attempt {attempt}: {' '.join(cmd)}")
  with open(attempt_log, "w") as out:
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=HERE, start_new_session=True,
                            env={**os.environ, "WATCHDOG_ATTEMPT": str(attempt)})
  sid = current_sid = proc.pid
  write_status(attempt=attempt, pid=proc.pid, log=attempt_log, hbm=hbm, state="running")

  start = last_progress = time.time()
  seen_game = False
  last_sizes = (0, size(games_path))
  scan_pos = 0
  reason = None
  while True:
    time.sleep(20)
    code = proc.poll()
    sizes = (size(attempt_log), size(games_path))
    if sizes != last_sizes:
      last_progress = time.time()
      seen_game = seen_game or sizes[1] != last_sizes[1]
      last_sizes = sizes
    with open(attempt_log, errors="replace") as f:
      f.seek(scan_pos)
      chunk = f.read()
      scan_pos = f.tell()
    m = OOM_RE.search(chunk)
    if m:
      reason = f"oom (log matched {m.group(0)!r})"
    elif code is not None:
      reason = "finished" if code == 0 else (
          "oom (exit 42)" if code == OOM_EXIT_CODE else f"crash (exit {code})")
    else:
      limit = args.stall_minutes if seen_game else args.startup_stall_minutes
      if time.time() - last_progress > limit * 60:
        reason = f"hang (no progress for {limit} min)"
    if reason:
      break

  log(f"attempt {attempt} ended: {reason} after {(time.time() - start) / 60:.1f} min")
  kill_session(sid)
  proc.wait()
  if reason == "finished":
    write_status(attempt=attempt, state="finished", hbm=hbm)
    log("training finished")
    sys.exit(0)
  consecutive_failures = consecutive_failures + 1 if not seen_game else 1
  # The same crash over and over (e.g. a code bug) won't fix itself by restarting.
  fatal_files = sorted(glob.glob(os.path.join(log_dir, "fatal_*.txt")), key=os.path.getmtime)
  signature = None
  if reason.startswith("crash") and fatal_files and os.path.getmtime(fatal_files[-1]) >= start:
    with open(fatal_files[-1], errors="replace") as f:
      lines = [l for l in f.read().splitlines() if l.strip() and not l.startswith("[fatal]")]
    signature = lines[-1] if lines else None
  repeats = repeats + 1 if signature is not None and signature == last_signature else 1
  last_signature = signature
  if signature is not None and repeats >= 3:
    log(f"same crash 3 times in a row, stopping: {signature}")
    write_status(attempt=attempt, state="failed_repeating_crash", hbm=hbm, reason=signature)
    sys.exit(1)
  if reason.startswith("oom"):
    new_hbm = max(args.min_hbm_utilization, hbm - args.oom_hbm_step)
    if new_hbm != hbm:
      log(f"OOM: lowering vllm_hbm_utilization {hbm:.3f} -> {new_hbm:.3f}")
      hbm = new_hbm
  if consecutive_failures >= 5:
    log("5 consecutive attempts failed before producing a game; stopping")
    write_status(attempt=attempt, state="failed", hbm=hbm, reason=reason)
    sys.exit(1)
  wait_for_tpu()
  time.sleep(10)
log("max attempts reached")
write_status(attempt=attempt, state="max_attempts", hbm=hbm)
sys.exit(1)
