#!/bin/bash
# Detached job queue: runs long jobs independently of any Claude Code / VS Code session.
#
#   setsid nohup scripts/run_queue.sh experiments/v2_queue.txt > /dev/null 2>&1 < /dev/null &
#
# Each line of the queue file is one shell command, run from the repo root in the
# dqn_ml env, one after another. Empty lines and '#' lines are skipped. The file is
# re-read before every line, so commands can be APPENDED while the queue runs, and
# `# done` markers are never needed: a line counter is kept in <queue>.pos, so a
# restarted queue continues after the last finished line. A failing command is logged
# and the queue moves on, so later report steps still run.
# Log: <queue>.log (start, end and exit code of every command, plus its output unless
# the command redirects it itself).
set -u
cd "$(dirname "$0")/.." || exit 1
Q="$1"; LOG="${Q%.*}.log"; POS="${Q%.*}.pos"
exec 9>"${Q%.*}.lock"
flock -n 9 || { echo "[$(date '+%F %T')] queue already running, exit" >> "$LOG"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate dqn_ml || exit 1

n=$(cat "$POS" 2>/dev/null || echo 0)                    # last finished line
while [ "$n" -lt "$(awk 'END {print NR}' "$Q")" ]; do
  n=$((n + 1))
  cmd=$(sed -n "${n}p" "$Q")
  case "$cmd" in ''|'#'*) echo "$n" > "$POS"; continue ;; esac
  echo "[$(date '+%F %T')] start line $n: $cmd" >> "$LOG"
  bash -c "$cmd" >> "$LOG" 2>&1
  rc=$?                                                   # read before $(date) resets $?
  echo "[$(date '+%F %T')] end   line $n: exit $rc" >> "$LOG"
  echo "$n" > "$POS"
done
echo "[$(date '+%F %T')] queue empty, exit" >> "$LOG"
