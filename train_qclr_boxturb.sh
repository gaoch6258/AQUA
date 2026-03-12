#!/usr/bin/env bash
set -u
set -o pipefail

log_file="outputs/train_qclr_boxturb.log"
heartbeat_seconds=60

dataset_name="boxTurb_tzxyc"
output_dir="./outputs/qclr_boxturb"

mkdir -p "outputs"

append_log() {
  echo "$1" >> "${log_file}"
}

print_and_log() {
  echo "$1"
  append_log "$1"
}

restart=0
run_ts="$(date "+%Y-%m-%d %H:%M:%S")"
err_file="outputs/elastic_error_qclr_boxturb.json"

print_and_log "==== qclr boxturb start ${run_ts} ===="
print_and_log "TORCHELASTIC_ERROR_FILE=${err_file}"

TORCHELASTIC_ERROR_FILE="${err_file}" PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
  GRPO_TIMING=1 GRPO_TIMING_EVERY=1 \
  GRPO_QCLR_ENABLE=1 GRPO_QCLR_LAMBDA_P=0.10 \
  accelerate launch --num_processes 1 --num_machines 1 \
    "real_fluid_grpo.py" \
    --dataset "${dataset_name}" \
    --output-dir "${output_dir}" \
  >> "${log_file}" 2>&1 &

train_pid=$!
start_ts="$(date +%s)"

while kill -0 "${train_pid}" 2>/dev/null; do
  sleep "${heartbeat_seconds}"
  if ! kill -0 "${train_pid}" 2>/dev/null; then
    break
  fi
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  echo "[monitor] running ${elapsed}s ..."
done

wait "${train_pid}"
exit_code=$?
print_and_log "exit_code=${exit_code} $(date "+%Y-%m-%d %H:%M:%S")"

if [[ -s "${err_file}" ]]; then
  print_and_log "---- elastic error file (${err_file}) ----"
  cat "${err_file}" >> "${log_file}"
  print_and_log "---- end elastic error file ----"
fi

exit "${exit_code}"
