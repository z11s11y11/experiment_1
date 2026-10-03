#!/usr/bin/env bash

set -euo pipefail

# 文件与目录定义
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="${REPO_ROOT}/configs/SM90_H100/gpgpusim.config"
RUN_SCRIPT="${REPO_ROOT}/tutorials/triton-gemm/run.sh"
SIM_LOG="${REPO_ROOT}/tutorials/triton-gemm/run/simulation.log"
LOG_DIR="${REPO_ROOT}/sim_logs"          # ← 新增：每配置的完整日志存这里
mkdir -p -- "${LOG_DIR}"                 # ← 新增
RESULT_CSV="${REPO_ROOT}/n_mem_results.csv"
RESULT_MD="${REPO_ROOT}/reslut.md"
TMP_DIR="$(mktemp -d)"

cleanup() {
  if [[ -f "${TMP_DIR}/original.config" ]]; then
    cp -- "${TMP_DIR}/original.config" "${CONFIG}"
  fi
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cp -- "${CONFIG}" "${TMP_DIR}/original.config"

active_count="$(awk '$1 == "-gpgpu_n_mem" { n++ } END { print n+0 }' "${CONFIG}")"
if [[ "${active_count}" -ne 1 ]]; then
  echo "Expected exactly one active -gpgpu_n_mem line in ${CONFIG}; found ${active_count}." >&2
  exit 1
fi

export OMP_NUM_THREADS=8
printf 'n_mem,run1,run2,average_gpu_tot_sim_cycle\n' > "${RESULT_CSV}"
printf '\n\n## GEMM memory channel sweep (%s)\n\n| -gpgpu_n_mem | Average gpu_tot_sim_cycle (2 runs) |\n| ---: | ---: |\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "${RESULT_MD}"

# 参数扫描主循环
for multiplier in {5..15}; do
  n_mem=$((multiplier * 16))
  awk -v n_mem="${n_mem}" '
    $1 == "-gpgpu_n_mem" { print "-gpgpu_n_mem " n_mem; next }
    { print }
  ' "${TMP_DIR}/original.config" > "${TMP_DIR}/next.config"
  # 2) ★ 必须写回真正被读取的配置文件 ★
  cp -- "${TMP_DIR}/next.config" "${CONFIG}"
  # 每次跑2次仿真
  row="${n_mem}"
  total=0
  for run in {1..2}; do
    echo "Running -gpgpu_n_mem=${n_mem} (${run}/2)"
    if ! (cd "${REPO_ROOT}/tutorials/triton-gemm" && PERF_SIM_CONFIG=SM90_H100 "${RUN_SCRIPT}") > "${TMP_DIR}/run.log" 2>&1; then
      tail -n 40 "${TMP_DIR}/run.log" >&2
      echo "Run failed for -gpgpu_n_mem=${n_mem}, repetition ${run}." >&2
      exit 1
    fi

    cp -- "${SIM_LOG}" "${LOG_DIR}/simulation_n${n_mem}_run${run}.log"   # ← 新增
    cycles="$(awk '$1 == "gpu_tot_sim_cycle" && $2 == "=" { print $3; exit }' "${SIM_LOG}")"
    if [[ ! "${cycles}" =~ ^[0-9]+$ ]]; then
      echo "Missing or invalid gpu_tot_sim_cycle for -gpgpu_n_mem=${n_mem}, repetition ${run}." >&2
      exit 1
    fi
    row+=",${cycles}"
    total=$((total + cycles))
  done

  # 结果汇总
  average="$(awk -v total="${total}" 'BEGIN { printf "%.2f", total / 2 }')"
  printf '%s,%s\n' "${row}" "${average}" >> "${RESULT_CSV}"
  printf '| %s | %s |\n' "${n_mem}" "${average}" >> "${RESULT_MD}"
done

printf '\nCompleted all 11 configurations with 2 runs each. Detailed results: `%s`.\n' "$(basename -- "${RESULT_CSV}")" >> "${RESULT_MD}"
echo "Results written to ${RESULT_CSV} and ${RESULT_MD}"