#!/bin/bash
# L2-03 の再実行ドライバー (再起動対策つき)。
# 第1段階で予測をチェックポイントに保存し、予測幅ごとにブートストラップして
# 断片を保存する。途中で止まっても、完了した予測幅は飛ばして再開できる。
cd ~/workspace/kinescope/repo/experiments/L2-03_salt_formation_leverage_k15_adjacent_mean || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/workspace/mt-venv/bin/python
if [ ! -f _checkpoint_phase1.npz ] || [ ! -f _checkpoint_phase1.json ]; then
  echo "=== phase1 ===" | tee -a driver.log
  $PY run_l2_03_phase1.py 2>&1 | tee -a run_l2_03_phase1.log
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "phase1 FAILED" | tee -a driver.log; exit 2; }
else
  echo "=== phase1 checkpoint exists, skipping ===" | tee -a driver.log
fi
for h in 1 5 10 30 50 90; do
  if [ -f _checkpoint_h${h}.json ]; then
    echo "=== horizon $h: checkpoint exists, skipping ===" | tee -a driver.log
    continue
  fi
  echo "=== horizon $h ===" | tee -a driver.log
  $PY run_l2_03_boot.py $h 2>&1 | tee -a run_l2_03_boot_h${h}.log
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "horizon $h FAILED" | tee -a driver.log; exit 3; }
done
echo "=== merge ===" | tee -a driver.log
$PY run_l2_03_merge.py 2>&1 | tee -a run_l2_03_merge.log
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "merge FAILED" | tee -a driver.log; exit 4; }
echo "ALL DONE" | tee -a driver.log
