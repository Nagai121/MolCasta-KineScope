"""K3-05 補足: 起点フレームの水の量を取るための、取得→抽出→後片付けのドライバ。

run_pipeline.py と同じ流れで、XTC の先頭をフレーム100 まで含む大きさ
（環境変数 K3_XTC_TARGET_MB=32）で取り、extract_water_f100.py で
起点フレームの量を抽出し、確認できたら取得した先頭ファイルは消す。
途中まで終わっている場合は続きから進む。

使い方: python run_pipeline_f100.py <list_file>
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/hatch/workspace/mt-venv/bin/python"
HEAD_ROOT = Path("/home/hatch/workspace/kinescope/work/k3_05_total_head")
WATER_OUT = Path("/home/hatch/workspace/kinescope/work/k3_05_water")
ENV = dict(os.environ, K3_XTC_TARGET_MB="32")


def main():
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    for i, prot in enumerate(proteins):
        t0 = time.time()
        out_file = WATER_OUT / f"{prot}__f100.npz"
        if out_file.exists():
            print(f"[{i+1}/{len(proteins)}] {prot} skip (extracted)", flush=True)
            continue
        r = subprocess.run([PY, str(HERE / "fetch_total_head.py"), prot], env=ENV)
        if r.returncode != 0:
            print(f"[{i+1}/{len(proteins)}] {prot} FETCH FAILED", flush=True)
            continue
        r = subprocess.run([PY, str(HERE / "extract_water_f100.py"), prot])
        if r.returncode != 0 or not out_file.exists():
            print(f"[{i+1}/{len(proteins)}] {prot} EXTRACT FAILED", flush=True)
            continue
        shutil.rmtree(HEAD_ROOT / prot, ignore_errors=True)
        print(f"[{i+1}/{len(proteins)}] {prot} done ({time.time()-t0:.0f}s), head files removed", flush=True)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
