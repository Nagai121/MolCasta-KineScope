"""K3-05: 取得→抽出→後片付けをタンパク質ごとに進めるドライバ。

対象は強い土台（K06(b)）の未見100タンパク質と較正用8タンパク質。
タンパク質ごとに、完全形式の先頭を取り（fetch_total_head）、
水の量を抽出し（extract_water_between）、確認できたら取得した
先頭ファイルは消す。途中まで終わっている場合は続きから進む。

使い方: python run_pipeline.py <list_file>
  list_file は1行に1タンパク質名。並行に走らせる場合はリストを分ける。
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/hatch/workspace/mt-venv/bin/python"
HEAD_ROOT = Path("/home/hatch/workspace/kinescope/work/k3_05_total_head")
WATER_OUT = Path("/home/hatch/workspace/kinescope/work/k3_05_water")


def main():
    proteins = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    for i, prot in enumerate(proteins):
        t0 = time.time()
        out_file = WATER_OUT / f"{prot}__head.npz"
        if out_file.exists():
            print(f"[{i+1}/{len(proteins)}] {prot} skip (extracted)", flush=True)
            continue
        r = subprocess.run([PY, str(HERE / "fetch_total_head.py"), prot])
        if r.returncode != 0:
            print(f"[{i+1}/{len(proteins)}] {prot} FETCH FAILED", flush=True)
            continue
        r = subprocess.run([PY, str(HERE / "extract_water_between.py"), prot, "--src", "head"])
        if r.returncode != 0 or not out_file.exists():
            print(f"[{i+1}/{len(proteins)}] {prot} EXTRACT FAILED", flush=True)
            continue
        shutil.rmtree(HEAD_ROOT / prot, ignore_errors=True)
        print(f"[{i+1}/{len(proteins)}] {prot} done ({time.time()-t0:.0f}s), head files removed", flush=True)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
