"""K3-05: ATLAS 完全形式（total）の zip から、先頭のフレームだけを取り出す。

全部で 3.4 GB ある zip を丸ごと落とす代わりに、HTTP の範囲リクエストで
中央ディレクトリと各メンバーの先頭だけを取得する。

- 末尾 64 KB -> 終端レコード -> 中央ディレクトリの位置と大きさ
- 中央ディレクトリ -> 各メンバーの圧縮方式・大きさ・ローカルヘッダ位置
- .tpr は小さいので丸ごと取る（各 0.6 MB 程度）
- .xtc は先頭の圧縮データだけを取り、zlib で少しずつ伸ばしながら
  必要な長さ（目安 24 MB）に達したら止める。XTC はもともと圧縮済みで
  zip の圧縮はほぼ効いていない（圧縮率ほぼ 1）。
  切り取った XTC は MDAnalysis で読め、完結したフレームだけが読める。

使い方: python fetch_total_head.py <protein> [<protein> ...]
出力: <out_dir>/<protein>/<protein>_prod_R{r}.tpr / .xtc（切り取り済み）
     完了したタンパク質には done.txt を置き、再実行時は飛ばす。
"""
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

BASE = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total/{prot}"
OUT_ROOT = Path("/home/hatch/workspace/kinescope/work/k3_05_total_head")
XTC_RAW_TARGET = 3 * 1024 * 1024  # 伸ばした XTC の目標バイト数（完全形式は 10 ps 間隔で、必要なのはフレーム 0 と 10 だけ。1フレームは 1a62_A で約 114 KB、大きなタンパク質でも数百 KB なので、数 MB あれば 11 フレームに足りる。足りない場合は抽出側の確認で失敗として現れ、大きめの設定で取り直す）
CHUNK = 2 * 1024 * 1024


def http_get(url, range_header):
    req = urllib.request.Request(url, headers={"Range": range_header})
    with urllib.request.urlopen(req, timeout=300) as r:
        assert r.status == 206, r.status
        return r.read(), r.headers


def get_range(url, start, end):
    body, _ = http_get(url, f"bytes={start}-{end}")
    return body


def fetch_member_head(url, entry, want_raw):
    """zip メンバーの先頭 want_raw バイト（伸ばした後）を取り出して返す。"""
    lho = entry["lho"]
    hdr = get_range(url, lho, lho + 30 - 1)
    assert hdr[:4] == b"PK\x03\x04", hdr[:4]
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    data_start = lho + 30 + nlen + elen
    if entry["method"] == 0:
        return get_range(url, data_start, data_start + want_raw - 1)
    assert entry["method"] == 8, entry["method"]
    d = zlib.decompressobj(-15)
    out = bytearray()
    pos = data_start
    remaining = entry["csize"]
    while len(out) < want_raw and remaining > 0:
        n = min(CHUNK, remaining)
        buf = get_range(url, pos, pos + n - 1)
        pos += len(buf)
        remaining -= len(buf)
        out += d.decompress(buf)
    return bytes(out)


def fetch_protein(prot):
    out_dir = OUT_ROOT / prot
    if (out_dir / "done.txt").exists():
        print(prot, "skip (done)", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BASE.format(prot=prot)
    # ファイル全体の大きさは末尾の終端レコードから読む。末尾 64 KB を
    # 「bytes=-65536」形式で取る。
    tail, headers = http_get(url, "bytes=-65536")
    total_size = None
    cr = headers.get("Content-Range")
    if cr and "/" in cr:
        total_size = int(cr.rsplit("/", 1)[1])
    i = tail.rfind(b"PK\x05\x06")
    assert i >= 0, "EOCD not found"
    _, _, _, _, n_tot, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_tot == 0xFFFF:
        # ZIP64: ロケータ（終端レコードの直前20バイト）から
        # ZIP64 終端レコードの位置を読み、そこから CD の位置と大きさを取る
        loc = tail[i - 20:i]
        assert loc[:4] == b"PK\x06\x07", loc[:4]
        z64_off = struct.unpack("<Q", loc[8:16])[0]
        rec = get_range(url, z64_off, z64_off + 56 - 1)
        assert rec[:4] == b"PK\x06\x06", rec[:4]
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    if total_size is None:
        total_size = cd_off + cd_size + 22
    cd = get_range(url, cd_off, cd_off + cd_size - 1)
    entries = {}
    off = 0
    while off < len(cd) and cd[off:off + 4] == b"PK\x01\x02":
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[off:off + 46])
        nlen, elen, clen = f[10], f[11], f[12]
        name = cd[off + 46:off + 46 + nlen].decode()
        csize, usize, lho = f[8], f[9], f[16]
        if 0xFFFFFFFF in (csize, usize, lho) or f[13] == 0xFFFF:
            # ZIP64 拡張フィールド（タグ 0x0001）: センチネルの項目だけが
            # usize, csize, lho, disk の順に入っている
            extra = cd[off + 46 + nlen:off + 46 + nlen + elen]
            eo = 0
            while eo + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[eo:eo + 4])
                body = extra[eo + 4:eo + 4 + sz]
                if tag == 0x0001:
                    p = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                    break
                eo += 4 + sz
        entries[name] = {"method": f[4], "csize": csize, "usize": usize, "lho": lho}
        off += 46 + nlen + elen + clen
    print(prot, f"zip {total_size/1e9:.2f} GB, members {len(entries)}", flush=True)
    for rep in (1, 2, 3):
        tpr_name = f"{prot}_prod_R{rep}.tpr"
        xtc_name = f"{prot}_prod_R{rep}.xtc"
        tpr = fetch_member_head(url, entries[tpr_name], entries[tpr_name]["usize"])
        (out_dir / tpr_name).write_bytes(tpr)
        xtc = fetch_member_head(url, entries[xtc_name], XTC_RAW_TARGET)
        (out_dir / xtc_name).write_bytes(xtc)
        print(prot, f"R{rep}: tpr {len(tpr)} B, xtc head {len(xtc)} B", flush=True)
    (out_dir / "done.txt").write_text("ok\n")


if __name__ == "__main__":
    t0 = time.time()
    for prot in sys.argv[1:]:
        fetch_protein(prot)
    print(f"done, total {time.time()-t0:.1f}s", flush=True)
