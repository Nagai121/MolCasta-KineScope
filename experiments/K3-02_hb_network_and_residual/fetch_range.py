"""Range-request fetcher for ATLAS total zips: downloads only the zip
central directory, the TPR members (full) and a prefix of each XTC
member (enough for frames 0..100), via HTTP Range + raw-deflate
streaming with early stop. Returns the extracted directory."""
import struct, urllib.request, zlib
from pathlib import Path

BASE = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total"
UA = {"User-Agent": "MiniTopol/0.2 (ATLAS data client)", "Accept": "application/zip"}

def _get_range(url, start, end=None):
    h = dict(UA)
    h["Range"] = f"bytes={start}-" + ("" if end is None else str(end))
    req = urllib.request.Request(url, headers=h)
    return urllib.request.urlopen(req, timeout=300)

def zip_members(url):
    """Return (total_size, {name: (local_hdr_off, comp_size, raw_size, method)}).
    Handles zip64 archives (sentinel values in the classic EOCD / CD)."""
    with _get_range(url, 0, 0) as r:
        total = int(r.headers["Content-Range"].split("/")[-1])
    tail_len = min(262144, total)
    with _get_range(url, total - tail_len, total - 1) as r:
        tail = r.read()
    tail_base = total - tail_len
    i = tail.rfind(b"PK\x05\x06")
    assert i >= 0, "EOCD not found"
    _, _, _, _, nrec, cdsize, cdoff, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if nrec == 0xFFFF or cdsize == 0xFFFFFFFF or cdoff == 0xFFFFFFFF:
        j = tail.rfind(b"PK\x06\x07")
        assert j >= 0, "zip64 locator not found"
        z64off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        with _get_range(url, z64off, z64off + 55) as r:
            z64 = r.read()
        assert z64[:4] == b"PK\x06\x06", "bad zip64 EOCD"
        nrec = struct.unpack("<Q", z64[32:40])[0]
        cdsize = struct.unpack("<Q", z64[40:48])[0]
        cdoff = struct.unpack("<Q", z64[48:56])[0]
    with _get_range(url, cdoff, cdoff + cdsize - 1) as r:
        cd = r.read()
    members = {}
    pos = 0
    for _ in range(nrec):
        f = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos:pos + 46])
        nlen, elen, comlen = f[10], f[11], f[12]
        name = cd[pos + 46:pos + 46 + nlen].decode()
        extra = cd[pos + 46 + nlen:pos + 46 + nlen + elen]
        raw, comp, hdr_off = f[9], f[8], f[16]
        if raw == 0xFFFFFFFF or comp == 0xFFFFFFFF or hdr_off == 0xFFFFFFFF:
            vals = []
            ep = 0
            while ep + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[ep:ep + 4])
                if tag == 0x0001:
                    vals = [struct.unpack("<Q", extra[ep + 4 + 8 * k:ep + 12 + 8 * k])[0]
                            for k in range(sz // 8)]
                    break
                ep += 4 + sz
            vi = 0
            if raw == 0xFFFFFFFF:
                raw = vals[vi]; vi += 1
            if comp == 0xFFFFFFFF:
                comp = vals[vi]; vi += 1
            if hdr_off == 0xFFFFFFFF:
                hdr_off = vals[vi]; vi += 1
        members[name] = (hdr_off, comp, raw, f[4])
        pos += 46 + nlen + elen + comlen
    return total, members

def fetch_member(url, hdr_off, raw_target, out_path):
    """Stream-decompress one member starting at its local header, stopping
    after raw_target decompressed bytes (or member end). Returns bytes written."""
    with _get_range(url, hdr_off, hdr_off + 4095) as r:
        hdr = r.read()
    assert hdr[:4] == b"PK\x03\x04", "bad local header"
    nlen, elen = struct.unpack("<HH", hdr[26:30])
    data_start = hdr_off + 30 + nlen + elen
    dec = zlib.decompressobj(-15)
    written = 0
    with _get_range(url, data_start) as r, open(out_path, "wb") as f:
        while written < raw_target:
            block = r.read(1024 * 1024)
            if not block:
                break
            data = dec.decompress(block, raw_target - written)
            f.write(data)
            written += len(data)
            if dec.eof:
                break
    return written

def fetch_protein(protein, dest_dir, xtc_raw_target=48 * 1024 * 1024):
    dest = Path(dest_dir) / protein
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{protein}"
    _, members = zip_members(url)
    got = {}
    for r in (1, 2, 3):
        for ext, target in (("tpr", None), ("xtc", xtc_raw_target)):
            name = f"{protein}_prod_R{r}.{ext}"
            hdr_off, comp, raw, method = members[name]
            assert method == 8, (name, method)
            w = fetch_member(url, hdr_off, raw if target is None else min(target, raw),
                             dest / name)
            got[name] = w
    return dest, got
