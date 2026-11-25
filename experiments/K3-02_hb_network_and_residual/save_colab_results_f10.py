"""Save RESULT lines from a Colab batch output file into counts_f10/*.npz
(frame-10 counts, same format as extract_counts_f10.py).
Usage: save_colab_results_f10.py output.txt"""
import base64, io, json, sys
from pathlib import Path
import numpy as np
COLUMNS = ["n_out_donor_res", "n_in_donor_res", "n_out_acceptor_res", "n_in_acceptor_res",
           "n_hb_total", "n_partner_backbone", "n_partner_sidechain", "n_partner_water",
           "n_competing", "dist_ab_frame10", "bonded_frame10", "n_hb_donor_res_total", "n_hb_acceptor_res_total"]
OUT = Path("/home/hatch/workspace/k3_02/counts_f10")
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("RESULT "):
        rec = json.loads(line[len("RESULT "):])
        z = np.load(io.BytesIO(base64.b64decode(rec["counts_b64"])))
        np.savez_compressed(OUT / f"{rec['protein']}.npz", columns=np.array(COLUMNS),
                            premises=json.dumps(rec["premises"]),
                            **{k: z[k] for k in z.keys()})
        print("saved", rec["protein"], "premises",
              {r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
               for r, p in rec["premises"].items()})
    elif line.startswith("FAILED "):
        print(line)
