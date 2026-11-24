"""Reserve a future evaluation panel using sequence and structure metadata only."""
import csv
import hashlib
import json
from pathlib import Path

from Bio.Align import PairwiseAligner


def main():
    path=Path("atlas-info.tsv")
    with path.open(encoding="utf-8",newline="") as h:
        catalog=list(csv.DictReader(h,delimiter="\t"))
    used={p.name for p in Path("atlas-workspace").iterdir() if p.is_dir()}
    known=[r for r in catalog if r["PDB"] in used and r.get("sequence")]
    banned={r["UniProt"] for r in known}
    pool=[r for r in catalog if r.get("sequence") and r["PDB"] not in used and
          r["UniProt"] not in banned and r["UniProt"] not in ("","-") and
          "virus" not in r["organism"].lower() and "SARS" not in r["organism"].upper() and r["non_redundant_protein"].lower()=="true" and
          66<=int(r["length"])<=145]
    by_length=sorted(pool,key=lambda p:(int(p["length"]),p["PDB"]))
    split=[by_length[len(by_length)*i//8:len(by_length)*(i+1)//8] for i in range(8)]
    align=PairwiseAligner(mode="global")
    align.match_score=1.
    align.mismatch_score=-.5
    align.open_gap_score=-3.
    align.extend_gap_score=-.1
    selected=[]
    for batch in split:
        ranked=sorted(batch,key=lambda p:hashlib.sha256(("GA-351/"+p["PDB"]).encode()).digest())
        for r in ranked:
            s=r["sequence"]
            similarities=[]
            for q in (*known,*selected):
                t=q["sequence"]
                if min(len(s),len(t))/max(len(s),len(t))<.65:
                    continue
                count=align.align(s,t)[0].counts().identities
                similarities.append(count/min(len(s),len(t)))
            max_identity=max(similarities,default=0.)
            if max_identity<.39:
                selected.append(dict(r,max_identity=max_identity))
                break
    if len(selected)!=8:
        raise RuntimeError("Too few eligible metadata bins")
    report=dict(dataset="ATLAS",purpose="future confirmation only",
                rule="outside all local protein workspaces, annotated distinct UniProt, nonviral; nonredundant ATLAS metadata; "
                     "eight length bins, hashed ordering; global sequence identity<.39 against local "
                     "and selected proteins of comparable length (shorter/longer>=.65)",
                proteins=[dict(id=r["PDB"],uniprot=r["UniProt"],length=int(r["length"]),
                               max_identity=r["max_identity"]) for r in selected])
    output=Path("runs/ga351/reserved.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
