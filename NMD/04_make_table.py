#!/usr/bin/env python3
"""
Stage 4 – the deliverable.

Joins placement (stage 2) + verdict (stage 3) + the CHX readout, BY BARCODE, into one table per
library. Every barcode of the genuine BED appears, nothing is filtered.

The DESeq2 result is reused as is. The annotation columns of the logFC table are NOT reused – gene
identity, exonic/intronic placement and the feature index all come from our own re-derivation,
because those columns disagree with the MANE v1.0 BED12. The one thing taken from them is the
accession -> gene symbol mapping, which is just a lookup.

    python3 NMD/04_make_table.py [--root DIR] [--libs a,b]

in   <root>/work/barcodes_<lib>.tsv, <root>/work/nmd_<lib>.tsv, <root>/logFC/CHX_<chx>.txt
out  <root>/output/nmd_table_<lib>.tsv
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_args, split_lib

CHX = ["CHXbaseMean", "CHXlog2FoldChange", "CHXlfcSE", "CHXpadj"]

FROM_BC = ["chrom", "pos", "strand", "score",
           "in_gene", "excluded_reason", "tx_id", "n_tx_overlap",
           "loc", "feat_idx", "n_exons", "cds_last_exon", "tx_pos", "region"]
FROM_NMD = ["insert_frame", "insert_len", "bc_has_stop", "ptc_source",
            "ptc_tx_pos", "last_junction_pos", "n_junc_downstream", "dist_to_last_junction",
            "dist_gt_50", "selenocysteine", "note"]

COLS = (["barcode"] + FROM_BC[:4] + ["tx_id", "gene_symbol"]
        + ["in_gene", "excluded_reason", "n_tx_overlap", "loc", "feat_idx", "n_exons",
           "cds_last_exon",
           "tx_pos", "region"]
        + FROM_NMD + CHX
        + ["nmd_expected", "verdict_reason", "category"])


def load_tsv(path, key="barcode"):
    """Keyed by barcode, and it refuses to collapse a duplicate key.

    Every join in this pipeline is by barcode. A duplicate would silently drop a row here rather
    than produce a visible error, which is exactly the failure mode that is hardest to notice.
    """
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r[key] in out:
                sys.exit("%s: %s appears twice – refusing to join" % (os.path.basename(path),
                                                                       r[key]))
            out[r[key]] = r
    return out


def na(v):
    return v in ("", "NA", "NaN", "nan", "NULL")


def main():
    root, libs = parse_args()
    work = os.path.join(root, "work")
    outdir = os.path.join(root, "output")
    os.makedirs(outdir, exist_ok=True)
    symbol = {}

    print("stage 4 – final tables   (root %s)" % root)
    for spec in libs:
        lib, chx = split_lib(spec)
        bc = load_tsv(os.path.join(work, "barcodes_%s.tsv" % lib))
        nmd = load_tsv(os.path.join(work, "nmd_%s.tsv" % lib))

        chxrow = {}
        with open(os.path.join(root, "logFC", "CHX_%s.txt" % chx)) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["Barcode"] in chxrow:
                    sys.exit("CHX_%s.txt: barcode %s appears twice" % (chx, r["Barcode"]))
                chxrow[r["Barcode"]] = r
                if not na(r.get("MANE_all")) and not na(r.get("MANE_all_symbol")):
                    symbol.setdefault(r["MANE_all"], r["MANE_all_symbol"])

        stat = Counter()
        dst = os.path.join(outdir, "nmd_table_%s.tsv" % lib)
        with open(dst, "w") as out:
            out.write("\t".join(COLS) + "\n")
            for b, brow in bc.items():
                nrow = nmd[b]
                crow = chxrow.get(b)
                if crow is None:
                    stat["absent from logFC"] += 1

                row = {"barcode": b, "gene_symbol": symbol.get(brow["tx_id"], "")}
                row.update({k: brow[k] for k in FROM_BC})
                row.update({k: nrow[k] for k in FROM_NMD})
                row["nmd_expected"] = nrow["nmd_expected"]
                row["verdict_reason"] = nrow["verdict_reason"]
                for c in CHX:
                    v = crow[c] if crow else ""
                    row[c] = "" if na(v) else v

                if na(row["CHXlog2FoldChange"]):
                    cat = 1
                elif brow["in_gene"] != "1":
                    cat = 2
                elif na(nrow["nmd_expected"]):
                    cat = ""                      # in a gene but no verdict – should not happen
                elif nrow["nmd_expected"] == "1":
                    cat = 4
                else:
                    cat = 3
                row["category"] = cat
                stat["category %s" % cat] += 1
                if cat in (3, 4):
                    stat["  via %s" % nrow["verdict_reason"]] += 1
                out.write("\t".join(str(row[c]) for c in COLS) + "\n")

        print("\n  %s -> output/nmd_table_%s.tsv   (%d columns)" % (lib, lib, len(COLS)))
        print("    rows: %d" % sum(v for k, v in stat.items() if k.startswith("category")))
        for k in sorted(stat, key=lambda x: (not x.startswith("category"), x)):
            print("    %-28s %6d" % (k, stat[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
