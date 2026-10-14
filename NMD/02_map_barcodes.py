#!/usr/bin/env python3
"""
Stage 2 – place every barcode on the transcript reference.

For each barcode: which MANE gene it sits in (same strand only), exonic or intronic, the exon/intron
index in TRANSCRIPT order, the insertion point in transcript coordinates and the region
(5utr / cds / 3utr).

Two things this stage deliberately does not do. It never looks at the CHX result, so placement is
recorded even for barcodes that turn out not to be expressed. And it ignores where inside an intron
the integration landed – that does not change the mature mRNA.

    python3 NMD/02_map_barcodes.py [--root DIR] [--libs a,b]

in   <root>/work/transcripts.tsv
     <root>/genuine_barcodes/<lib>.txt.retained_barcodes_genuine.bed
out  <root>/work/barcodes_<lib>.tsv
"""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import genomic_to_tx, parse_args, split_lib

BIN = 100_000

COLS = [
    "barcode", "chrom", "pos", "strand", "score",
    "in_gene",        # 1 only for a same-strand overlap with a MANE transcript
    "excluded_reason",  # intergenic / antisense / unplaced_contig – empty when in_gene is 1
    "tx_id",
    "n_tx_overlap",   # same-strand MANE transcripts containing this barcode, >1 means a tie was broken
    "loc",            # exonic / intronic
    "feat_idx",       # 1-based index of that exon or intron in TRANSCRIPT order
    "n_exons",
    "cds_last_exon",  # exon holding the stop codon – this is what "last exon" means in the rule
    "tx_pos",         # insertion point in transcript coordinates
    "region",         # 5utr / cds / 3utr
]


def load_transcripts(path):
    txs = {}
    index = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = dict(zip(hdr, line.rstrip("\n").split("\t")))
            t = {
                "tx_id": f["tx_id"],
                "chrom": f["chrom"],
                "strand": f["strand"],
                "tx_start": int(f["tx_start"]),
                "tx_end": int(f["tx_end"]),
                "n_exons": int(f["n_exons"]),
                "tx_len": int(f["tx_len"]),
                "cds_tx_start": int(f["cds_tx_start"]) if f["cds_tx_start"] else None,
                "cds_tx_end": int(f["cds_tx_end"]) if f["cds_tx_end"] else None,
                "cds_last_exon": f["cds_last_exon"],
                "junctions": [int(x) for x in f["junctions"].split(",")] if f["junctions"] else [],
                "exons": [tuple(int(y) for y in x.split("-")) for x in f["exons_genomic"].split(",")],
            }
            txs[t["tx_id"]] = t
            for b in range(t["tx_start"] // BIN, t["tx_end"] // BIN + 1):
                index[(t["chrom"], t["strand"])][b].append(t["tx_id"])
    return txs, index


def place(t, pos):
    """Where does a genomic position sit in this transcript? -> (loc, feat_idx, tx_pos)

    feat_idx is 1-based in TRANSCRIPT order, which on the minus strand is the reverse of the
    genomic order – that inversion is exactly the bug in the annotation's MANE_intron_ind.
    """
    n = t["n_exons"]
    for i, (a, b) in enumerate(t["exons"]):
        if a <= pos < b:
            idx = i if t["strand"] == "+" else n - 1 - i
            return "exonic", idx + 1, genomic_to_tx(t["exons"], t["strand"], t["tx_len"], pos)
    for i in range(n - 1):
        if t["exons"][i][1] <= pos < t["exons"][i + 1][0]:
            k = i if t["strand"] == "+" else n - 2 - i          # index into junctions
            return "intronic", k + 1, t["junctions"][k]
    return None, None, None                                     # inside the span but in no feature


def region_of(t, tx_pos, loc):
    """5utr / cds / 3utr for the insertion point.

    An intron whose upstream exon is still UTR and whose downstream exon already carries the ATG
    counts as UTR: the only thing that decides is whether the insertion point lies above the CDS
    start or below the CDS end, both in transcript coordinates.
    """
    cs, ce = t["cds_tx_start"], t["cds_tx_end"]
    if cs is None:
        return "noncoding"
    if loc == "intronic":
        if tx_pos <= cs:
            return "5utr"
        if tx_pos >= ce:
            return "3utr"
        return "cds"
    if tx_pos < cs:
        return "5utr"
    if tx_pos >= ce:
        return "3utr"
    return "cds"


def main():
    root, libs = parse_args()
    txs, index = load_transcripts(os.path.join(root, "work", "transcripts.tsv"))
    annotated_chroms = {t["chrom"] for t in txs.values()}
    print("stage 2 – barcode placement   (root %s)" % root)
    print("  transcripts loaded: %d on %d chromosomes" % (len(txs), len(annotated_chroms)))

    for spec in libs:
        lib = split_lib(spec)[0]
        src = os.path.join(root, "genuine_barcodes", "%s.txt.retained_barcodes_genuine.bed" % lib)
        dst = os.path.join(root, "work", "barcodes_%s.tsv" % lib)
        stat = Counter()
        ties = 0
        seen = set()
        with open(src) as fh, open(dst, "w") as out:
            out.write("\t".join(COLS) + "\n")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                chrom, pos, barcode, score, strand = f[0], int(f[1]), f[3], f[4], f[5]
                stat["total"] += 1
                # everything downstream joins by barcode, so a repeated barcode would silently
                # overwrite a row instead of being counted twice
                if barcode in seen:
                    sys.exit("barcode %s appears twice in %s – joining by barcode is unsafe"
                             % (barcode, os.path.basename(src)))
                seen.add(barcode)

                same = [txs[i] for i in index.get((chrom, strand), {}).get(pos // BIN, [])
                        if txs[i]["tx_start"] <= pos < txs[i]["tx_end"]]
                other = "-" if strand == "+" else "+"
                anti = [txs[i] for i in index.get((chrom, other), {}).get(pos // BIN, [])
                        if txs[i]["tx_start"] <= pos < txs[i]["tx_end"]]

                if not same:
                    # "intergenic" is a claim about the annotation, so it may only be made where
                    # there IS annotation. The barcode BEDs carry ~110 integrations per library on
                    # unplaced chrUn_* scaffolds that MANE does not cover at all. Those are unknown,
                    # not intergenic.
                    if chrom not in annotated_chroms:
                        why = "unplaced_contig"
                    elif anti:
                        why = "antisense"
                    else:
                        why = "intergenic"
                    stat[why] += 1
                    out.write("\t".join([barcode, chrom, str(pos), strand, score,
                                         "0", why, "", "0", "", "", "", "", "", ""]) + "\n")
                    continue

                if len(same) > 1:
                    ties += 1
                # longest genomic span wins, with deterministic fallbacks so the choice never depends
                # on file order
                t = sorted(same, key=lambda x: (-(x["tx_end"] - x["tx_start"]), -x["tx_len"],
                                                x["tx_id"]))[0]
                loc, idx, tx_pos = place(t, pos)
                if loc is None:                       # cannot happen for a well-formed BED12
                    stat["unplaced"] += 1
                    continue
                region = region_of(t, tx_pos, loc)
                stat["in_gene"] += 1
                stat["%s/%s" % (loc, region)] += 1
                out.write("\t".join([barcode, chrom, str(pos), strand, score,
                                     "1", "", t["tx_id"], str(len(same)),
                                     loc, str(idx), str(t["n_exons"]), t["cds_last_exon"],
                                     str(tx_pos), region]) + "\n")

        print("\n  %s -> work/barcodes_%s.tsv" % (lib, lib))
        print("    barcodes            : %d" % stat["total"])
        print("    same-strand in gene : %d (%.1f%%)"
              % (stat["in_gene"], 100.0 * stat["in_gene"] / stat["total"]))
        print("    antisense in a gene : %d" % stat["antisense"])
        print("    intergenic          : %d" % stat["intergenic"])
        print("    unplaced contig     : %d (chrUn_*, no MANE annotation there at all)"
              % stat["unplaced_contig"])
        print("    overlap ties broken : %d" % ties)
        print("    breakdown of in-gene barcodes:")
        for k in ("intronic/cds", "intronic/5utr", "intronic/3utr",
                  "exonic/cds", "exonic/5utr", "exonic/3utr"):
            print("      %-16s %6d" % (k, stat[k]))
        if stat["unplaced"]:
            print("    UNPLACED (should be 0): %d" % stat["unplaced"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
