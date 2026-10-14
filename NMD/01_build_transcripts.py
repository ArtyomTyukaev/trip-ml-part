#!/usr/bin/env python3
"""
Stage 1 – the transcript reference.

Turns the MANE BED12 (genomic exon structure) plus the MANE transcript FASTA into one row per
transcript holding everything the later stages need, already projected into TRANSCRIPT coordinates:
exon lengths, junction positions, CDS bounds.

Nothing in this stage knows about barcodes or about the reporter construct.

    python3 NMD/01_build_transcripts.py [--root DIR]

--root points at a directory that holds genuine_barcodes/, work/, logFC/ and output/. It defaults
to the folder of this script.

in   <root>/genuine_barcodes/MANE.GRCh38.v1.0.refseq_genomic.bed
     <root>/genuine_barcodes/MANE.GRCh38.v1.0.refseq_rna.fna
out  <root>/work/transcripts.tsv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import genomic_to_tx, parse_args, read_fasta

STOPS = ("TAA", "TAG", "TGA")

COLS = [
    "tx_id",          # RefSeq accession, e.g. NM_030937.6
    "chrom",
    "strand",
    "tx_start",       # genomic span of the transcript, 0-based half-open
    "tx_end",
    "n_exons",
    "tx_len",         # length of the mature mRNA = sum of block sizes = length of the FASTA record
    "cds_tx_start",   # CDS in TRANSCRIPT coordinates, half-open, stop codon included
    "cds_tx_end",
    "cds_last_exon",  # 1-based transcript-order index of the exon HOLDING THE STOP CODON. This is
                      # what "last exon" means in the rule – not the last exon of the transcript.
                      # The two differ in the 837 MANE genes (4.4%) that have an intron in the 3'UTR.
    "junctions",      # transcript positions of the seams, in transcript order, comma-separated
    "exons_genomic",  # "start-end,..." in ascending GENOMIC order (what stage 2 searches)
    "start_codon",    # first CDS codon, and 33 MANE transcripts are deliberately non-canonical
    "sel_tga",        # internal in-frame TGA positions (selenocysteines), transcript coords
]


def main():
    root, _ = parse_args()
    bed_path = os.path.join(root, "genuine_barcodes", "MANE.GRCh38.v1.0.refseq_genomic.bed")
    fna_path = os.path.join(root, "genuine_barcodes", "MANE.GRCh38.v1.0.refseq_rna.fna")
    out_path = os.path.join(root, "work", "transcripts.tsv")

    seqs = read_fasta(fna_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_in = n_out = 0
    bad_len = bad_cds_mod3 = bad_cds_stop = no_cds = cds_intronic = 0
    noncanonical, seleno, missing_seq = [], [], []
    intronless = utr3_intron = 0

    with open(bed_path) as fh, open(out_path, "w") as out:
        out.write("\t".join(COLS) + "\n")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            n_in += 1
            chrom, start, end, tx_id, strand = f[0], int(f[1]), int(f[2]), f[3], f[5]
            thick_start, thick_end, n_exons = int(f[6]), int(f[7]), int(f[9])
            sizes = [int(x) for x in f[10].rstrip(",").split(",")]
            starts = [int(x) for x in f[11].rstrip(",").split(",")]

            exons = [(start + s, start + s + l) for s, l in zip(starts, sizes)]
            total = sum(sizes)

            seq = seqs.get(tx_id)
            if seq is None:
                missing_seq.append(tx_id)
                continue
            if len(seq) != total:
                bad_len += 1
                continue
            if n_exons == 1:
                intronless += 1

            # junction positions, in transcript order
            order = exons if strand == "+" else list(reversed(exons))
            junctions, off = [], 0
            for a, b in order[:-1]:
                off += b - a
                junctions.append(off)

            # CDS bounds. thickEnd is exclusive, so the last CDS base is thickEnd - 1.
            if thick_start >= thick_end:
                no_cds += 1
                cds_tx_start = cds_tx_end = cds_last_exon = ""
                start_codon = ""
                sel = []
            else:
                lo = genomic_to_tx(exons, strand, total, thick_start)
                hi = genomic_to_tx(exons, strand, total, thick_end - 1)
                if lo is None or hi is None:
                    cds_intronic += 1
                    continue
                cds_tx_start, cds_tx_end = (lo, hi + 1) if strand == "+" else (hi, lo + 1)
                # the exon holding the stop codon: junctions are cumulative exon lengths, so exon k
                # (1-based) spans [junctions[k-2], junctions[k-1]) and the index is one more than
                # the number of junctions at or below the position
                cds_last_exon = 1 + sum(1 for j in junctions if j <= cds_tx_end - 1)
                cds = seq[cds_tx_start:cds_tx_end]
                if len(cds) % 3:
                    bad_cds_mod3 += 1
                if cds[-3:] not in STOPS:
                    bad_cds_stop += 1
                start_codon = cds[:3]
                if start_codon != "ATG":
                    noncanonical.append((tx_id, start_codon))
                # in-frame stops before the annotated one: selenoproteins read TGA through
                sel = [cds_tx_start + i for i in range(0, len(cds) - 3, 3)
                       if cds[i:i + 3] in STOPS]
                if sel:
                    seleno.append((tx_id, len(sel)))
                if cds_last_exon != n_exons:
                    utr3_intron += 1

            out.write("\t".join(str(x) for x in [
                tx_id, chrom, strand, start, end, n_exons, total,
                cds_tx_start, cds_tx_end, cds_last_exon,
                ",".join(str(j) for j in junctions),
                ",".join("%d-%d" % (a, b) for a, b in exons),
                start_codon,
                ",".join(str(p) for p in sel),
            ]) + "\n")
            n_out += 1

    print("stage 1 – transcript reference   (root %s)" % root)
    print("  BED12 records read        : %d" % n_in)
    print("  written to transcripts.tsv: %d" % n_out)
    print("  intronless (1 exon)       : %d" % intronless)
    print("  stop codon NOT in the last exon of the transcript (3'UTR intron): %d" % utr3_intron)
    print("  must be zero:")
    print("    FASTA length != block sum : %d" % bad_len)
    print("    no sequence in FASTA      : %d %s" % (len(missing_seq), missing_seq[:3]))
    print("    CDS bound inside an intron: %d" % cds_intronic)
    print("    CDS not divisible by 3    : %d" % bad_cds_mod3)
    print("    CDS not ending in a stop  : %d" % bad_cds_stop)
    print("    no CDS at all             : %d" % no_cds)
    print("  tolerated, not errors:")
    print("    non-canonical start codon : %d %s" % (len(noncanonical), noncanonical[:4]))
    print("    internal in-frame TGA     : %d %s" % (len(seleno), seleno[:4]))

    fatal = bad_len + cds_intronic + bad_cds_mod3 + bad_cds_stop + len(missing_seq)
    if fatal:
        print("\nFAILED: %d records violate an invariant" % fatal)
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
