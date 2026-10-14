#!/usr/bin/env python3
"""
Stage 3 – the verdict.

The decision tree, in order. The first matching row wins:

    outside a MANE gene / antisense   ->  not analysed
    any UTR (5' or 3', exon or intron) -> no NMD   – no PTC is created, the rule does not apply
    last exon                          -> no NMD   – a PTC exists but no junction remains below it
    internal coding exon               -> NMD      – a PTC exists and junctions remain below it
    intron inside the CDS              -> computed – build the mRNA, find the stop, measure.
                                          Splits in two: `cds_intron` when the stop really is
                                          premature, `no_ptc_frameshift` when the frameshift
                                          carries termination to or past the host's own stop.

Only the last branch touches sequence. The exonic branches are decided structurally because the
splicing outcome of an integration inside a host exon cannot be derived from coordinates.

Nothing about the inserted exon is hard-coded: its length and the position of the barcode slot are
read from the N-run in exon_and_5arm_sequence.txt.

    python3 NMD/03_call_nmd.py [--root DIR] [--libs a,b]
                                   [--insert PATH] [--out-suffix S] [--extra-junction N]

All three optional flags default to absent, and with none of them the run is byte-identical to what
it was. They exist for IntronTRIP, whose cassette carries its own intron and therefore reaches the
mRNA in TWO forms of different length, one per library and one per form:

    --insert PATH         the insert file to use instead of exon_and_5arm_sequence.txt. May contain
                          `{lib}`, which is filled with the library name – the eight IntronTRIP
                          reporters carry a different donor exon each, so one run over all eight
                          needs eight different inserts.
    --out-suffix S        write work/nmd_<lib><S>.tsv instead of work/nmd_<lib>.tsv, so the spliced
                          and the unspliced run of the same library do not overwrite each other.
    --extra-junction N    the insert itself carries an exon-exon junction at offset N. True of the
                          SPLICED IntronTRIP form only, where excising the reporter intron joins the
                          two halves of the cassette exon and deposits an EJC there. It never
                          becomes last_junction_pos, so it moves n_junc_downstream and nothing else.

in   <root>/work/transcripts.tsv, <root>/work/barcodes_<lib>.tsv
     <root>/genuine_barcodes/exon_and_5arm_sequence.txt   (or --insert)
     <root>/genuine_barcodes/MANE.GRCh38.v1.0.refseq_rna.fna
out  <root>/work/nmd_<lib>.tsv                            (or with --out-suffix)
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_args, read_fasta, split_lib

STOPS = ("TAA", "TAG", "TGA")
THRESHOLD = 50

COLS = [
    "barcode",
    "nmd_expected",        # 1 / 0 – the verdict, filled for every in-gene barcode
    "verdict_reason",      # which branch of the tree decided it
    "insert_frame",        # 0/1/2 – phase the insert is read in, computed rows only
    "insert_len",
    "bc_has_stop",         # the barcode's own 18 nt contain an in-frame stop
    "ptc_source",          # which part of the mRNA the PTC came from
    "ptc_tx_pos",          # PTC position in the MODIFIED transcript
    "last_junction_pos",   # last junction between CODING exons of the modified transcript
    "n_junc_downstream",   # coding-exon junctions at or below the PTC
    "dist_to_last_junction",   # last_junction - (ptc + 3), the quantity the 50-nt rule tests
    "dist_gt_50",
    "selenocysteine",      # the first in-frame stop was a read-through TGA, search continued
    "note",
]


def read_insert(path):
    """The 400 nt that reach the mRNA, plus the barcode slot found as the run of Ns."""
    seq = open(path).read().strip().upper()
    i = seq.find("N")
    if i < 0:
        sys.exit("no N run in the insert file – cannot locate the barcode slot")
    j = i
    while j < len(seq) and seq[j] == "N":
        j += 1
    if "N" in seq[j:]:
        sys.exit("more than one N run in the insert file")
    return seq, i, j - i


def load_transcripts(path):
    txs = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = dict(zip(hdr, line.rstrip("\n").split("\t")))
            txs[f["tx_id"]] = {
                "n_exons": int(f["n_exons"]),
                "tx_len": int(f["tx_len"]),
                "cds_tx_start": int(f["cds_tx_start"]) if f["cds_tx_start"] else None,
                "cds_tx_end": int(f["cds_tx_end"]) if f["cds_tx_end"] else None,
                "cds_last_exon": int(f["cds_last_exon"]) if f["cds_last_exon"] else None,
                "junctions": [int(x) for x in f["junctions"].split(",")] if f["junctions"] else [],
                "sel_tga": [int(x) for x in f["sel_tga"].split(",")] if f["sel_tga"] else [],
            }
    return txs


def blank(**kw):
    row = {c: "" for c in COLS}
    row.update(kw)
    return row


def segment(off, slot, bc_len, insert_len):
    """Label the part of the insert an offset falls in. Labels only – never affects a verdict."""
    if off < slot:
        return "insert_5p"
    if off < slot + bc_len:
        return "barcode"
    return "insert_3p"


def compute(t, tx_pos, seq, insert, slot, bc_len, extra_junction=None):
    """The one branch that reads sequence: an integration in an intron inside the CDS."""
    p = tx_pos
    L = len(insert)
    new_seq = seq[:p] + insert + seq[p:]

    # The rule is about the last junction between CODING exons, not the last junction of the
    # transcript. They differ in the 837 MANE genes (4.4%) with an intron in the 3'UTR, where the
    # transcript's final junction sits behind the stop codon and joins two untranslated exons.
    # Coding exons are 1..cds_last_exon, so the junctions among them are the first
    # cds_last_exon - 1 entries.
    coding_junc = t["junctions"][:t["cds_last_exon"] - 1] if t["cds_last_exon"] else []
    # the host junction at p is split in two: one at p, one at p + L. The cassette exon lies
    # inside the CDS (this branch only runs for cds_intron rows), so both are coding junctions.
    # `extra_junction` adds one MORE, inside the insert itself. The IntronTRIP cassette carries its
    # own intron, and in the spliced form excising it joins the two halves of the cassette exon,
    # which deposits an EJC there as well. It can never be the LAST junction – p + L is always
    # further down – so it moves n_junc_downstream and leaves the verdict alone.
    inside = [p + extra_junction] if extra_junction is not None else []
    new_junc = sorted([j if j < p else j + L for j in coding_junc] + [p] + inside)
    cds_start = t["cds_tx_start"]
    if cds_start >= p:                      # cannot happen for region == cds, kept for generality
        cds_start += L
    sel = set(s if s < p else s + L for s in t["sel_tga"])

    # boundaries a codon can straddle, in modified-transcript coordinates
    bounds = (p, p + slot, p + slot + bc_len, p + L)

    stop = None
    seleno_hit = 0
    for i in range(cds_start, len(new_seq) - 2, 3):
        if new_seq[i:i + 3] in STOPS:
            if i in sel:                    # selenocysteine, read through
                seleno_hit = 1
                continue
            stop = i
            break

    # does the barcode itself carry an in-frame stop, whether or not it is the first?
    bc_lo, bc_hi = p + slot, p + slot + bc_len
    bc_stop = 0
    for i in range(cds_start, len(new_seq) - 2, 3):
        if bc_lo <= i < bc_hi and new_seq[i:i + 3] in STOPS:
            bc_stop = 1
            break
        if i >= bc_hi:
            break

    frame = (-(p - cds_start)) % 3
    common = dict(insert_frame=frame, insert_len=L, bc_has_stop=bc_stop,
                  selenocysteine=seleno_hit)

    if stop is None:
        return blank(nmd_expected=0, verdict_reason="no_ptc_frameshift", ptc_source="none",
                     note="no in-frame stop anywhere downstream of the CDS start", **common)

    # A codon straddling a boundary belongs to neither side, so it gets its own label. In frame 1
    # the commonest PTC of all is host "TA" + the insert's first base "G" = TAG, right on the seam.
    straddles = any(stop < b < stop + 3 for b in bounds)
    if stop < p:
        src = "host_insert_seam" if straddles else "host_upstream"
    elif stop < p + L:
        src = segment(stop - p, slot, bc_len, L) + ("_seam" if straddles else "")
    else:
        src = "host_downstream" + ("_seam" if straddles else "")

    # Is the stop premature at all? Below the insert the host is read out of frame, and that can
    # carry the ribosome PAST the position where the host's own stop codon sits. Termination is then
    # late, not early: no premature stop is created and the 50-nt rule has nothing to act on. This
    # can only happen on the computed branch, since only there is the insert read through.
    normal_stop = t["cds_tx_end"] - 3
    if normal_stop >= p:
        normal_stop += L
    if stop >= normal_stop:
        return blank(nmd_expected=0, verdict_reason="no_ptc_frameshift", ptc_source=src,
                     ptc_tx_pos=stop, last_junction_pos=max(new_junc),
                     note="frameshift carries termination to or past the host's own stop codon",
                     **common)

    last_j = max(new_junc)
    dist = last_j - (stop + 3)
    n_down = sum(1 for j in new_junc if j >= stop + 3)
    nmd = 1 if dist > THRESHOLD else 0
    return blank(nmd_expected=nmd, verdict_reason="cds_intron", ptc_source=src,
                 ptc_tx_pos=stop, last_junction_pos=last_j,
                 n_junc_downstream=n_down, dist_to_last_junction=dist,
                 dist_gt_50=nmd, **common)


def take_flag(argv, name, default=None):
    """Pull `--name VALUE` out of argv and return (value, remaining argv).

    Done here rather than in common.parse_args on purpose: stages 1, 2 and 4 must keep rejecting
    unknown arguments instead of silently accepting three flags that mean nothing to them.
    """
    if name not in argv:
        return default, argv
    i = argv.index(name)
    if i + 1 >= len(argv):
        sys.exit("%s needs a value" % name)
    return argv[i + 1], argv[:i] + argv[i + 2:]


def main():
    argv = sys.argv[1:]
    insert_arg, argv = take_flag(argv, "--insert")
    out_suffix, argv = take_flag(argv, "--out-suffix", "")
    extra_arg, argv = take_flag(argv, "--extra-junction")
    extra_junction = int(extra_arg) if extra_arg is not None else None
    root, libs = parse_args(argv)

    gb = os.path.join(root, "genuine_barcodes")
    work = os.path.join(root, "work")
    insert_pattern = insert_arg or os.path.join(gb, "exon_and_5arm_sequence.txt")

    print("stage 3 – the verdict   (root %s)" % root)
    if extra_junction is not None:
        print("  extra exon-exon junction inside the insert at offset %d" % extra_junction)

    txs = load_transcripts(os.path.join(work, "transcripts.tsv"))
    seqs = read_fasta(os.path.join(gb, "MANE.GRCh38.v1.0.refseq_rna.fna"))

    for spec in libs:
        lib = split_lib(spec)[0]
        # A pattern without `{lib}` is returned unchanged by format(), so the single-insert case
        # keeps working. With it, one run covers libraries whose inserts differ.
        insert_path = insert_pattern.format(lib=lib)
        insert_tpl, slot, bc_len = read_insert(insert_path)
        if extra_junction is not None and not 0 < extra_junction < len(insert_tpl):
            sys.exit("--extra-junction %d is outside the %d nt insert %s"
                     % (extra_junction, len(insert_tpl), insert_path))
        print("\n  %s insert: %s" % (lib, insert_path))
        print("    %d nt, barcode slot at %d-%d (%d nt), len %% 3 = %d"
              % (len(insert_tpl), slot, slot + bc_len, bc_len, len(insert_tpl) % 3))

        stat = Counter()
        src_stat = Counter()
        dists = []
        with open(os.path.join(work, "barcodes_%s.tsv" % lib)) as fh, \
             open(os.path.join(work, "nmd_%s%s.tsv" % (lib, out_suffix)), "w") as out:
            hdr = fh.readline().rstrip("\n").split("\t")
            out.write("\t".join(COLS) + "\n")
            for line in fh:
                b = dict(zip(hdr, line.rstrip("\n").split("\t")))
                bc = b["barcode"]

                if b["in_gene"] != "1":
                    row = blank(barcode=bc, verdict_reason="not_analysed")
                elif b["region"] in ("5utr", "3utr"):
                    row = blank(barcode=bc, nmd_expected=0, verdict_reason="utr",
                                note="insertion in 5'UTR" if b["region"] == "5utr"
                                     else "insertion in 3'UTR")
                elif b["region"] == "noncoding":
                    row = blank(barcode=bc, verdict_reason="not_analysed",
                                note="host transcript has no annotated CDS")
                elif b["loc"] == "exonic":
                    # "last exon" means the exon HOLDING THE STOP CODON, i.e. where translation
                    # terminates – not the last exon of the transcript. The two differ in the 4.4%
                    # of MANE genes with an intron in the 3'UTR, and there the stop-codon exon is
                    # the one that matters.
                    if int(b["feat_idx"]) == int(b["cds_last_exon"]):
                        row = blank(barcode=bc, nmd_expected=0, verdict_reason="last_exon",
                                    note="insertion is in the exon where translation terminates")
                    else:
                        row = blank(barcode=bc, nmd_expected=1, verdict_reason="internal_exon",
                                    note="coding exon upstream of the one holding the stop codon")
                else:
                    t = txs[b["tx_id"]]
                    seq = seqs[b["tx_id"]]
                    if len(bc) != bc_len:
                        sys.exit("barcode %s is %d nt but the slot is %d nt – refusing to guess"
                                 % (bc, len(bc), bc_len))
                    # an N or an ambiguity code would never match a stop codon, so the search would
                    # quietly run past a real PTC instead of failing
                    if set(bc) - set("ACGT"):
                        sys.exit("barcode %s is not plain ACGT – a stop search over it is unsafe"
                                 % bc)
                    insert = insert_tpl[:slot] + bc + insert_tpl[slot + bc_len:]
                    row = compute(t, int(b["tx_pos"]), seq, insert, slot, bc_len,
                                  extra_junction)
                    row["barcode"] = bc
                    src_stat[row["ptc_source"]] += 1
                    if row["dist_to_last_junction"] != "":
                        dists.append(row["dist_to_last_junction"])

                stat[row["verdict_reason"]] += 1
                if row["nmd_expected"] != "":
                    stat["nmd=%s" % row["nmd_expected"]] += 1
                out.write("\t".join(str(row[c]) for c in COLS) + "\n")

        print("    -> work/nmd_%s%s.tsv" % (lib, out_suffix))
        for k in ("not_analysed", "utr", "last_exon", "internal_exon", "cds_intron",
                  "no_ptc_frameshift"):
            print("    %-14s %6d" % (k, stat[k]))
        print("    verdict: NMD expected %d, not expected %d"
              % (stat["nmd=1"], stat["nmd=0"]))
        print("    PTC source among the %d cds_intron rows:" % stat["cds_intron"])
        for k, v in src_stat.most_common():
            print("      %-24s %6d" % (k, v))
        if dists:
            dists.sort()
            n = len(dists)
            print("    dist to last junction: min %d, 5%% %d, median %d, max %d"
                  % (dists[0], dists[n // 20], dists[n // 2], dists[-1]))
            print("    rows the 50-nt threshold actually decides (0 < dist <= 50): %d"
                  % sum(1 for d in dists if 0 < d <= THRESHOLD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
