"""Shared bits of the NMD pipeline: argument parsing and the two coordinate helpers.

Everything here is deliberately tiny. Every path hangs off a --root, which defaults to the folder of
these scripts, so the same code can run over any directory laid out the same way.
"""

import os
import sys

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIBS = ["5ss_noP", "CPA_noP"]


def parse_args(argv=None):
    """-> (root, libs). libs entries may be 'lib' or 'lib:chx_suffix'."""
    root, libs = DEFAULT_ROOT, list(DEFAULT_LIBS)
    a = sys.argv[1:] if argv is None else argv
    i = 0
    while i < len(a):
        if a[i] == "--root" and i + 1 < len(a):
            root = os.path.abspath(a[i + 1])
            i += 2
        elif a[i] == "--libs" and i + 1 < len(a):
            libs = [x for x in a[i + 1].split(",") if x]
            i += 2
        else:
            sys.exit("usage: %s [--root DIR] [--libs a,b]" % os.path.basename(sys.argv[0]))
    return root, libs


def split_lib(spec):
    """'5ss_noP:5ss' -> ('5ss_noP', '5ss'), '5ss_noP' -> ('5ss_noP', '5ss')."""
    if ":" in spec:
        lib, chx = spec.split(":", 1)
        return lib, chx
    return spec, spec[:-4] if spec.endswith("_noP") else spec


def genomic_to_tx(exons, strand, total, pos):
    """Genomic position -> offset in the mature mRNA, or None if the position is intronic.

    `exons` is in ascending genomic order. On the minus strand transcript coordinates run the other
    way, hence the mirror – this is the single place that inversion is implemented.
    """
    off = 0
    for a, b in exons:
        if a <= pos < b:
            fwd = off + (pos - a)
            return fwd if strand == "+" else total - 1 - fwd
        off += b - a
    return None


def read_fasta(path):
    """accession -> uppercased sequence. Header is '>NM_000014.6 description'."""
    seqs, acc, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(buf).upper()
                acc = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(buf).upper()
    return seqs
