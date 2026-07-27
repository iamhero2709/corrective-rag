"""
End-to-end demo on a toy corpus.
Run:  python demo.py  or  rag demo
"""

import sys
from src.cli import cmd_demo


class Args:
    verbose = False
    model = ""
    no_quant = False
    hybrid = False


if __name__ == "__main__":
    args = Args()
    args.verbose = "-v" in sys.argv or "--verbose" in sys.argv
    for i, a in enumerate(sys.argv):
        if a in ("-m", "--model") and i + 1 < len(sys.argv):
            args.model = sys.argv[i + 1]
    cmd_demo(args)
