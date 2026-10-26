#!/usr/bin/env python3
"""Name a network for publication: tsumi-<the first 12 hex digits of its SHA-256>.nnue, the scheme `make net` checks.

    python3 tools/release_net.py nets/best.nnue [--out DIR] [--set-default]

Copies the network, and the .pt checkpoint beside it if there is one, to DIR (default: the current directory) under
that name, and prints their SHA-256 sums.  --set-default makes it the engine's default network (NNUE::DEFAULT_NET
in src/nnue.h), the one `make net` downloads and a release ships.  Then upload both files to the "networks" release
(README: Releasing).
"""
import argparse
import hashlib
import os
import re
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", help="the network (.nnue)")
    ap.add_argument("--out", default=".", help="where to put the named copies (default: the current directory)")
    ap.add_argument("--set-default", action="store_true", help="make it the engine's default network in src/nnue.h")
    args = ap.parse_args()
    if not os.path.isfile(args.net):
        sys.exit(f"{args.net} not found")
    name = f"tsumi-{sha256(args.net)[:12]}.nnue"
    for src, dst in ((args.net, os.path.join(args.out, name)), (args.net + ".pt", os.path.join(args.out, name + ".pt"))):
        if not os.path.isfile(src):
            print(f"(no checkpoint {src}: only the network is published)")
            continue
        if not os.path.exists(dst) or not os.path.samefile(src, dst):
            shutil.copyfile(src, dst)
        print(f"{sha256(dst)}  {dst}")
    if args.set_default:
        header = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "nnue.h")
        with open(header) as f:
            text = f.read()
        new, n = re.subn(r'(DEFAULT_NET\[\] = ")[^"]*(")', rf"\g<1>{name}\g<2>", text)
        if n != 1:
            sys.exit(f"{header}: DEFAULT_NET not found")
        with open(header, "w") as f:
            f.write(new)
        print(f"src/nnue.h: DEFAULT_NET = {name} (rebuild the engine)")


if __name__ == "__main__":
    main()
