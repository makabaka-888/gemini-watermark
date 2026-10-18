# -*- coding: utf-8 -*-
"""Collect the chosen result for every file into the delivery folder.

Usage: python assemble.py <choices.tsv> <dest_dir>
choices.tsv: name<TAB>source path. A source equal to "-" means "no change
needed": the original is copied byte-for-byte (no re-encode).
Copies only; nothing is deleted. Prints a count summary.
"""
import os
import shutil
import sys


def main(choices, dest):
    os.makedirs(dest, exist_ok=True)
    n = kept = 0
    for line in open(choices, encoding="utf-8-sig"):
        if not line.strip() or line.startswith("#"):
            continue
        name, src = line.rstrip("\r\n").split("\t")[:2]
        dst = os.path.join(dest, name)
        if os.path.exists(dst) and os.path.samefile(src, dst):
            kept += 1
            continue
        shutil.copyfile(src, dst)
        n += 1
    print(f"copied {n} files to {dest}, {kept} already in place")


if __name__ == "__main__":
    main(*sys.argv[1:3])
