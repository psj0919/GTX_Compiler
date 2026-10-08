#!/usr/bin/env python3
"""Restore ignored models/outputs from a migration backup without overwriting files."""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path, help="Directory containing GTX_Compiler/ and vision.cpp/")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    backup, destination = args.backup.resolve(), args.destination.resolve()
    if not (backup / "GTX_Compiler").is_dir():
        parser.error("backup must contain GTX_Compiler/")
    if backup == destination or backup in destination.parents or destination in backup.parents:
        parser.error("backup and destination must be separate directory trees")
    copied = skipped = 0
    pairs = [(backup / "GTX_Compiler" / name, destination / name)
             for name in ("output", "checkpoints", "export1")]
    pairs += [(backup / "GTX_Compiler/vision.cpp/models", destination / "vision.cpp/models"),
              (backup / "vision.cpp/models", destination / "vision.cpp/models")]
    for source, target in pairs:
        if not source.is_dir():
            continue
        for path in source.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            # Rebuild executables against the new shared ABI.
            if path.name.startswith("run_") and path.suffix == "":
                continue
            output = target / path.relative_to(source)
            if output.exists():
                skipped += 1
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            copied += 1
    print(f"Restored {copied} files; kept {skipped} existing files. Backup unchanged.")


if __name__ == "__main__":
    main()
