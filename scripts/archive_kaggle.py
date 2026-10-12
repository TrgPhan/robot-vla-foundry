"""Optionally archive only one run, with a disk budget and atomic ZIP replacement."""

import argparse
from pathlib import Path

from kaggle_storage import inside

from robot_foundry.io import read_json
from robot_foundry.storage import archive_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--destination", default="dist")
    parser.add_argument("--include-prepared", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    out = inside(project / "outputs" / args.run, project / "outputs")
    cfg = read_json(out / "run_config.json")
    destination = Path(args.destination)
    print(archive_directory(out, destination / f"robot_foundry_{args.run}_results.zip", project))
    if args.include_prepared:
        prepared = inside(project / cfg["prepared_dir"], project / "data/prepared")
        print(archive_directory(prepared, destination / f"robot_foundry_{args.run}_prepared.zip", project))


if __name__ == "__main__":
    main()
