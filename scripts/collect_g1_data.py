"""Record full-body teacher demonstrations; no optimizer/training happens in this step."""

import argparse
from robot_foundry.g1_collect import collect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/g1_data.json")
    collect_dataset(parser.parse_args().config)
