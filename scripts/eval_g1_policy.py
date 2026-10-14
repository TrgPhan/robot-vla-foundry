"""CPU simulation client for a trained G1 policy server; saves learned-policy full-body videos."""

import argparse
from robot_foundry.g1_rollout import evaluate_g1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy-url", default="http://127.0.0.1:8775")
    p.add_argument("--output", required=True)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--execution-horizon", type=int, default=4)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--policy-mode", choices=["trained", "mean_token"], default="trained")
    a = p.parse_args()
    evaluate_g1(
        a.checkpoint, a.policy_url, a.output, a.trials, a.execution_horizon, a.max_frames, a.policy_mode
    )
