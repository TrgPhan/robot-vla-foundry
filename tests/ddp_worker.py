"""Worker for the actual two-process integration test, including checkpoint interruption."""

import argparse

from robot_foundry.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-at", type=int, default=0)
    args = parser.parse_args()
    if args.interrupt_at:
        import robot_foundry.train_optimized as engine

        save = engine.atomic_checkpoint

        def interrupted(path, state):
            save(path, state)
            if state["step"] == args.interrupt_at:
                raise InterruptedError("Test interruption after a complete atomic checkpoint")

        engine.atomic_checkpoint = interrupted
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
