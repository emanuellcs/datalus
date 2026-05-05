"""Compatibility entrypoint for DATALUS training."""

from datalus.training import DatalusTrainer, TrainingConfig

__all__ = ["DatalusTrainer", "TrainingConfig"]


if __name__ == "__main__":
    from datalus.cli import app

    app()
