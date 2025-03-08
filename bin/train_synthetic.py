import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from gbdsim.causal.data import generate_synthetic_causal_data_example
from gbdsim.data.generator_dataset import GeneratorDataset
from gbdsim.experiment_config import ExperimentConfig
from gbdsim.results.metric_learning_results import MetricLearningResults
from gbdsim.training.metric_learning import MetricLearner
from gbdsim.utils.constants import DEVICE
from gbdsim.utils.model_factory import ModelFactory


def main():
    logger.info("Initializing script")
    torch.set_float32_matmul_precision("high")
    seed_everything(123)

    logger.info("Parsing args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path)
    args = parser.parse_args()

    logger.info("Parsing config")
    with open(args.config_path) as f:
        config = ExperimentConfig.model_validate(
            yaml.load(f, Loader=yaml.CLoader)
        )

    logger.info("Initializing output directory")
    output_dir = Path(
        f"results/synthetic/{config.model.type}/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"  # noqa: E501
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config_path, output_dir / "config.yaml")

    logger.info("Initializing training data")
    train_dataset = GeneratorDataset(
        generate_synthetic_causal_data_example,
        config.data.train_dataset_size,
        False,
    )
    train_loader = DataLoader(
        train_dataset,
        config.data.train_batch_size,
        collate_fn=lambda x: x,
        num_workers=7,
        pin_memory=True,
        worker_init_fn=lambda id: seed_everything(id, verbose=False),  # type: ignore # noqa: E501
    )

    logger.info("Initializing validation data")
    val_dataset = GeneratorDataset(
        generate_synthetic_causal_data_example,
        config.data.val_dataset_size,
        True,
    )
    val_loader = DataLoader(
        val_dataset,
        config.data.val_batch_size,
        collate_fn=lambda x: x,
        num_workers=7,
        pin_memory=True,
        worker_init_fn=lambda id: seed_everything(id, verbose=False),  # type: ignore # noqa: E501
    )

    logger.info("Preparing model training")
    model = MetricLearner(ModelFactory.get_model(config.model))
    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping("val/mae", min_delta=1e-3, patience=5, mode="min"),
            ModelCheckpoint(output_dir),
        ],
    )
    trainer.fit(model, train_loader, val_loader)

    logger.info("Calculating metrics")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            MetricLearningResults().evaluate_model(  # TODO: restore
                model.model.eval().to(DEVICE), 1024, 100, 5  # type: ignore
            ),
            f,
            indent=4,
        )
    logger.info("Finished script")


if __name__ == "__main__":
    main()
