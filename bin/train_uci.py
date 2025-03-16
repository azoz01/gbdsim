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

from gbdsim.data.data_pairs_generator import DatasetsPairsGenerator
from gbdsim.data.generator_dataset import GeneratorDataset
from gbdsim.experiment_config import ExperimentConfig
from gbdsim.results.origin_classification_results import (
    OriginClassificationResults,
)
from gbdsim.training.origin_classification import OriginClassificationLearner
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
        f"results/uci/{config.model.type}/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"  # noqa: E501
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config_path, output_dir / "config.yaml")

    logger.info("Initializing training data")
    with open("data/uci/meta_split.json", "r") as f:
        meta_split = json.load(f)
    train_paths_with_data = list(
        filter(
            lambda p: p.stem in meta_split["train"],
            Path("data/uci/raw").iterdir(),
        )
    )
    train_dataset = GeneratorDataset(
        DatasetsPairsGenerator.from_paths(
            train_paths_with_data
        ).generate_pair_of_datasets_with_label,
        config.data.train_dataset_size,
        False,
    )
    train_loader = DataLoader(
        train_dataset,
        config.data.train_batch_size,
        collate_fn=lambda x: x,
    )

    logger.info("Initializing validation data")
    val_paths_with_data = list(
        filter(
            lambda p: p.stem in meta_split["val"],
            Path("data/uci/raw").iterdir(),
        )
    )
    val_dataset = GeneratorDataset(
        DatasetsPairsGenerator.from_paths(
            val_paths_with_data
        ).generate_pair_of_datasets_with_label,
        config.data.val_dataset_size,
        False,
    )
    val_loader = DataLoader(
        val_dataset,
        config.data.val_batch_size,
        collate_fn=lambda x: x,
    )

    logger.info("Preparing model training")
    model = OriginClassificationLearner(ModelFactory.get_model(config.model))
    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping(
                "val/accuracy", min_delta=1e-3, patience=10, mode="max"
            ),
            ModelCheckpoint(output_dir),
        ],
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader)

    logger.info("Calculating metrics")
    results = OriginClassificationResults()
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            results.evaluate_model(
                model.model.eval().to(DEVICE),  # type: ignore
                val_loader,
                config.data.val_dataset_size // config.data.val_batch_size,
            ),
            f,
            indent=4,
        )
    results.visualize_clustering(
        model.model.eval().to(DEVICE),  # type: ignore
        output_dir,
    )
    logger.info("Finished script")


if __name__ == "__main__":
    main()
