import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
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

CONFIG_PATH = Path("config/artificial_data.yaml")
OUTPUT_DIR = Path(
    f"results/synthetic/{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config.yaml")
    torch.set_float32_matmul_precision("high")
    seed_everything(123)
    with open(CONFIG_PATH) as f:
        config = ExperimentConfig.model_validate(
            yaml.load(f, Loader=yaml.CLoader)
        )

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
    model = MetricLearner(ModelFactory.get_model(config.model))

    trainer = Trainer(
        max_epochs=config.training.num_epochs,
        default_root_dir=OUTPUT_DIR,
        callbacks=[
            EarlyStopping("val/mae", min_delta=1e-3, patience=5, mode="min"),
            ModelCheckpoint(OUTPUT_DIR),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(
            MetricLearningResults().evaluate_model(
                model.model.eval().to(DEVICE), 1024, 100, 5  # type: ignore
            ),
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
