import pickle as pkl
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbdsim.data.data_preprocessor import DataPreprocessor
from gbdsim.utils.constants import DEVICE
from gbdsim.utils.protocols import (
    DatasetRepresentationCalculator,
    OriginClassifier,
)


class OriginClassificationResults:
    VISUALIZATION_PATH: Path = Path("data/uci/visualization")

    def evaluate_model(
        self,
        model: OriginClassifier,
        val_dataloader: DataLoader,
        n_batches: int,
    ):
        final_results = dict()
        self.__evaluate_metrics(
            final_results, model, val_dataloader, n_batches
        )
        return final_results

    def __evaluate_metrics(
        self,
        final_results: dict[str, Any],
        model: OriginClassifier,
        val_dataloader: DataLoader,
        n_batches: int,
    ):
        model = model.eval()
        labels = []
        predictions = []
        for i, batch in enumerate(tqdm(val_dataloader)):
            if i == n_batches:
                break
            for X1, y1, X2, y2, label in batch:
                labels.append(label)
                predictions.append(
                    model.calculate_dataset_origin_probability(
                        X1.to(DEVICE),
                        y1.to(DEVICE),
                        X2.to(DEVICE),
                        y2.to(DEVICE),
                    ).item()
                )
        labels = np.array(labels)
        predictions = np.array(predictions)
        final_results["cross_entropy"] = log_loss(labels, predictions)
        final_results["accuracy"] = accuracy_score(
            labels, (predictions >= 0.5).astype(np.int32)
        )

    def visualize_clustering(
        self, model: DatasetRepresentationCalculator, output_path: Path
    ) -> None:
        preprocessor = DataPreprocessor()
        representations = []
        labels = []
        for label_path in tqdm(list(self.VISUALIZATION_PATH.iterdir())):
            for dataset_path in label_path.iterdir():
                df = pd.read_csv(dataset_path)
                X, y = preprocessor.preprocess_pandas_data(df)
                representations.append(
                    model.calculate_dataset_representation(
                        X.to(DEVICE), y.to(DEVICE)
                    )
                )
                labels.append(label_path.stem)
        representation_data = (
            torch.concat(representations).detach().cpu().numpy()
        )
        with open(output_path / "representations.pkl", "wb") as f:
            torch.save(representation_data, f)
        with open(output_path / "representation_labels.pkl", "wb") as f:
            pkl.dump(labels, f)

        representation_data = TSNE().fit_transform(representation_data)
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=representation_data[:, 0],
            y=representation_data[:, 1],
            hue=labels,
            ax=ax,
        )
        ax.set_title("Data representations")
        plt.savefig(output_path / "representations.jpg")
