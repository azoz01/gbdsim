import json
from pathlib import Path

import openml
from tqdm import tqdm

DATASETS_PATH = Path("data/tabrepo/datasets")


def download_dataset(metadata):
    dataset = openml.datasets.get_dataset(metadata["did"]).get_data()[0]
    dataset.to_csv(DATASETS_PATH / f"{metadata['did']}.csv", index=False)  # type: ignore # noqa: E501
    return metadata["did"]


def main():
    DATASETS_PATH.mkdir(exist_ok=True, parents=True)
    with open("data/tabrepo/metadatas.json", "r") as f:
        metadatas = json.load(f)
    for metadata in tqdm(metadatas.values()):
        download_dataset(metadata)


if __name__ == "__main__":
    main()
