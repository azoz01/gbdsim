import torch
from torch.utils.data import DataLoader

from gbdsim.causal.data import generate_synthetic_causal_data_example
from gbdsim.data.generator_dataset import GeneratorDataset


def test_generator_dataset_returns_proper_batches_when_loaded() -> None:
    # Given
    ds = GeneratorDataset(generate_synthetic_causal_data_example, 32, False)
    dl = DataLoader(ds, batch_size=3, collate_fn=lambda x: x)

    # When
    batches = list(dl)

    # Then
    assert len(batches) == 11
    assert [len(b) for b in batches] == [3] * 10 + [2]
    for b in batches:
        for (X1, y1), (X2, y2), label in b:
            assert X1.shape[0] == y1.shape[0]
            assert X2.shape[0] == y2.shape[0]
            assert label.shape == torch.Size([])


def test_generator_dataset_returns_repeatable_batches_when_specified() -> None:
    # Given
    ds = GeneratorDataset(generate_synthetic_causal_data_example, 10, True)
    dl = DataLoader(ds, batch_size=3, collate_fn=lambda x: x)

    # When
    l1 = list(dl)
    l2 = list(dl)

    # Then
    for b1, b2 in zip(l1, l2):
        for ex1, ex2 in zip(b1, b2):
            assert (ex1[0][0] == ex2[0][0]).all()
            assert (ex1[0][1] == ex2[0][1]).all()
            assert (ex1[1][0] == ex2[1][0]).all()
            assert (ex1[1][1] == ex2[1][1]).all()
            assert (ex1[2] == ex2[2]).all()


def test_generator_dataset_returns_non_repeatable_batches_when_specified() -> (
    None
):
    # Given
    ds = GeneratorDataset(generate_synthetic_causal_data_example, 10, False)
    dl = DataLoader(ds, batch_size=3, collate_fn=lambda x: x)

    # When
    l1 = list(dl)
    l2 = list(dl)

    # Then
    for b1, b2 in zip(l1, l2):
        for ex1, ex2 in zip(b1, b2):
            assert (
                ex1[0][0].shape != ex2[0][0].shape
                or ex1[1][0].shape != ex2[1][0].shape
                or ex1[2] != ex2[2]
            )
