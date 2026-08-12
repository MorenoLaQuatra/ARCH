import numpy as np
import pytest
import torch

from arch_eval.datasets.classification_dataset import ClassificationDataset


class DummyModel:
    def get_embeddings(self, audio):
        return torch.tensor([float(audio[0])])


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([10, 20, 30]),
        torch.tensor([10, 20, 30]),
        [10, 20, 30],
        (10, 20, 30),
    ],
)
def test_precompute_skips_undecodable_paths_without_misaligning_labels(
    monkeypatch,
    labels,
):
    def load_audio(_dataset, audio_path):
        if audio_path == "bad.mp3":
            raise RuntimeError("invalid audio")
        value = float(audio_path[0])
        return torch.tensor([value, value])

    monkeypatch.setattr(ClassificationDataset, "_load_audio_from_path", load_audio)
    original_label_type = type(labels)
    dataset = ClassificationDataset(
        audio_paths=["0.mp3", "bad.mp3", "2.mp3"],
        labels=labels,
        model=DummyModel(),
        precompute_embeddings=True,
    )

    assert dataset.audio_paths == ["0.mp3", "2.mp3"]
    assert isinstance(dataset.labels, original_label_type)
    assert torch.as_tensor(dataset.labels).tolist() == [10, 30]
    assert dataset.embeddings.tolist() == [[0.0], [2.0]]
    assert len(dataset) == 2


def test_precompute_does_not_hide_model_runtime_errors(monkeypatch):
    class FailingModel:
        def get_embeddings(self, _audio):
            raise RuntimeError("model failure")

    monkeypatch.setattr(
        ClassificationDataset,
        "_load_audio_from_path",
        lambda _dataset, _path: torch.tensor([0.0, 0.0]),
    )

    with pytest.raises(RuntimeError, match="model failure"):
        ClassificationDataset(
            audio_paths=["valid.mp3"],
            labels=np.asarray([0]),
            model=FailingModel(),
            precompute_embeddings=True,
        )
