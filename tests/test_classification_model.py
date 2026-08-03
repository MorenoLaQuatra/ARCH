from types import MethodType

import pytest
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from arch_eval.models.classification_model import ClassificationModel


def build_classifier() -> ClassificationModel:
    return ClassificationModel(
        layers=[],
        input_embedding_size=2,
        num_classes=2,
        mode="linear",
    )


@pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)` before "
    "`optimizer.step\\(\\)`:UserWarning"
)
def test_train_restores_independent_lowest_validation_loss_snapshot() -> None:
    classifier = build_classifier()
    epoch_weights = iter((1.0, 2.0, 3.0))
    validation_losses = iter((0.2, 0.5, 0.4))

    def fake_train_epoch(self, *args, **kwargs):
        with torch.no_grad():
            self.model[0].weight.fill_(next(epoch_weights))
        return 0.0

    def fake_evaluate(self, *args, **kwargs):
        return {
            "loss": next(validation_losses),
            "accuracy": 0.0,
            "f1": 0.0,
        }

    classifier.train_epoch = MethodType(fake_train_epoch, classifier)
    classifier.evaluate = MethodType(fake_evaluate, classifier)
    dataset = TensorDataset(torch.zeros(10, 2), torch.zeros(10, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=1)

    best_model, best_metrics = classifier.train(
        train_dataloader=loader,
        val_dataloader=loader,
        max_num_epochs=3,
    )

    assert best_metrics["loss"] == pytest.approx(0.2)
    assert torch.all(classifier.model[0].weight == 1.0)
    assert torch.all(best_model["0.weight"] == 1.0)


def test_evaluate_weights_loss_by_example_count() -> None:
    classifier = build_classifier()
    classifier.model = torch.nn.Identity()
    classifier.criterion = torch.nn.CrossEntropyLoss()
    logits = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    metrics = classifier.evaluate(loader, device="cpu")

    assert metrics["loss"] == pytest.approx(
        functional.cross_entropy(logits, labels).item()
    )
