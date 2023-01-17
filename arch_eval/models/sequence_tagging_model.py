import torch
import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SequenceTaggingModel:
    """
    This class implements a classification model. It contains the basic methods for
    training and evaluating a classification model.
    """
    def __init__(
        self,
        layers: Union[List[int], Tuple[int]],
        input_embedding_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
        num_classes: int = 2,
        **kwargs,
    ):
        """
        :param layers: list of layer sizes
        :param activation: activation function
        :param dropout: dropout rate
        """
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        self.activation = activation
        self.dropout = dropout
        self.num_classes = num_classes
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the model according to the specified parameters.
        :return: a torch.nn.Module
        """

        # If no layers are specified, return a simple linear model
        if len(self.layers) == 0:
            clf_model = torch.nn.Linear(self.input_embedding_size, self.num_classes)
        else:
            # Build the model
            model = []
            for i, layer_size in enumerate(self.layers):
                if i == 0:
                    model.append(torch.nn.Linear(self.input_embedding_size, layer_size))
                else:
                    model.append(torch.nn.Linear(self.layers[i - 1], layer_size))
                model.append(torch.nn.Dropout(self.dropout))
                model.append(torch.nn.ReLU())
            model.append(torch.nn.Linear(self.layers[-1], self.num_classes))
            clf_model = torch.nn.Sequential(*model)

        return clf_model

    def train_epoch(
        self,
        train_dataloader,
        optimizer,
        criterion,
        device,
        **kwargs,
    ):
        """
        Train the model for one epoch.
        :param train_dataloader: training data loader
        :param optimizer: optimizer
        :param criterion: loss function
        :param device: device
        :return: loss
        """
        # TODO: implement this method
    
    def train(
        self,
        train_dataloader,
        val_dataloader,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        max_num_epochs: int = 10,
        **kwargs,
    ):
        """
        Train the model.
        :param train_dataloader: training data loader
        :param val_dataloader: validation data loader
        :param device: device to use for training (cpu or cuda)
        :return: best model and metrics
        """

        # TODO: implement this method

    def evaluate(
        self,
        data_loader,
        device,
        **kwargs,
    ):
        """
        Evaluate the model on the given data loader.
        :param data_loader: data loader containing the data to evaluate on
        :param device: device to use for evaluation (cpu or cuda)
        :return: loss, accuracy, f1 score
        """
        # TODO: implement this method
        # Include specific metrics for sequence tagging
