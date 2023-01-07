import torch
import numpy as np
from typing import List, Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ClassificationModel:
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
        train_loader,
        optimizer,
        criterion,
        device,
        **kwargs,
    ):
        """
        Train the model for one epoch.
        :param train_loader: training data loader
        :param optimizer: optimizer
        :param criterion: loss function
        :param device: device
        :return: loss
        """
        self.model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)
    
    def train(
        self,
        train_loader,
        val_loader,
        device,
        **kwargs,
    ):
        """
        Train the model.
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param device: device to use for training (cpu or cuda)
        :return: best model and metrics
        """

        # iterate over epochs
        best_model = None
        best_val_loss = np.inf
        best_val_acc = 0.0
        best_val_f1 = 0.0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            # train for one epoch
            train_loss = self.train_epoch(
                train_loader, optimizer, self.criterion, device
            )
            # evaluate on validation set
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, device)
            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_model = self.model.state_dict()

            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}")
            print(f"Val acc: {val_acc:.4f}")
            print(f"Val f1: {val_f1:.4f}")
            print()

        # load best model
        self.model.load_state_dict(best_model)
        return best_model, (best_val_loss, best_val_acc, best_val_f1)

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
        self.model.eval()
        running_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(np.argmax(outputs.cpu().numpy(), axis=1))
        return (
            running_loss / len(data_loader),
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred),
        )