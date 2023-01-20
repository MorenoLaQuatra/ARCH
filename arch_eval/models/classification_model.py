import torch
import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm
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
        verbose: bool = False,
        **kwargs,
    ):
        """
        :param layers: list of layer sizes
        :param input_embedding_size: size of the input embedding
        :param activation: activation function that will be used for non-linear evaluation
        :param dropout: dropout rate
        :param num_classes: number of classes
        :param verbose: whether to print progress
        """
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        self.activation = activation
        self.dropout = dropout
        self.num_classes = num_classes
        self.verbose = verbose
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
        self.model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)

            # print shapes
            print ("Output shape: ", outputs.shape)
            print ("Labels shape: ", labels.shape)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_dataloader)
    
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

        # iterate over epochs
        best_model = None
        best_val_loss = np.inf
        best_val_acc = 0.0
        best_val_f1 = 0.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = self.model.to(device)

        for epoch in tqdm(range(max_num_epochs), desc="Epochs"):
            # train for one epoch
            train_loss = self.train_epoch(
                train_dataloader, optimizer, self.criterion, device
            )
            scheduler.step()
            # evaluate on validation set
            val_loss, val_acc, val_f1 = self.evaluate(val_dataloader, device)
            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_model = self.model.state_dict()

            # report metrics in tqdm
            if self.verbose:
                tqdm.write(
                    f"Epoch {epoch + 1}/{max_num_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f} - Val f1: {val_f1:.4f}"
                )

        # load best model
        self.model.load_state_dict(best_model)
        return best_model, (best_val_loss, best_val_acc, best_val_f1)

    def evaluate(
        self,
        dataloader,
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
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(np.argmax(outputs.cpu().numpy(), axis=1))
        return (
            running_loss / len(dataloader),
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"),
        )