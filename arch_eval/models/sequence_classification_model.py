import torch
import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class SequenceClassificationModel:
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
        is_multilabel: bool = False,
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
        self.verbose = verbose
        self.is_multilabel = is_multilabel
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = self._build_model()

    def _build_model(self):
        """
        Build the model according to the specified parameters.
        :return: a torch.nn.Module
        """


        print ("\n\n\n Generating the model \n\n\n")

        # If no layers are specified, return a simple linear model
        if len(self.layers) == 0:
            model = [torch.nn.Linear(self.input_embedding_size, self.num_classes)]
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
        
        if self.is_multilabel:
            model.append(torch.nn.Sigmoid())

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
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            # compute loss for each frame - the loss is averaged over the batch
            reshaped_outputs = outputs.view(-1, self.num_classes) # BS * NFrames, NClasses
            
            if self.is_multilabel:
                reshaped_labels = labels.view(-1, self.num_classes)
                total_loss = 0
                for cl in range(self.num_classes):
                    print(reshaped_outputs[:, cl])
                    loss = criterion(reshaped_outputs[:, cl], reshaped_labels[:, cl])
                    total_loss += loss
                loss = total_loss / self.num_classes
            else:
                # reshaped_labels should 
                reshaped_labels = labels.view(-1)
                print ("Outputs shape: ", reshaped_outputs.shape)
                print ("Labels shape: ", reshaped_labels.shape)
                loss = criterion(reshaped_outputs, reshaped_labels)
            #loss = criterion(outputs, labels) # Check the shapes of the inputs and outputs
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_dataloader)

    def diarization_error_rate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Compute the diarization error rate (DER) for the given predictions and labels.
        :param predictions: predictions
        :param labels: labels
        :return: DER
        """
        pass
        
    
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
        Train the model for the "token classification" task.
        Each frame is assigned a label and the sequence is compared to the ground truth.

        :param train_dataloader: training data loader
        :param val_dataloader: validation data loader
        :param learning_rate: learning rate used for training (max learning rate for the scheduler)
        :param device: device to use for training (cpu or cuda)
        :param max_num_epochs: maximum number of epochs to train for
        :return: best model and metrics
        """

        best_model = None
        best_val_loss = np.inf
        # TODO: include additional metrics for sequence tagging (e.g., diarization error rate)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        if self.is_multilabel:
            print("Using BCELoss")
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.model = self.model.to(device)

        for epoch in tqdm(range(max_num_epochs), desc="Epoch"):
            train_loss = self.train_epoch(
                train_dataloader,
                optimizer,
                self.criterion,
                device,
            )

            scheduler.step()
            val_metrics = self.evaluate(
                val_dataloader,
                device,
                return_predictions=False,
            )
            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # TODO: include additional metrics for sequence tagging (e.g., diarization error rate)
                best_model = self.model.state_dict()

            if self.verbose:
                print(
                    f"Epoch {epoch + 1} - train loss: {train_loss:.3f} - val loss: {val_loss:.3f}"
                )

        self.model.load_state_dict(best_model)
        metrics = {"val_loss": best_val_loss} # TODO: include additional metrics for sequence tagging (e.g., diarization error rate)
        return best_model, metrics


    def evaluate(
        self,
        data_loader,
        device: str = "cpu",
        return_predictions: bool = False,
        **kwargs,
    ):
        """
        Evaluate the model on the given data loader.
        :param data_loader: data loader containing the data to evaluate on
        :param device: device to use for evaluation (cpu or cuda)
        :return: loss, accuracy, f1 score
        """
        # TODO: include additional metrics for sequence tagging (e.g., diarization error rate) 
        self.model.eval()
        running_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(data_loader, desc="Evaluating")):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                # compute loss for each frame - the loss is averaged over the batch
                reshaped_outputs = outputs.view(-1, self.num_classes)
                if self.is_multilabel:
                    reshaped_labels = labels.view(-1, self.num_classes)
                    total_loss = 0
                    for cl in range(self.num_classes):
                        loss = self.criterion(reshaped_outputs[:, cl], reshaped_labels[:, cl])
                        total_loss += loss
                    loss = total_loss / self.num_classes
                else:
                    reshaped_labels = labels.view(-1)
                    loss = self.criterion(reshaped_outputs, reshaped_labels)
                running_loss += loss.item()
                y_true.extend(labels.tolist())
                y_pred.extend(torch.argmax(outputs, dim=1).tolist())

        metrics = {
            "loss": running_loss / len(data_loader),
        }
        if return_predictions:
            predictions = {
                "y_true": y_true,
                "y_pred": y_pred,
            }
            return metrics, predictions
        else:
            return metrics
        
