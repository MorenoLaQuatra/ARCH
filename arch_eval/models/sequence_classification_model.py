import torch
import numpy as np
from typing import List, Union, Tuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionAccuracy, DetectionErrorRate

class extract_tensor(torch.nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]

class SequenceClassificationModel:
    """
    This class implements a classification model. It contains the basic methods for
    training and evaluating a classification model.
    """
    def __init__(
        self,
        layers: Union[int, List[int], Tuple[int]],
        input_embedding_size: int,
        activation: str = "relu",
        dropout: float = 0.1,
        num_classes: int = 2,
        verbose: bool = False,
        is_multilabel: bool = False,
        model_frame_length_ms: float = 20.00001,
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
        self.model_frame_length_ms = model_frame_length_ms
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

    '''
    def _build_model_lstm(self):
        #Create an LSTM for the task of sequence classification. Each frame is classified independently.

        print ("\n\n\n Generating the model \n\n\n")
        if self.layers == []:
            self.layers = 1

        # create the LSTM
        lstm_model = torch.nn.LSTM(
            input_size = self.input_embedding_size,
            hidden_size = self.input_embedding_size,
            num_layers = 1,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = False
        )

        if self.is_multilabel:
            # create the linear layer for multilabel, multiclass classification
            linear = torch.nn.Sequential(
                torch.nn.Linear(self.input_embedding_size, self.num_classes),
                torch.nn.Sigmoid()
            )
        else:
            # create the linear layer for multiclass classification
            linear = torch.nn.Sequential(
                torch.nn.Linear(self.input_embedding_size, self.num_classes)
            )

        # create the model but take care of extracting tensors from the LSTM output
        model = torch.nn.Sequential(
            lstm_model,
            extract_tensor(),
            linear
        )

        return model
    '''

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
            # compute loss for each frame - the loss is averaged over the batch
            reshaped_outputs = outputs.view(-1, self.num_classes) # BS * NFrames, NClasses
            
            if self.is_multilabel:
                reshaped_labels = labels.view(-1, self.num_classes)
                total_loss = 0
                for cl in range(self.num_classes):
                    loss = criterion(reshaped_outputs[:, cl], reshaped_labels[:, cl])
                    total_loss += loss
                loss = total_loss / self.num_classes
            else:
                reshaped_labels = labels.view(-1)
                '''
                # print the total number of frames per class
                for cl in range(self.num_classes):
                    print (f"Class {cl} has {torch.sum(reshaped_labels == cl)} frames")
                # also for the predicted classes
                for cl in range(self.num_classes):
                    print (f"Predicted class {cl} has {torch.sum(torch.argmax(reshaped_outputs, dim=1) == cl)} frames")
                '''
                
                loss = criterion(reshaped_outputs, reshaped_labels)
            #loss = criterion(outputs, labels) # Check the shapes of the inputs and outputs
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_dataloader)

    def sequence_accuracy(self, y_true, y_pred):
        """
        Compute the accuracy for each sequence in the batch.
        :param y_true: ground truth
        :param y_pred: predictions
        :return: accuracy
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        accuracies = []
        for i in range(y_true.shape[0]):
            accuracies.append(accuracy_score(y_true[i], y_pred[i]))
        return np.mean(accuracies)

    def _convert_predictions_to_segments(self, y):
        """
        Convert the predictions to segments.
        :param y: predictions
        :return: list of lists of tuples (class, start, end)
        """
        # find all contiguous segments of the same class
        elements = []
        for i in range(y.shape[0]):
            elements.append([])
            for j in range(y.shape[1]):
                if j == 0:
                    start = 0
                    end = 0
                    prev_class = y[i, j]
                    continue
                if y[i, j] != prev_class:
                    # append class, start, end
                    elements[i].append((prev_class, start, end))
                    start = j
                    end = j
                    prev_class = y[i, j]
                else:
                    end = j
            elements[i].append((prev_class, start, end))

        return elements

    def _convert_elements_to_annotations(self, elements, ignore_labels=[]):
        """
        Convert the list of lists of tuples to a list of lists of Segments.
        :param elements: list of lists of tuples (class, start, end)
        :return: list of lists of Segments
        """
        annotations = []
        for i in range(len(elements)):
            annotations.append(Annotation())
            for j in range(len(elements[i])):
                label = elements[i][j][0]
                if label in ignore_labels:
                    continue
                start = elements[i][j][1] * self.model_frame_length_ms / 1000
                end = elements[i][j][2] * self.model_frame_length_ms / 1000
                annotations[i][Segment(start, end)] = label
        return annotations

    def compute_detection_accuracy(self, y_true, y_pred):
        """
        Compute the detection accuracy for each sequence in the batch using pyannote.metrics.
        :param y_true: ground truth
        :param y_pred: predictions
        :return: detection accuracy
        """
    
        gt_elements = self._convert_predictions_to_segments(y_true)
        pred_elements = self._convert_predictions_to_segments(y_pred)

        # references = self._convert_elements_to_annotations(gt_elements, ignore_labels=[0])
        # hypotheses = self._convert_elements_to_annotations(pred_elements, ignore_labels=[0])

        '''
        for i in range(len(references)):
            print ("Reference: ", references[i])
            print ("Hypothesis: ", hypotheses[i])
        print ("\n\n\n")
        '''

        metric = DetectionAccuracy()

        class_specific_detection_accuracy = []

        all_labels = set(y_pred.flatten().tolist() + y_true.flatten().tolist())

        # compute the detection accuracy for each class - excluding 0
        for cl in range(1, self.num_classes):
            detection_accuracy = []
            # get only the segments with the current class for both references and hypotheses
            # ignore all other classes
            ignore_labels = all_labels - set([cl])

            references = self._convert_elements_to_annotations(gt_elements, ignore_labels=ignore_labels)
            hypotheses = self._convert_elements_to_annotations(pred_elements, ignore_labels=ignore_labels)

            for i in range(len(references)):
                detection_accuracy.append(metric(references[i], hypotheses[i]))

            class_specific_detection_accuracy.append(np.mean(detection_accuracy))

        return np.mean(class_specific_detection_accuracy)

    def compute_detection_error_rate(self, y_true, y_pred):
        """
        Compute the detection error rate for each sequence in the batch using pyannote.metrics.
        :param y_true: ground truth
        :param y_pred: predictions
        :return: detection error rate
        """
    
        gt_elements = self._convert_predictions_to_segments(y_true)
        pred_elements = self._convert_predictions_to_segments(y_pred)

        # references = self._convert_elements_to_annotations(gt_elements, ignore_labels=[0])
        # hypotheses = self._convert_elements_to_annotations(pred_elements, ignore_labels=[0])

        metric = DetectionErrorRate()

        class_specific_detection_error_rate = []

        all_labels = set(y_pred.flatten().tolist() + y_true.flatten().tolist())

        # compute the detection error rate for each class - excluding 0
        for cl in range(1, self.num_classes):
            detection_error_rate = []
            # get only the segments with the current class for both references and hypotheses
            # ignore all other classes
            ignore_labels = all_labels - set([cl])

            references = self._convert_elements_to_annotations(gt_elements, ignore_labels=ignore_labels)
            hypotheses = self._convert_elements_to_annotations(pred_elements, ignore_labels=ignore_labels)

            for i in range(len(references)):
                detection_error_rate.append(metric(references[i], hypotheses[i]))

            class_specific_detection_error_rate.append(np.mean(detection_error_rate))

        return np.mean(class_specific_detection_error_rate)

    def train(
        self,
        train_dataloader,
        val_dataloader,
        learning_rate: float = 1e-4,
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
        best_val_metrics = {}
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
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
                best_val_metrics = val_metrics
                best_model = self.model.state_dict()


            if self.verbose:
                print(
                    f"Epoch {epoch + 1} - train loss: {train_loss:.3f} - val loss: {val_loss:.3f} - val accuracy: {val_metrics['accuracy']:.3f} - val error rate: {val_metrics['error_rate']:.3f}"
                )

        self.model.load_state_dict(best_model)
        metrics = {
            "val_loss": best_val_loss,
            "val_accuracy": best_val_metrics["accuracy"],
            "val_error_rate": best_val_metrics["error_rate"],
        } 
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
            for i, (inputs, labels) in enumerate(data_loader):
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
                y_true.extend(labels)
                y_pred.extend(torch.argmax(outputs, dim=2))

        y_true = torch.stack(y_true).cpu().numpy()
        y_pred = torch.stack(y_pred).cpu().numpy()

        accuracy = self.compute_detection_accuracy(y_true, y_pred)
        error_rate = self.compute_detection_error_rate(y_true, y_pred)
        #print(y_pred)
        
        metrics = {
            "loss": running_loss / len(data_loader),
            "accuracy": accuracy,
            "error_rate": error_rate,
        }
        if return_predictions:
            predictions = {
                "y_true": y_true,
                "y_pred": y_pred,
            }
            return metrics, predictions
        else:
            return metrics
        
