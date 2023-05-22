import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class RAVDESS():
    '''
    This class implements the functionality to load the RAVDESS dataset.
    It implements a train/test split of the dataset (random split with seed 42).
    '''

    def __init__(
        self,
        path,
        verbose = False,
        precompute_embeddings: bool = False,
    ):

        self.path = path
        self.verbose = verbose
        self.is_multilabel = False
        self.precompute_embeddings = precompute_embeddings
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()

    def _load_data(self):
        '''
        Load the data and split it into train, validation and test sets.
        :return: a list of lists containing the audio paths and the labels
        '''

        # find all wav files in the path including subfolders
        audio_paths = glob.glob(os.path.join(self.path, '**/*.wav'), recursive=True)

        # get the labels from the file names (e.g. 03-01-01-01-01-01-01.wav) the 3rd element is the emotion
        labels = [int(os.path.basename(path).split('-')[2]) for path in audio_paths]

        # convert labels to integers
        labels = [int(label) - 1 for label in labels]
        self.num_classes = len(np.unique(labels))

        if self.verbose:
            print("Number of classes: ", self.num_classes)
            print("Number of samples: ", len(audio_paths))

        # split the data into train, validation and test sets
        train_paths, test_paths, train_labels, test_labels = train_test_split(audio_paths, labels, test_size=0.2, random_state=42)
        train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)

        return train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels

    def evaluate(
        self,
        model: Model,
        mode: str = 'linear',
        device: str = 'cpu',
        batch_size: int = 32,
        num_workers: int = 0,
        max_num_epochs: int = 100,
    ):
        '''
        Evaluate a model on the dataset.
        :param model: the model to evaluate
        :param mode: the mode to use for the evaluation (linear or nonlinear)
        :param device: the device to use for the evaluation (cpu or cuda)
        :param batch_size: the batch size to use for the evaluation
        :param num_workers: the number of workers to use for the evaluation
        :param max_num_epochs: the maximum number of epochs to use for the evaluation
        :return: the evaluation results
        '''

        if mode == 'linear':
            layers = []
        elif mode == 'non-linear':
            layers = [model.get_classification_embedding_size()]
        elif mode == 'attention-pooling':
            layers = []
        else:
            raise ValueError(f"Invalid mode {mode}")

        clf_model = ClassificationModel(
            layers = layers,
            input_embedding_size = model.get_classification_embedding_size(),
            activation = "relu",
            dropout = 0.1,
            num_classes = self.num_classes,
            verbose = self.verbose,
            is_multilabel = self.is_multilabel,
            mode = mode,
        )

        # create train, validation and test datasets
        train_dataset = ClassificationDataset(
            audio_paths = self.train_paths,
            labels = self.train_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
            mode = mode,
        )

        val_dataset = ClassificationDataset(
            audio_paths = self.validation_paths,
            labels = self.validation_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
            mode = mode,
        )

        test_dataset = ClassificationDataset(
            audio_paths = self.test_paths,
            labels = self.test_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
            mode = mode,
        )

        # create train, validation and test dataloaders

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
        )

        # train the model
        clf_model.train(
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader,
            max_num_epochs = max_num_epochs,
            device = device,
        )

        # evaluate the model
        metrics = clf_model.evaluate(
            dataloader = test_dataloader,
            device = device,
        )

        return metrics