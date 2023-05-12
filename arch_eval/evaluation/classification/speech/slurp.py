import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split


class SLURP():
    '''
    This class implements the functionality to load the SLURP dataset
    and the recipe for its evaluation.
    It implements the original train/devel/test split of the dataset.
    '''

    def __init__(
        self,
        path: str,
        verbose: bool = False,
        precompute_embeddings: bool = False,
    ):

        self.path = path
        self.verbose = verbose
        self.is_multilabel = False
        self.precompute_embeddings = precompute_embeddings
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()


    def _load_data(self):
        '''
        Load the data of the dataset and provide the train, validation and test splits.
        :return: a list of lists containing the audio paths and the labels.
        '''

        # train.jsonl, devel.jsonl, test.jsonl contain the paths to the audio files and the labels
        train_df = pd.read_json(os.path.join(self.path, 'train.jsonl'), lines=True)
        validation_df = pd.read_json(os.path.join(self.path, 'devel.jsonl'), lines=True)
        test_df = pd.read_json(os.path.join(self.path, 'test.jsonl'), lines=True)

        train_paths = []
        train_labels = []
        for index, row in train_df.iterrows():
            # get the list of recordings
            recordings = row['recordings']
            for recording in recordings:
                # get the path to the audio file
                train_paths.append(self.path + "/slurp_real/" + recording['file'])
                train_labels.append(row["intent"])

        validation_paths = []
        validation_labels = []
        for index, row in validation_df.iterrows():
            # get the list of recordings
            recordings = row['recordings']
            for recording in recordings:
                # get the path to the audio file
                validation_paths.append(self.path + "/slurp_real/" + recording['file'])
                validation_labels.append(row["intent"])

        test_paths = []
        test_labels = []
        for index, row in test_df.iterrows():
            # get the list of recordings
            recordings = row['recordings']
            for recording in recordings:
                # get the path to the audio file
                test_paths.append(self.path + "/slurp_real/" + recording['file'])
                test_labels.append(row["intent"])

        # get unique list of labels
        self.all_labels = list(set(train_labels + validation_labels + test_labels))

        # map labels to integers
        self.label2int = {label: i for i, label in enumerate(self.all_labels)}
        self.int2label = {i: label for i, label in enumerate(self.all_labels)}

        # convert labels to integers
        train_labels = [self.label2int[label] for label in train_labels]
        validation_labels = [self.label2int[label] for label in validation_labels]
        test_labels = [self.label2int[label] for label in test_labels]

        self.num_classes = len(self.all_labels)

        if self.verbose:
            print(f"Number of classes: {self.num_classes}")
            print(f"Number of training samples: {len(train_paths)}")
            print(f"Number of validation samples: {len(validation_paths)}")
            print(f"Number of test samples: {len(test_paths)}")
            print(f"Total number of samples: {len(train_paths) + len(validation_paths) + len(test_paths)}")

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
            layers = [model.get_embedding_layer()]
        else:
            raise ValueError('Invalid mode: ' + mode)

        clf_model = ClassificationModel(
            layers = layers,
            input_embedding_size = model.get_classification_embedding_size(),
            activation = "relu",
            dropout = 0.1,
            num_classes = self.num_classes,
            verbose = self.verbose,
        )

        # create train, validation and test datasets
        train_dataset = ClassificationDataset(
            audio_paths = self.train_paths,
            labels = self.train_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
        )

        val_dataset = ClassificationDataset(
            audio_paths = self.validation_paths,
            labels = self.validation_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
        )

        test_dataset = ClassificationDataset(
            audio_paths = self.test_paths,
            labels = self.test_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = self.precompute_embeddings,
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