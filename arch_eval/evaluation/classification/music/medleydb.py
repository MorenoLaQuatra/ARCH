import os
import glob
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class MedleyDB():
    '''
    This class implements the functionality to load the Medley Solos DB dataset.
    It implements the original train/validation/test split proposed by the authors.
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
        :return: lists of audio paths and labels for train, validation and test sets
        '''

        # load metadata Medley-solos-DB_metadata.csv
        metadata = pd.read_csv(os.path.join(self.path, "Medley-solos-DB_metadata.csv"))

        # subset train, validation and test sets are defined by the authors
        train = metadata[metadata["subset"] == "training"]
        validation = metadata[metadata["subset"] == "validation"]
        test = metadata[metadata["subset"] == "test"]

        # get the audio paths and the labels
        train_ids = train["uuid4"].values
        validation_ids = validation["uuid4"].values
        test_ids = test["uuid4"].values

        # labels are the instrument_id
        train_labels = train["instrument_id"].values
        validation_labels = validation["instrument_id"].values
        test_labels = test["instrument_id"].values

        # map to integers
        train_labels = [int(label) for label in train_labels]
        validation_labels = [int(label) for label in validation_labels]
        test_labels = [int(label) for label in test_labels]

        all_paths = glob.glob(os.path.join(self.path, "audio", "*.wav"))

        # for each id look in the audio folder for the wav file containing the id string
        train_audio_paths = []
        for id in tqdm(train_ids, desc="Loading train set"):
            # search in the all_paths list for the path containing the id string
            train_audio_paths.append([path for path in all_paths if id in path][0])

        validation_audio_paths = []
        for id in tqdm(validation_ids, desc="Loading validation set"):
            validation_audio_paths.append([path for path in all_paths if id in path][0])

        test_audio_paths = []
        for id in tqdm(test_ids, desc="Loading test set"):
            test_audio_paths.append([path for path in all_paths if id in path][0])


        self.num_classes = len(set(train_labels))

        if self.verbose:
            print("Train set: ", len(train_audio_paths))
            print("Validation set: ", len(validation_audio_paths))
            print("Test set: ", len(test_audio_paths))
            # print some statistics - total number of audio files, number of classes
            print("Total number of audio files: ", len(train_audio_paths) + len(validation_audio_paths) + len(test_audio_paths))
            print (f"Number of classes: {self.num_classes}")

        return train_audio_paths, train_labels, validation_audio_paths, validation_labels, test_audio_paths, test_labels


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