import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

class IRMAS():
    '''
    This class implements the functionality to load the IRMAS dataset
    It follows the authors' provided train/test split. The validation set is created by splitting the training set.
    '''

    def __init__(
        self,
        path,
        verbose = False,
    ):

        self.path = path
        self.verbose = verbose
        self.is_multilabel = True
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()

    def _load_data(self):
        '''
        Load the data and split it into train, validation and test sets.
        :return: a list of lists containing the audio paths and the labels
        '''

        # IRMAS-TestingData-Part1/ IRMAS-TestingData-Part2/ IRMAS-TestingData-Part3/ contains the test data
        # IRMAS-TrainingData/ contains the training data

        # load the training data
        # cel, cla, flu, gac, gel, org, pia, sax, tru, vio, voi
        train_paths = []
        train_labels = []
        for instrument in ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']:
            for audio_path in glob.glob(os.path.join(self.path, 'IRMAS-TrainingData', instrument, '*.wav')):
                train_paths.append(audio_path)
                train_labels.append([instrument])

        # load the test data
        test_paths = []
        test_labels = []

        parts = ['Part1', 'Part2', 'Part3']
        for part in parts:
            for audio_path in glob.glob(os.path.join(self.path, 'IRMAS-TestingData-' + part, part, '*.wav')):
                test_paths.append(audio_path)
                # the label is provided in the txt file that has the same name as the audio file - list of strings \n-separated
                labels = []
                with open(audio_path.replace('.wav', '.txt'), 'r') as f:
                    for line in f:
                        labels.append(line.strip())
                test_labels.append(labels)

        # split the training data into train and validation sets
        train_paths, validation_paths, train_labels, validation_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)

        # convert the labels
        multi_label_binarizer = MultiLabelBinarizer()
        train_labels = multi_label_binarizer.fit_transform(train_labels)
        validation_labels = multi_label_binarizer.transform(validation_labels)
        test_labels = multi_label_binarizer.transform(test_labels)

        self.num_classes = len(multi_label_binarizer.classes_)

        if self.verbose:
            # print some statistics - total number of audio files, number of classes
            print ("Total number of audio files: ", len(train_paths) + len(validation_paths) + len(test_paths))
            print ("Number of classes: ", self.num_classes)

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
            is_multilabel = self.is_multilabel,
        )

        # create train, validation and test datasets
        train_dataset = ClassificationDataset(
            audio_paths = self.train_paths,
            labels = self.train_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = True,
        )

        val_dataset = ClassificationDataset(
            audio_paths = self.validation_paths,
            labels = self.validation_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = True,
        )

        test_dataset = ClassificationDataset(
            audio_paths = self.test_paths,
            labels = self.test_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = True,
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