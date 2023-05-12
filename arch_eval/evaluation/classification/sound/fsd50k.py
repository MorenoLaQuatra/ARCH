import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class FSD50K():
    '''
    This class implements the functionality to load the FSD50K dataset.
    It implements the original split of the dataset, as described in the paper:
    https://arxiv.org/pdf/2010.00475.pdf
    '''

    def __init__(
        self,
        path,
        verbose = False,
        precompute_embeddings: bool = False,
    ):

        self.path = path
        self.verbose = verbose
        self.is_multilabel = True
        self.precompute_embeddings = precompute_embeddings
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()

    def _load_data(self):
        '''
        Load the data and split it into train, validation and test sets.
        :return: a list of lists containing the audio paths and the labels
        '''

        # FSDD50K.ground_truth is the path to the ground truth files
        # dev.csv and eval.csv contain the metadata for the development and evaluation sets, respectively.
        # 

        df_train_val = pd.read_csv(os.path.join(self.path, 'FSD50K.ground_truth', 'dev.csv'))
        df_test = pd.read_csv(os.path.join(self.path, 'FSD50K.ground_truth', 'eval.csv'))


        # fname is the name of the audio file, labels is a list of comma-separated labels
        # split is the split of the data (train, validation, test)

        audio_paths_train = [
            os.path.join(self.path, 'FSD50K.dev_audio', str(fname) + '.wav')
            for fname in df_train_val['fname'] 
            if df_train_val['split'][df_train_val['fname'] == fname].values[0] == 'train'
        ]

        audio_paths_val = [
            os.path.join(self.path, 'FSD50K.dev_audio', str(fname) + '.wav')
            for fname in df_train_val['fname'] 
            if df_train_val['split'][df_train_val['fname'] == fname].values[0] == 'val'
        ]

        audio_paths_test = [
            os.path.join(self.path, 'FSD50K.eval_audio', str(fname) + '.wav')
            for fname in df_test['fname'] 
        ]
        # labels are a list of lists of labels 
        labels_train = [
            labels.split(',') 
            for labels, fname in zip(df_train_val['labels'], df_train_val['fname'])
            if df_train_val['split'][df_train_val['fname'] == fname].values[0] == 'train'
        ]

        labels_val = [
            labels.split(',') 
            for labels, fname in zip(df_train_val['labels'], df_train_val['fname'])
            if df_train_val['split'][df_train_val['fname'] == fname].values[0] == 'val'
        ]

        labels_test = [
            labels.split(',') 
            for labels in df_test['labels']
        ]

        # encode labels with integers
        le = preprocessing.LabelEncoder()
        # get all the labels
        all_labels = []
        for labels in labels_train:
            all_labels.extend(labels)
        for labels in labels_val:
            all_labels.extend(labels)
        for labels in labels_test:
            all_labels.extend(labels)

        all_labels = list(set(all_labels))
        le.fit(all_labels)
        self.num_classes = len(all_labels)

        # encode labels with integers
        labels_train = [le.transform(labels) for labels in labels_train]
        labels_val = [le.transform(labels) for labels in labels_val]
        labels_test = [le.transform(labels) for labels in labels_test]

        # convert to one-hot encoding - each index in the label list corresponds to a class
        # and the value is 1 if the class is present in the label list, 0 otherwise
        train_labels = np.zeros((len(labels_train), self.num_classes))
        for i, labels in enumerate(labels_train):
            train_labels[i, labels] = 1

        val_labels = np.zeros((len(labels_val), self.num_classes))
        for i, labels in enumerate(labels_val):
            val_labels[i, labels] = 1

        test_labels = np.zeros((len(labels_test), self.num_classes))
        for i, labels in enumerate(labels_test):
            test_labels[i, labels] = 1

        print(labels_train[0].shape)
        print(labels_train[0])

        # convert to tensors
        labels_train = torch.tensor(train_labels, dtype=torch.float32)
        labels_val = torch.tensor(val_labels, dtype=torch.float32)
        labels_test = torch.tensor(test_labels, dtype=torch.float32)
        

        if self.verbose:
            # print some statistics
            # total number of audio files
            print('Total number of audio files: {}'.format(len(audio_paths_train) + len(audio_paths_val) + len(audio_paths_test)))
            # number of audio files in the train set
            print('Number of audio files in the train set: {}'.format(len(audio_paths_train)))
            # number of audio files in the validation set
            print('Number of audio files in the validation set: {}'.format(len(audio_paths_val)))
            # number of audio files in the test set
            print('Number of audio files in the test set: {}'.format(len(audio_paths_test)))
            # number of classes
            print('Number of classes: {}'.format(self.num_classes))
        

        return audio_paths_train, labels_train, audio_paths_val, labels_val, audio_paths_test, labels_test

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