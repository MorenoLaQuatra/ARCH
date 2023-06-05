import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class FMASmall():
    '''
    This class implements the functionality to load the FMA-small dataset.
    It implements a train/test split of the dataset (random split with seed 42).
    '''

    def __init__(
        self,
        path: str,
        verbose = False,
        precompute_embeddings = False,
    ):
        '''
        :param config_path: path to the folder containing the config files (fma_metadata)
        :param audio_files_path: path to the folder containing the audio files (fma_small)
        :param verbose: if True, print some information about the dataset
        '''

        self.config_path = path + "fma_metadata/"
        self.audio_files_path = path + "fma_small/"
        self.verbose = verbose
        self.is_multilabel = False
        self.precompute_embeddings = precompute_embeddings
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()

    def _load_data(self):
        '''
        Load the train and test splits of the dataset.
        :return: a dictionary containing as keys the split names
        and as values a dictionary with the following keys:
        - audio_paths: list of audio paths
        - labels: list of labels
        - readable_labels: list of readable labels
        '''
        # load the tracks.csv file
        tracks = pd.read_csv(os.path.join(self.config_path, 'tracks.csv'), index_col=0, header=[0, 1])
        # get track ids
        #track_ids = tracks.index.values

        # labels : track -> genre_top - drop rows with NaN
        tracks = tracks.dropna(subset=[('track', 'genre_top')])
        labels = tracks[('track', 'genre_top')].values
        # convert labels to integers
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        self.num_classes = len(np.unique(labels))
        # audio paths: df -> track_id
        audio_paths = tracks.index.values
        # 6-digit format for track_id
        audio_paths = [os.path.join(self.audio_files_path, str(track_id).zfill(6) + '.mp3') for track_id in audio_paths]
        # remove audio files that do not exist - take care of the labels
        audio_paths, labels = zip(*[(audio_path, label) for audio_path, label in zip(audio_paths, labels) if os.path.exists(audio_path)])

        if self.verbose:
            print ("Original metadata shape: ", tracks.shape)
            print ("FMA-small parsed data: ", len(audio_paths))
            # print some statistics - total number of audio files, number of classes
            print ("Total number of audio files: ", len(audio_paths))
            print ("Number of classes: ", self.num_classes)

        # split the dataset into train, validation and test - 80% train, 10% validation, 10% test
        # use a random split with seed 42
        train_audio_paths, test_audio_paths, train_labels, test_labels = train_test_split(audio_paths, labels, test_size=0.2, random_state=42)
        test_audio_paths, val_audio_paths, test_labels, val_labels = train_test_split(test_audio_paths, test_labels, test_size=0.5, random_state=42)

        return train_audio_paths, train_labels, val_audio_paths, val_labels, test_audio_paths, test_labels

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
        Evaluate the model on the dataset running train/validation/test tests.
        :param model: the self-supervised model to evaluate, it must be an instance of Model
        :param mode: the mode to use for the evaluation, it can be either 'linear' or 'non-linear'
        :param device: the device to use for the evaluation, it can be either 'cpu' or 'cuda'
        :param batch_size: the batch size to use for the evaluation
        :param num_workers: the number of workers to use for the evaluation
        :param max_num_epochs: the maximum number of epochs to use for the evaluation
        :return: a dictionary containing the results of the evaluation
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
            is_multilabel = False,
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

        return {
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
        }





