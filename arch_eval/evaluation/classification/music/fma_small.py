import os
import glob
import pandas as pd
import numpy as np
import torch
import torchaudio

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

    @staticmethod
    def _audio_path(audio_files_path, track_id):
        """Resolve an official nested path, with legacy flat-layout fallback."""
        track_filename = f"{int(track_id):06d}.mp3"
        nested_path = os.path.join(
            audio_files_path,
            track_filename[:3],
            track_filename,
        )
        if os.path.isfile(nested_path):
            return nested_path
        flat_path = os.path.join(audio_files_path, track_filename)
        if os.path.isfile(flat_path):
            return flat_path
        return nested_path

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

        # The shared metadata covers every FMA subset. Restrict it explicitly when
        # that column is available, while retaining compatibility with older files.
        if ('set', 'subset') in tracks.columns:
            tracks = tracks[tracks[('set', 'subset')] == 'small']

        # labels : track -> genre_top - drop rows with NaN
        tracks = tracks.dropna(subset=[('track', 'genre_top')])
        readable_labels = tracks[('track', 'genre_top')].values
        audio_paths = [
            self._audio_path(self.audio_files_path, track_id)
            for track_id in tracks.index.values
        ]

        # Metadata describes all FMA subsets. Filter to the files in the downloaded
        # archive before encoding labels so FMA-small has its actual eight classes.
        available_examples = [
            (audio_path, label)
            for audio_path, label in zip(audio_paths, readable_labels)
            if os.path.isfile(audio_path)
        ]
        if not available_examples:
            raise FileNotFoundError(
                "No FMA audio files matched the metadata under "
                f"{self.audio_files_path!r}. A standard extraction must retain "
                "the nested layout fma_small/000/000002.mp3; a legacy flat "
                "fma_small/000002.mp3 layout is also accepted."
            )
        audio_paths, readable_labels = zip(*available_examples)

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(readable_labels)
        self.num_classes = len(le.classes_)

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

    def get_average_duration(self):
        '''
        Compute the average duration of the audio files in the dataset.
        :return: the average duration of the audio files in the dataset
        '''
        durations = []
        audio_paths = self.train_paths + self.validation_paths + self.test_paths
        audio_paths = list(set(audio_paths))
        for audio_path in audio_paths:
            try:
                audio, sr = torchaudio.load(audio_path)
            except Exception as e:
                print (e)
                print (audio_path)
                continue
            durations.append(audio.shape[1] / sr)
        return torch.tensor(durations).mean().item()

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



