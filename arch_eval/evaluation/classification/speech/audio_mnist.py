import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split


class AudioMNIST():
    '''
    This class implements the functionality to load the AudioMNIST dataset
    and the recipe for its evaluation.
    It implements the fold-based evaluation, where each fold is a
    different split of the dataset.
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
        self.folds = self._load_folds()


    def _load_folds(self):
        '''
        Load the folds of the dataset.
        Folds are defined in the metadata file at meta/esc50.csv
        :return: a dictionary containing as keys the fold numbers
        and as values a dictionary with the following keys:
        - audio_paths: list of audio paths
        - labels: list of labels
        - readable_labels: list of readable labels
        '''

        speakers_fold_ids = [
            [12, 47, 59, 15, 27, 41,  4, 11, 21, 31, 44, 50],
            [26, 52, 60, 18, 32, 42,  5, 13, 22, 33, 45, 51],
            [28, 56,  7, 19, 35,  1,  6, 16, 23, 34, 46, 53],
            [36, 57,  9, 24, 37,  2,  8, 17, 29, 39, 48, 54],
            [43, 58, 14, 25, 38,  3, 10, 20, 30, 40, 49, 55]

        ]

        # data are in the folder self.path + 'data' - it contains one folder for each speaker - each folder contains multiple files
        # the files are named as : <digit>_<speaker_id>_<repetition_id>.wav
        # the digit is the label of the audio file

        # get all audio files paths
        audio_files = glob.glob(os.path.join(self.path, 'data', '*', '*.wav'))

        data = {}

        # iterate over the folds
        for fold_id, speakers in enumerate(speakers_fold_ids):
            data[fold_id] = {
                'audio_paths': [],
                'labels': [],
            }

            # iterate over the audio files
            for audio_file in audio_files:
                # get the speaker id from the file name
                speaker_id = int(audio_file.split('/')[-1].split('_')[1])

                # if the speaker is in the current fold, add the audio file to the fold
                if speaker_id in speakers:
                    data[fold_id]['audio_paths'].append(audio_file)
                    data[fold_id]['labels'].append(int(audio_file.split('/')[-1].split('_')[0]))

        # set the number of classes
        self.num_classes = len(set(data[0]['labels']))
        if self.verbose:
            print(f"Number of folds: {len(data.keys())}")
            print(f"Total number of audio files: {len(audio_files)}")
            print(f"Number of classes: {self.num_classes}")
        return data

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
        Evaluate the model on the dataset running the 5-fold cross-validation.
        :param model: the self-supervised model to evaluate, it must be an instance of Model
        :param mode: the mode to use for the evaluation, it can be either 'linear' or 'non-linear'
        :param device: the device to use for the evaluation, it can be either 'cpu' or 'cuda'
        :param batch_size: the batch size to use for the evaluation
        :param num_workers: the number of workers to use for the evaluation
        :param max_num_epochs: the maximum number of epochs to use for the evaluation
        :return: a dictionary containing the results of the evaluation
        '''

        # Iterate over the folds
        results = {}

        for fold in sorted(list(self.folds.keys())):

            if self.verbose:
                print(f"Fold {fold} of {len(self.folds.keys())}")

            if mode == 'linear':
                layers = []
            elif mode == 'non-linear':
                layers = [model.get_classification_embedding_size()]
            elif mode == 'attention-pooling':
                layers = []
            else:
                raise ValueError(f"Invalid mode {mode}")

            clf_model = ClassificationModel(
                layers=layers,
                input_embedding_size=model.get_classification_embedding_size(),
                activation='relu',
                dropout=0.1,
                num_classes=self.num_classes,
                verbose=self.verbose,
                is_multilabel = self.is_multilabel,
                mode = mode,
            )

            # Create train, validation and test datasets
            audio_paths_train = []
            labels_train = []
            audio_paths_test = []
            labels_test = []
            audio_paths_val = []
            labels_val = []

            # fold - is the test fold
            # the next fold is the validation fold
            # the rest are the training folds

            # iterate over the folds
            for fold_id in sorted(list(self.folds.keys())):
                # if the fold is the test fold
                if fold_id == fold:
                    audio_paths_test.extend(self.folds[fold_id]['audio_paths'])
                    labels_test.extend(self.folds[fold_id]['labels'])
                # if the fold is the validation fold
                elif fold_id == (fold + 1) % len(self.folds.keys()):
                    audio_paths_val.extend(self.folds[fold_id]['audio_paths'])
                    labels_val.extend(self.folds[fold_id]['labels'])
                # if the fold is a training fold
                else:
                    audio_paths_train.extend(self.folds[fold_id]['audio_paths'])
                    labels_train.extend(self.folds[fold_id]['labels'])


            # Create the datasets
            train_dataset = ClassificationDataset(
                audio_paths=audio_paths_train,
                labels=labels_train,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings = self.precompute_embeddings,
                mode = mode,
            )

            val_dataset = ClassificationDataset(
                audio_paths=audio_paths_val,
                labels=labels_val,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings = self.precompute_embeddings,
                mode = mode,
            )

            test_dataset = ClassificationDataset(
                audio_paths=audio_paths_test,
                labels=labels_test,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings = self.precompute_embeddings,
                mode = mode,
            )

            # create data loaders
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            # train the model
            clf_model.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                max_num_epochs=max_num_epochs,
                device=device,
            )

            # evaluate the model
            metrics = clf_model.evaluate(
                dataloader=test_dataloader,
                device=device,
            )
            
            if self.verbose:
                for metric in metrics.keys():
                    print(f"{metric}: {metrics[metric]}")
                
            results[fold] = metrics

        # compute the average results - independently of the names of the metrics
        avg_results = {}
        for metric in results[0].keys():
            avg_results[metric] = np.mean([results[fold][metric] for fold in results.keys()])

        return avg_results
