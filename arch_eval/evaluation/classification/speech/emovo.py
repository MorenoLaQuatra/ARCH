import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class EMOVO():
    '''
    This class implements the functionality to load the EMOVO dataset.
    It implements a speaker-based split.
    '''

    def __init__(
        self,
        path,
        verbose = False,
    ):

        self.path = path
        self.verbose = verbose
        self.folds = self._load_data()

    def _load_data(self):
        '''
        Load the data and divide it into folds, one for each speaker.
        :return: a dictionary containing the folds
        '''

        speakers = [
            "f1", "f2", "f3",
            "m1", "m2", "m3",
        ]

        labels_mapping = {
            "dis": 0,
            "gio": 1,
            "neu": 2,
            "pau": 3,
            "rab": 4,
            "sor": 5,
            "tri": 6,
        }

        data = {}
        for speaker in speakers:
            data[speaker] = {
                'audio_paths': [],
                'labels': [],
                'readable_labels': [],
            }
            # get the list of audio files, self.path + "EMOVO" + speaker + "/*.wav"
            audio_files = glob.glob(self.path + "EMOVO/" + speaker + "/*.wav")
            for audio_file in audio_files:
                # structure is <label>_<speaker>_<number>.wav
                label = audio_file.split("/")[-1]
                label = label.split("-")[0]
                data[speaker]['audio_paths'].append(audio_file)
                data[speaker]['labels'].append(labels_mapping[label])
                data[speaker]['readable_labels'].append(label)

        if self.verbose:
            # print some statistics
            for speaker in speakers:
                print("Speaker: ", speaker)
                print("Number of samples: ", len(data[speaker]['audio_paths']))
                print("Number of classes: ", len(np.unique(data[speaker]['labels'])))
        
        self.num_classes = len(labels_mapping)

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

            # Create a classification model
            if mode == 'linear':
                layers = []
            elif mode == 'non-linear':
                layers = [model.get_classification_embedding_size()]
            else:
                raise ValueError(f"Invalid mode {mode}")

            clf_model = ClassificationModel(
                layers=layers,
                input_embedding_size=model.get_classification_embedding_size(),
                activation='relu',
                dropout=0.1,
                num_classes=self.num_classes,
                verbose=self.verbose,
            )

            # Create train, validation and test datasets
            audio_paths_train = []
            labels_train = []
            audio_paths_test = []
            labels_test = []
            audio_paths_val = []
            labels_val = []

            # fold - is the test fold
            # all the other folds are the samples that should be used for training and validation

            # iterate over the folds
            for fold_id in sorted(list(self.folds.keys())):
                # if the fold is the test fold
                if fold_id == fold:
                    audio_paths_test.extend(self.folds[fold_id]['audio_paths'])
                    labels_test.extend(self.folds[fold_id]['labels'])
                else:
                    audio_paths_train.extend(self.folds[fold_id]['audio_paths'])
                    labels_train.extend(self.folds[fold_id]['labels'])

            # split the train set into train and validation
            audio_paths_train, audio_paths_val, labels_train, labels_val = train_test_split(
                audio_paths_train,
                labels_train,
                test_size=0.1,
                random_state=42,
            )
            
            # Create the datasets
            train_dataset = ClassificationDataset(
                audio_paths=audio_paths_train,
                labels=labels_train,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings=True,
            )

            val_dataset = ClassificationDataset(
                audio_paths=audio_paths_val,
                labels=labels_val,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings=True,
            )

            test_dataset = ClassificationDataset(
                audio_paths=audio_paths_test,
                labels=labels_test,
                model=model,
                sampling_rate=model.get_sampling_rate(),
                precompute_embeddings=True,
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
        # get the key for the first fold
        key = list(results.keys())[0]
        for metric in results[key].keys():
            avg_results[metric] = np.mean([results[fold][metric] for fold in results.keys()])

        return avg_results
