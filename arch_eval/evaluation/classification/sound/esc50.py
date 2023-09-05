import os
import glob
import pandas as pd
import numpy as np
import torch
import torchaudio

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split


class ESC50():
    '''
    This class implements the functionality to load the ESC-50 dataset 
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
        '''
        :param path: path to the dataset
        '''

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
        
        # Read metadata at meta/esc50.csv
        metadata = pd.read_csv(os.path.join(self.path, 'meta', 'esc50.csv'))

        # Get the number of folds at column 'fold'
        folds = metadata['fold'].unique()

        # Get the number of classes at column 'target' and set it at self.num_classes
        self.num_classes = len(metadata['target'].unique())

        # iterate over metadata rows
        # the audio paths (column 'filename', root=self.path + 'audio')
        # and the labels (column 'target')
        # and readable labels (column 'category')
        data = {}
        for fold in folds:
            data[fold] = {
                'audio_paths': [],
                'labels': [],
                'readable_labels': [],
            }
            for _, row in metadata[metadata['fold'] == fold].iterrows():
                data[fold]['audio_paths'].append(os.path.join(self.path, 'audio', row['filename']))
                data[fold]['labels'].append(row['target'])
                data[fold]['readable_labels'].append(row['category'])
        
        if self.verbose:
            print (f"Loaded {len(data.keys())} folds")
            # total number of samples
            print (f"Total number of samples: {sum([len(data[fold]['audio_paths']) for fold in data.keys()])}")
            # number of classes
            print (f"Number of classes: {self.num_classes}")

        return data

    def get_average_duration(self):
        '''
        Compute the average duration of the audio files in the dataset.
        :return: the average duration of the audio files in the dataset
        '''
        durations = []
        audio_paths = []
        for fold in self.folds.keys():
            audio_paths += self.folds[fold]['audio_paths']

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
        for fold in self.folds.keys():
            if self.verbose:
                print(f"Fold {fold} of {len(self.folds.keys())}")
            
            # fold is the test set
            # the other folds are the training set

            # Create a classification model
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
                mode=mode,
            )

            # Create train, validation and test datasets
            audio_paths_train = []
            labels_train = []
            audio_paths_test = []
            labels_test = []

            for fold_train in self.folds.keys():
                if fold_train != fold:
                    audio_paths_train += self.folds[fold_train]['audio_paths']
                    labels_train += self.folds[fold_train]['labels']
                else:
                    audio_paths_test += self.folds[fold_train]['audio_paths']
                    labels_test += self.folds[fold_train]['labels']

            # split train into train and validation
            # 80% train, 20% validation
            audio_paths_train, audio_paths_val, labels_train, labels_val = train_test_split(
                audio_paths_train,
                labels_train,
                test_size=0.2,
                random_state=42,
            )

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

        # compute the average results
        avg_results = {
            'loss': np.mean([results[fold]['loss'] for fold in results.keys()]),
            'accuracy': np.mean([results[fold]['accuracy'] for fold in results.keys()]),
            'f1': np.mean([results[fold]['f1'] for fold in results.keys()]),
        }

        return avg_results



