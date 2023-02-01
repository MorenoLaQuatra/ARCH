import os
import glob
import pandas as pd
import numpy as np
import torch
import xmltodict
import json
import torchaudio

from typing import List, Tuple

from arch_eval import Model, SequenceClassificationModel
from arch_eval import SequenceClassificationDataset

from sklearn.model_selection import train_test_split


class Jamendo():
    '''
    This class is used to load and evaluate sequence classification models on the Jamendo dataset.
    '''

    def __init__(
        self,
        path: str,
        verbose: bool = False,
    ):
        '''
        :param path: Path to the Jamendo dataset.
        :param verbose: If True, print information about the dataset.
        '''
        self.path = path
        self.verbose = verbose

        self.labels_mapping = {
            "nosing": 0,
            "sing": 1,
        }
        self._load_dataset()

    def _load_annotations(self, filenames: List[str]) -> List[List[str]]:
        '''
        Load the annotations for the Jamendo dataset.
        :param filenames: List of filenames of the annotations.
        :return: List of annotations.
        '''
        

        annotations = []
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
                # remove newlines
                lines = [line.strip() for line in lines]
                # split each line by space - start end label
                lines = [line.split(' ') for line in lines]
                # store (label, start, end) for each line
                lines = [(self.labels_mapping[line[2]], float(line[0]), float(line[1])) for line in lines]
                annotations.append(lines)
        return annotations

    def _load_filenames(self, filename: str) -> List[str]:
        '''
        Load the filenames for the Jamendo dataset.
        :param filename: Path to the file containing the filenames.
        :return: List of filenames.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            return lines

    def _load_dataset(self):
        '''
        Load the Jamendo dataset using provided train/validation/test splits.
        '''

        # filelists folder contains one file per split - train, valid, test
        filelists_path = os.path.join(self.path, 'filelists')
        # load the filenames for each split 
        train_filenames = self._load_filenames(os.path.join(filelists_path, 'train'))
        validation_filenames = self._load_filenames(os.path.join(filelists_path, 'valid'))
        test_filenames = self._load_filenames(os.path.join(filelists_path, 'test'))

        # create labels filenames - the same of filenames but with .lab extension - remove the extension (after the last dot) and add .lab
        train_labels_filenames = [filename.rsplit('.', 1)[0] + '.lab' for filename in train_filenames]
        validation_labels_filenames = [filename.rsplit('.', 1)[0] + '.lab' for filename in validation_filenames]
        test_labels_filenames = [filename.rsplit('.', 1)[0] + '.lab' for filename in test_filenames]

        # add the path to the labels folder
        train_labels_filenames = [os.path.join(self.path, 'labels', filename) for filename in train_labels_filenames]
        validation_labels_filenames = [os.path.join(self.path, 'labels', filename) for filename in validation_labels_filenames]
        test_labels_filenames = [os.path.join(self.path, 'labels', filename) for filename in test_labels_filenames]

        # add the path to the audio folder
        train_filenames = [os.path.join(self.path, 'audio', filename) for filename in train_filenames]
        validation_filenames = [os.path.join(self.path, 'audio', filename) for filename in validation_filenames]
        test_filenames = [os.path.join(self.path, 'audio', filename) for filename in test_filenames]

        # load the labels for each split
        train_annotations = self._load_annotations(train_labels_filenames)
        validation_annotations = self._load_annotations(validation_labels_filenames)
        test_annotations = self._load_annotations(test_labels_filenames)

        self.train_audio_filenames = train_filenames
        self.validation_audio_filenames = validation_filenames
        self.test_audio_filenames = test_filenames

        self.train_annotations = train_annotations
        self.validation_annotations = validation_annotations
        self.test_annotations = test_annotations

        print ("Train set size: ", len(self.train_audio_filenames))
        print ("Validation set size: ", len(self.validation_audio_filenames))
        print ("Test set size: ", len(self.test_audio_filenames))

        self.labels = list(self.labels_mapping.keys())

        return 

    def from_annotations_to_labels(
        self,
        annotations: List,
        model_frame_length_ms: float,
        max_duration_ms: float,
        return_tensors: bool = False,
    ):
        '''
        This function converts the annotations to labels.
        E.g., [class_id, start_second, end_second] -> [class_id, class_id, class_id, ...]
        The labels are represented as a list of integers.
        :param annotations: list of annotations
        :param model_frame_length_ms: frame length of the model
        :param max_duration_ms: maximum duration of the audio in the dataset. It is used for padding labels.
        :param return_tensors: if True, the labels are returned as a tensor
        :return: list of labels
        '''
        n_frames = int(np.floor(max_duration_ms / model_frame_length_ms))
        # convert annotations to labels
        labels = []
        for i, annotation in enumerate(annotations):
            # initialize label
            label = torch.zeros(n_frames, dtype=int)
            # fill label
            for ann in annotation:
                class_id = ann[0]
                start_frame = int(np.floor(float(ann[1] * 1000) / model_frame_length_ms))
                end_frame = int(np.floor(float(ann[2] * 1000) / model_frame_length_ms))
                label[start_frame:end_frame] = class_id
            labels.append(label)
        
        if return_tensors:
            return torch.stack(labels)
        else:
            return labels


    def from_paths_to_audios(
        self, 
        audio_paths,
        sampling_rate: int = 16000,
    ):
        '''
        This function loads the audios from the paths.
        :param audio_paths: list of paths to the audios
        :return: list of audio vectors
        '''

        if self.verbose:
            print ("Loading audios...")

        # load audios
        audios = []
        for audio_path in audio_paths:
            audio, sr = torchaudio.load(audio_path)
            # resample if needed
            if sr != sampling_rate:
                audio = torchaudio.transforms.Resample(sr, sampling_rate)(audio)

            # convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, axis=0) 
            
            audios.append(audio.squeeze())
        
        return audios

    def pad_audios_to_max_duration(
        self,
        audios: List,
        max_duration_ms: float,
        sampling_rate: int = 16000,
        return_tensors: bool = False,
    ):
        '''
        This function pads the audios to the maximum duration.
        :param audios: list of audio vectors
        :param max_duration_ms: maximum duration of the audio in the dataset. It is used for padding.
        :return: list or tensor of padded audio vectors
        '''

        if self.verbose:
            print ("Padding audios to max duration...")
        
        # pad with zeros to the maximum duration
        max_duration_samples = int(np.ceil( (max_duration_ms / 1000) * sampling_rate ))
        if self.verbose:
            print ("Max duration (samples): ", max_duration_samples)
        padded_audios = []
        for audio in audios:
            padded_audio = torch.zeros(max_duration_samples)
            padded_audio[:audio.shape[0]] = audio
            padded_audios.append(padded_audio)

        if return_tensors:
            return torch.stack(padded_audios)
        else:
            return padded_audios

    def evaluate(
        self,
        model: Model,
        mode: str = 'linear',
        device: str = 'cpu',
        batch_size: int = 32,
        num_workers: int = 0,
        max_num_epochs: int = 100,
        model_frame_length_ms: float = 20.00001,
        **kwargs,
    ):
        '''
        Evaluate the model on the dataset using the fold-based evaluation.
        :param model: model to evaluate, it must be an instance of Model
        :param mode: mode of the evaluation, it can be 'linear' or 'non-linear'
        :param device: device to use for the evaluation, it can be 'cpu' or 'cuda'
        :param batch_size: batch size to use for the evaluation
        :param num_workers: number of workers to use for the evaluation
        :param max_num_epochs: maximum number of epochs to use for the evaluation
        :param kwargs: additional arguments for the evaluation
        :return: dictionary containing the evaluation results
        '''

        if self.verbose:
            print ("Evaluating model...")

        if model_frame_length_ms is None:
            raise ValueError('model_frame_length_ms is mandatory for the evaluation of sequence classification models, please specify it according to the model under evaluation')

        sampling_rate = model.get_sampling_rate()

        train_audios = self.from_paths_to_audios(self.train_audio_filenames, sampling_rate=sampling_rate)
        validation_audios = self.from_paths_to_audios(self.validation_audio_filenames, sampling_rate=sampling_rate)
        test_audios = self.from_paths_to_audios(self.test_audio_filenames, sampling_rate=sampling_rate)

        # compute the maximum duration of the audios in the dataset
        max_duration_ms = max(
            max([audio.shape[0] for audio in train_audios]) / sampling_rate * 1000,
            max([audio.shape[0] for audio in validation_audios]) / sampling_rate * 1000,
            max([audio.shape[0] for audio in test_audios]) / sampling_rate * 1000,
        )

        if self.verbose:
            print ("Max duration: {} ms".format(max_duration_ms), " - in seconds: {} s".format(max_duration_ms / 1000))

        # pad the audios to the maximum duration
        train_audios = self.pad_audios_to_max_duration(train_audios, max_duration_ms, sampling_rate=sampling_rate, return_tensors=True)
        validation_audios = self.pad_audios_to_max_duration(validation_audios, max_duration_ms, sampling_rate=sampling_rate, return_tensors=True)
        test_audios = self.pad_audios_to_max_duration(test_audios, max_duration_ms, sampling_rate=sampling_rate, return_tensors=True)

        # convert annotations to labels
        train_labels = self.from_annotations_to_labels(self.train_annotations, model_frame_length_ms, max_duration_ms, return_tensors=True)
        validation_labels = self.from_annotations_to_labels(self.validation_annotations, model_frame_length_ms, max_duration_ms, return_tensors=True)
        test_labels = self.from_annotations_to_labels(self.test_annotations, model_frame_length_ms, max_duration_ms, return_tensors=True)

        # create the dataset
        train_dataset = SequenceClassificationDataset(
            audios=train_audios,
            labels=train_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=False,
        )

        val_dataset = SequenceClassificationDataset(
            audios=validation_audios,
            labels=validation_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=False,
        )

        test_dataset = SequenceClassificationDataset(
            audios=test_audios,
            labels=test_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=False,
        )

        # create the dataloaders
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

        # create the model
        seq_clf_model = SequenceClassificationModel(
            layers = [], 
            input_embedding_size = model.get_token_embedding_size(),
            activation = 'relu',
            dropout = 0.1,
            num_classes = len(self.labels),
            is_multilabel = False,
            verbose = self.verbose,
            model_frame_length_ms = model_frame_length_ms,
        )

        # train model
        seq_clf_model.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            max_num_epochs=max_num_epochs,
        )

        # evaluate model
        eval_metrics = seq_clf_model.evaluate(
            data_loader = test_dataloader,
            device = device,
        )

        if self.verbose:
            print('Evaluation metrics:')
            print(eval_metrics)

        return eval_metrics

