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


class MIR1K():
    '''
    This class is used to load and evaluate sequence classification models on the MIR1K dataset.
    '''

    def __init__(
        self,
        path: str,
        verbose: bool = False,
    ):
        '''
        Args:
            path (str): Path to the root directory of the MIR1K dataset.
            verbose (bool): If True, print additional information.
        '''
        self.path = path
        self.verbose = verbose
        self._load()

    def _load(self):
        '''
        Load the MIR1K dataset and provide a unified interface, similar to the other datasets.
        In this case, the dataset is not provided with a train/test split and we need to create it ourselves.
        '''

        # wav files are inside the "Wavfile" folder
        audio_files = glob.glob(os.path.join(self.path, "Wavfile", "*.wav"))

        # labels are provided in a separate folder vocal-nonvocalLabel in files with .vocal extension
        label_files = glob.glob(os.path.join(self.path, "vocal-nonvocalLabel", "*.vocal"))

        # each annotation file contains a list of labels, one per line, 0 for non-vocal and 1 for vocal - each label corresponds to 20 ms
        # we need to convert this to start:end timestamps in seconds

        annotations = []
        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()
                lines = [int(line.strip()) for line in lines]
                # convert to timestamps
                timestamps = []
                # aggregate consecutive 1s
                start = None
                for i, label in enumerate(lines):
                    if label == 1:
                        if start is None:
                            start = i
                    else:
                        if start is not None:
                            timestamps.append((1, start, i)) # label = 1 for vocal - we don't care about non-vocal
                            start = None
                # add last one
                if start is not None:
                    timestamps.append((1, start, i)) # label = 1 for vocal - we don't care about non-vocal
                # convert to seconds

                timestamps = [(label, start * 0.02, end * 0.02) for label, start, end in timestamps]
                annotations.append(timestamps)

        print ("Number of audio files: ", len(audio_files))
        print ("Number of annotations: ", len(annotations))

        self.audio_files = audio_files
        self.annotations = annotations
        self.labels = set([label for annotation in annotations for label, _, _ in annotation])
        # add label 0 for non-vocal
        self.labels.add(0)
        print ("Labels: ", self.labels)



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
        # pad with zeros to the maximum duration
        max_duration_samples = int(np.ceil( (max_duration_ms / 1000) * sampling_rate ))
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

        if model_frame_length_ms is None:
            raise ValueError('model_frame_length_ms is mandatory for the evaluation of sequence classification models, please specify it according to the model under evaluation')

        sampling_rate = model.get_sampling_rate()

        audio_paths = self.audio_files
        annotations = self.annotations
        audios = self.from_paths_to_audios(audio_paths, sampling_rate=sampling_rate)
        

        # train/validation/test split
        train_audios, test_audios, train_labels, test_labels = train_test_split(audios, annotations, test_size=0.2, random_state=42)

        test_audios, val_audios, test_labels, val_labels = train_test_split(test_audios, test_labels, test_size=0.5, random_state=42)

        # pad audios to the maximum duration
        max_duration_ms = max([audio.shape[0] for audio in audios]) / (sampling_rate / 1000)
        train_audios = self.pad_audios_to_max_duration(train_audios, max_duration_ms, sampling_rate, return_tensors=True)
        val_audios = self.pad_audios_to_max_duration(val_audios, max_duration_ms, sampling_rate, return_tensors=True)
        test_audios = self.pad_audios_to_max_duration(test_audios, max_duration_ms, sampling_rate, return_tensors=True)

        # convert annotations to labels
        train_labels = self.from_annotations_to_labels(train_labels, model_frame_length_ms, max_duration_ms, return_tensors=True)
        val_labels = self.from_annotations_to_labels(val_labels, model_frame_length_ms, max_duration_ms, return_tensors=True)
        test_labels = self.from_annotations_to_labels(test_labels, model_frame_length_ms, max_duration_ms, return_tensors=True)

        print("Max duration: ", max_duration_ms)

        print ('train_audios.shape', train_audios.shape)
        print ('val_audios.shape', val_audios.shape)
        print ('test_audios.shape', test_audios.shape)

        print ('train_labels.shape', train_labels.shape)
        print ('val_labels.shape', val_labels.shape)
        print ('test_labels.shape', test_labels.shape)

        # create dataset
        train_dataset = SequenceClassificationDataset(
            audios=train_audios,
            labels=train_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=True,
        )

        val_dataset = SequenceClassificationDataset(
            audios=val_audios,
            labels=val_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=True,
        )

        test_dataset = SequenceClassificationDataset(
            audios=test_audios,
            labels=test_labels,
            model=model,
            sampling_rate=sampling_rate,
            mode=mode,
            precompute_embeddings=True,
        )

        # create dataloaders
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

        # create model
        if mode == 'linear':
            layers = []
        elif mode == 'non-linear':
            layers = [model.get_token_embedding_size()]

        seq_clf_model = SequenceClassificationModel(
            layers = layers, 
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

        
