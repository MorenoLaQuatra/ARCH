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

class MiviaRoad():
    '''
    This class implements the functionality to load the Mivia Road Audio Events dataset.
    It implements the fold-based evaluation as suggested by the authors.
    '''

    def __init__(
        self,
        path: str,
        verbose: bool = False,
    ):
        '''
        :param path: path to the dataset
        '''
        self.path = path
        self.verbose = verbose
        self._load_dataset()

    def _load_dataset(self):

        # the folder audio/ contains 4 subfolders, one for each fold A, B, C, D
        # each subfolder contains 1 xml file for each audio file (in the v2 sub subfolder)
        # the xml file contains the annotations for the audio file
        # the audio files are in the v2 sub subfolder

        self.labels = set()

        self.folds = {}

        for fold in ['A', 'B', 'C', 'D']:
            print(f"Loading fold {fold}")
            self.folds[fold] = {}
            # find keys from xml file names
            fold_path = os.path.join(self.path, 'audio', fold)
            print(f"Fold path: {fold_path}")

            # load xml files
            xml_files = glob.glob(os.path.join(fold_path, '*.xml'))
            print(f"Number of xml files: {len(xml_files)}")
            # load audio files
            audio_files = glob.glob(os.path.join(fold_path, 'v2', '*.wav'))
            print(f"Number of audio files: {len(audio_files)}")

            self.folds[fold]['keys'] = [os.path.basename(xml_file).split('.')[0] for xml_file in xml_files]
            self.folds[fold]['xml_files'] = xml_files
            self.folds[fold]['audio_files'] = audio_files

            # load annotations
            self.folds[fold]['annotations'] = {}
            for xml_file in xml_files:
                file_key = os.path.basename(xml_file).split('.')[0]
                # convert from xml to dict
                self.folds[fold]['annotations'][file_key] = []
                with open(xml_file) as fd:
                    d = xmltodict.parse(fd.read())
                    # root -> events -> item -> each element has CLASS_ID (#text), STARTSECOND(#text), ENDSECOND(#text)
                    for it in d['root']['events']['item']:
                        class_id = int(it['CLASS_ID']['#text'])
                        self.labels.add(class_id)
                        start_second = float(it['STARTSECOND']['#text'])
                        end_second = float(it['ENDSECOND']['#text'])
                        self.folds[fold]['annotations'][file_key].append([class_id, start_second, end_second])
                        print(f"File key: {file_key}, class_id: {class_id}, start_second: {start_second}, end_second: {end_second}")
            
            print(f"Number of labels: {len(self.labels)}")
            print(f"Labels: {self.labels}")

            # map all labels in range [0, n_labels-1]
            self.label2id = {}
            self.id2label = {}
            for i, label in enumerate(sorted(self.labels)):
                self.label2id[label] = i
                self.id2label[i] = label

            # convert annotations to labels
            for file_key in self.folds[fold]['annotations'].keys():
                for i, annotation in enumerate(self.folds[fold]['annotations'][file_key]):
                    class_id = annotation[0]
                    self.folds[fold]['annotations'][file_key][i][0] = self.label2id[class_id]

            n_samples = 0
            for file_key in self.folds[fold]['annotations'].keys():
                n_samples += len(self.folds[fold]['annotations'][file_key])
            print(f"Number of samples: {n_samples}")


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

        # the parameter model_frame_length_ms is mandatory for the evaluation of sequence classification models
        if model_frame_length_ms is None:
            raise ValueError('model_frame_length_ms is mandatory for the evaluation of sequence classification models, please specify it according to the model under evaluation')

        sampling_rate = model.get_sampling_rate()

        # iterate over folds
        results = {}
        for fold in self.folds.keys():
            
            test_audio_paths = self.folds[fold]['audio_files']
            test_audios = self.from_paths_to_audios(
                test_audio_paths, sampling_rate=sampling_rate
            )
            test_audio_durations_ms = [len(audio) / sampling_rate * 1000 for audio in test_audios]

            train_audio_paths = []
            for f in self.folds.keys():
                if f != fold:
                    train_audio_paths += self.folds[f]['audio_files']
            train_audios = self.from_paths_to_audios(
                train_audio_paths, sampling_rate=sampling_rate
            )
            train_audio_durations_ms = [len(audio) / sampling_rate * 1000 for audio in train_audios]

            # get max duration
            max_duration_ms = max(max(train_audio_durations_ms), max(test_audio_durations_ms))
            print(f"Max duration Mivia Road Dataset: {max_duration_ms} ms")

            # pad audios to max duration
            train_audios = self.pad_audios_to_max_duration(
                train_audios, max_duration_ms, sampling_rate=sampling_rate, return_tensors=True
            )
            test_audios = self.pad_audios_to_max_duration(
                test_audios, max_duration_ms, sampling_rate=sampling_rate, return_tensors=True
            )

            # assert all audios have the same duration
            print(f"Train audios.shape: {train_audios.shape}")
            print(f"Test audios.shape: {test_audios.shape}")


            # use annotations + model_frame_length_ms to create test labels
            test_labels = self.from_annotations_to_labels(
                [self.folds[fold]['annotations'][k] for k in self.folds[fold]['annotations'].keys()],
                model_frame_length_ms, 
                max_duration_ms = max_duration_ms,
                return_tensors=True
            )

            train_labels = self.from_annotations_to_labels(
                [self.folds[f]['annotations'][k] for f in self.folds.keys() if f != fold for k in self.folds[f]['annotations'].keys()],
                model_frame_length_ms,
                max_duration_ms = max_duration_ms,
                return_tensors=True
            )

            # split train set in train and validation
            train_audios, val_audios, train_labels, val_labels = train_test_split(
                train_audios, train_labels, test_size=0.2, random_state=42
            )

            # create dataset
            train_dataset = SequenceClassificationDataset(
                audios=train_audios,
                labels=train_labels,
                model = model,
                sampling_rate = sampling_rate,
                mode = mode,
                precompute_embeddings = True,
            )

            val_dataset = SequenceClassificationDataset(
                audios=val_audios,
                labels=val_labels,
                model = model,
                sampling_rate = sampling_rate,
                mode = mode,
                precompute_embeddings = True,
            )

            test_dataset = SequenceClassificationDataset(
                audios=test_audios,
                labels=test_labels,
                model = model,
                sampling_rate = sampling_rate,
                mode = mode,
                precompute_embeddings = True,
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

            seq_clf_model = SequenceClassificationModel(
                layers = [],
                input_embedding_size = model.get_token_embedding_size(),
                activation = 'relu',
                dropout = 0.1,
                num_classes = len(self.labels),
                is_multilabel = False,
                verbose = self.verbose,
            )

            seq_clf_model.train(
                train_dataloader = train_dataloader,
                val_dataloader = val_dataloader,
                device = device,
                max_num_epochs = max_num_epochs,
            )

            # evaluate on test set
            loss, m1, m2 = seq_clf_model.evaluate(
                data_loader = test_dataloader,
                device = device
            )

            results[fold] = {
                'loss': loss,
                'm1': m1,
                'm2': m2,
            }

            if self.verbose:
                print(f'Fold {fold} results:')
                print(results[fold])

            

        # compute average results
        avg_results = {
            'loss': np.mean([results[fold]['loss'] for fold in results.keys()]),
            'm1': np.mean([results[fold]['m1'] for fold in results.keys()]),
            'm2': np.mean([results[fold]['m2'] for fold in results.keys()]),
        }

        if self.verbose:
            print('Average results:')
            print(avg_results)

        return avg_results









    




