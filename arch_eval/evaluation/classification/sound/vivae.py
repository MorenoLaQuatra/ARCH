import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class VIVAE():
    '''
    This class implements the functionality to load the VIVAE dataset.
    It implements a speaker-based cross-validation.
    '''

    def __init__(
        self,
        path,
        verbose = False,
    ):

        self.path = path
        self.verbose = verbose
        self.dataset = self._load_data()

    def _load_data(self):
        '''
        Load the data and split it into train, validation and test sets.
        :return: a dictionary containing the audio paths and the labels divided by speaker.
        '''

        # get the audio files in "full_set" folder
        audio_paths = glob.glob(os.path.join(self.path, 'full_set', '*.wav'))
        speakers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

        # get the labels from the file names S<speaker>_<emotion>_<intensity>_<repetition>.wav
        labels = [os.path.basename(path).split('_')[1] for path in audio_paths]

        dataset = {}
        for speaker in speakers: # to avoid random order
            dataset[speaker] = {"audio_paths": [], "labels": []}

        for audio_path in audio_paths:
            # get the speaker and emotion from the file name
            speaker = os.path.basename(audio_path).split('_')[0].replace("S", "")
            emotion = os.path.basename(audio_path).split('_')[1]
            if speaker in dataset.keys():
                dataset[speaker]["audio_paths"].append(audio_path)
                dataset[speaker]["labels"].append(emotion)

        # map emotions to integers
        self.emotion_map = {emotion: i for i, emotion in enumerate(np.unique(labels))}
        self.inverse_emotion_map = {i: emotion for i, emotion in enumerate(np.unique(labels))}

        for speaker in dataset.keys():
            dataset[speaker]["labels"] = [self.emotion_map[label] for label in dataset[speaker]["labels"]]
        
        self.num_classes = len(self.emotion_map)
        if self.verbose:
            print("Number of classes: ", self.num_classes)
            for speaker in dataset.keys():
                print("Speaker: ", speaker)
                print("Number of samples: ", len(dataset[speaker]["labels"]))
                print("Number of classes: ", len(np.unique(dataset[speaker]["labels"])))
                print("Classes: ", np.unique(dataset[speaker]["labels"]))
                print("")

        return dataset



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

        speakers = list(self.dataset.keys())
        results = []
        for i_speaker, speaker in enumerate(speakers):
            if self.verbose:
                print("Speaker test: ", speaker)
                print("Speaker validation: ", speakers[(i_speaker + 1) % len(speakers)])
                print("Speaker train: ", [s for s in speakers if s != speaker and s != speakers[(i_speaker + 1) % len(speakers)]])

            # get the train, validation and test sets
            train_paths = []
            train_labels = []
            validation_paths = []
            validation_labels = []
            test_paths = []
            test_labels = []

            for i, s in enumerate(speakers):
                if i == i_speaker:
                    test_paths.extend(self.dataset[s]["audio_paths"])
                    test_labels.extend(self.dataset[s]["labels"])
                elif i == (i_speaker + 1) % len(speakers):
                    validation_paths.extend(self.dataset[s]["audio_paths"])
                    validation_labels.extend(self.dataset[s]["labels"])
                else:
                    train_paths.extend(self.dataset[s]["audio_paths"])
                    train_labels.extend(self.dataset[s]["labels"])

            clf_model = ClassificationModel(
                layers = layers,
                input_embedding_size = model.get_classification_embedding_size(),
                activation = "relu",
                dropout = 0.1,
                num_classes = self.num_classes,
                verbose = self.verbose,
            )

            # create train, validation and test datasets
            train_dataset = ClassificationDataset(
                audio_paths = train_paths,
                labels = train_labels,
                model = model,
                sampling_rate = model.get_sampling_rate(),
                precompute_embeddings = True,
            )

            val_dataset = ClassificationDataset(
                audio_paths = validation_paths,
                labels = validation_labels,
                model = model,
                sampling_rate = model.get_sampling_rate(),
                precompute_embeddings = True,
            )

            test_dataset = ClassificationDataset(
                audio_paths = test_paths,
                labels = test_labels,
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

            if self.verbose:
                print("Speaker: ", speaker)
                print("Metrics: ", metrics)
                print("")

            results.append(metrics)

        # compute the average metrics
        average_metrics = {}
        for metric in results[0].keys():
            average_metrics[metric] = np.mean([result[metric] for result in results])

        return average_metrics