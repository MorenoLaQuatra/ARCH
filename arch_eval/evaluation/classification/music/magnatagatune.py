import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class MagnaTagATune():
    '''
    This class implements the functionality to load the MagnaTagATune dataset
    It implements a train/validation/test split provided by MusiCNN authors.
    '''

    def __init__(
        self,
        path,
        verbose = False,
        is_top50 = True,
    ):

        self.path = path
        self.verbose = verbose
        self.is_multilabel = True
        self.is_top50 = is_top50
        self.train_paths, self.train_labels, self.validation_paths, self.validation_labels, self.test_paths, self.test_labels = self._load_data()

    def _load_data(self):
        '''
        Load the data and split it into train, validation and test sets.
        :return: a list of lists containing the audio paths and the labels
        '''

        # load the file self.path + "annotations_final.csv"
        df = pd.read_csv(self.path + "annotations_final.csv", sep="\t")

        # the mp3_path column contains the path to the mp3 file, clip_id contains the id of the clip, all the others are 1 if the tag is present and 0 otherwise
        # we want to have a list of audio paths and a list of one-hot encoded labels
        audio_paths = []
        labels = []
        if self.is_top50:
            top_50 = [
                "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock", "fast", "piano", 
                "ambient", "beat", "violin", "vocal", "synth", "female", "indian", "opera", "male", "singing", "vocals", 
                "no vocals", "harpsichord", "loud", "quiet", "flute", "woman", "male vocal", "no vocal", "pop", "soft", 
                "sitar", "solo", "man", "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal", 
                "beats", "harp", "cello", "no voice", "weird", "country", "metal", "female voice", "choral"
            ]

            # filter only the songs that have at least one of the top 50 tags
            df = df[df[top_50].sum(axis=1) > 0]
            # remove the columns that are not in top_50 + mp3_path and clip_id
            df = df[top_50 + ["mp3_path", "clip_id"]]
            print ("Number of songs with at least one of the top 50 tags: ", len(df))
            
        else:
            synonyms = [
                ['beat', 'beats'],
                ['chant', 'chanting'],
                ['choir', 'choral'],
                ['classical', 'clasical', 'classic'],
                ['drum', 'drums'],
                ['electro', 'electronic', 'electronica', 'electric'],
                ['fast', 'fast beat', 'quick'],
                ['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing', 'women'],
                ['flute', 'flutes'],
                ['guitar', 'guitars'],
                ['hard', 'hard rock'],
                ['harpsichord', 'harpsicord'],
                ['heavy', 'heavy metal', 'metal'],
                ['horn', 'horns'],
                ['india', 'indian'],
                ['jazz', 'jazzy'],
                ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
                ['no beat', 'no drums'],
                ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
                ['opera', 'operatic'],
                ['orchestra', 'orchestral'],
                ['quiet', 'silence'],
                ['singer', 'singing'],
                ['space', 'spacey'],
                ['string', 'strings'],
                ['synth', 'synthesizer'],
                ['violin', 'violins'],
                ['vocal', 'vocals', 'voice', 'voices'],
                ['strange', 'weird']
            ]

            # merge synonyms into one tag - put 1 
            for syn in synonyms:
                df[syn[0]] = df[syn].max(axis=1)
                df = df.drop(syn[1:], axis=1)

            print ("Number of tags merged: ", len(synonyms))

        # test_gt_mtt.tsv - train_gt_mtt.tsv - val_gt_mtt.tsv contains as first column the clip_id - use it to perform the split
        
        train_ids_df = pd.read_csv(self.path + "train_gt_mtt.tsv", sep="\t", header=None)
        train_ids = train_ids_df[0].tolist()
        train_df = df[df["clip_id"].isin(train_ids)]

        validation_ids_df = pd.read_csv(self.path + "val_gt_mtt.tsv", sep="\t", header=None)
        validation_ids = validation_ids_df[0].tolist()
        validation_df = df[df["clip_id"].isin(validation_ids)]

        test_ids_df = pd.read_csv(self.path + "test_gt_mtt.tsv", sep="\t", header=None)
        test_ids = test_ids_df[0].tolist()
        test_df = df[df["clip_id"].isin(test_ids)] 

        # label_names are the names of the tags - all columns except the mp3_path and clip_id columns
        label_names = df.columns
        label_names = label_names.drop(["mp3_path", "clip_id"])

        train_paths = []
        train_labels = []
        validation_paths = []
        validation_labels = []
        test_paths = []
        test_labels = []

        count_empty_train = 0
        count_empty_validation = 0
        count_empty_test = 0

        # iterate over all rows in train_df
        for index, row in train_df.iterrows():
            # get the mp3_path
            audio_path = self.path + row["mp3_path"]
            # get the one-hot encoded tags
            label = row[label_names].tolist()
            # if the song has no tags, skip it
            if sum(label) == 0:
                count_empty_train += 1
                continue
            train_paths.append(audio_path)
            train_labels.append(label)

        # iterate over all rows in validation_df
        for index, row in validation_df.iterrows():
            # get the mp3_path
            audio_path = self.path + row["mp3_path"]
            # get the one-hot encoded tags
            label = row[label_names].tolist()
            # if the song has no tags, skip it
            if sum(label) == 0:
                count_empty_validation += 1
                continue
            validation_paths.append(audio_path)
            validation_labels.append(label)

        # iterate over all rows in test_df
        for index, row in test_df.iterrows():
            # get the mp3_path
            audio_path = self.path + row["mp3_path"]
            # get the one-hot encoded tags
            label = row[label_names].tolist()
            # if the song has no tags, skip it
            if sum(label) == 0:
                count_empty_test += 1
                continue
            test_paths.append(audio_path)
            test_labels.append(label)

        if self.verbose:
            if count_empty_train > 0:
                print(f"Warning: {count_empty_train}/{len(train_df)} = {count_empty_train/len(train_df)*100:.2f}% of the audio files have no tags")
            if count_empty_validation > 0:
                print(f"Warning: {count_empty_validation}/{len(validation_df)} = {count_empty_validation/len(validation_df)*100:.2f}% of the audio files have no tags")
            if count_empty_test > 0:
                print(f"Warning: {count_empty_test}/{len(test_df)} = {count_empty_test/len(test_df)*100:.2f}% of the audio files have no tags")
            # print some statistics
            print (f"Train set: {len(train_paths)}")
            print (f"Validation set: {len(validation_paths)}")
            print (f"Test set: {len(test_paths)}")


        # limit dataset size for testing
        '''
        train_paths = train_paths[:100]
        train_labels = train_labels[:100]
        validation_paths = validation_paths[:100]
        validation_labels = validation_labels[:100]
        test_paths = test_paths[:100]
        test_labels = test_labels[:100]
        '''

        # convert the labels to tensors
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        self.num_classes = len(label_names)

        self.id_to_label = {i: label for i, label in enumerate(label_names)}

        if self.verbose:
            # print some statistics - total number of audio files, number of classes
            print (f"Total number of audio files: {len(train_paths) + len(validation_paths) + len(test_paths)}")
            print (f"Number of classes: {self.num_classes}")

        return train_paths, train_labels, validation_paths, validation_labels, test_paths, test_labels



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
            precompute_embeddings = True,
        )

        val_dataset = ClassificationDataset(
            audio_paths = self.validation_paths,
            labels = self.validation_labels,
            model = model,
            sampling_rate = model.get_sampling_rate(),
            precompute_embeddings = True,
        )

        test_dataset = ClassificationDataset(
            audio_paths = self.test_paths,
            labels = self.test_labels,
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

        return metrics