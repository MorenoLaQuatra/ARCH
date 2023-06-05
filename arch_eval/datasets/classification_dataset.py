import torch
import torchaudio
import numpy as np
from arch_eval.models.model import Model
from typing import List, Union
from tqdm import tqdm

class ClassificationDataset(torch.utils.data.Dataset):
    """
    This class implements a PyTorch dataset for classification tasks.
    It also takes as input the model that will be used to generate the embeddings.
    """

    def __init__(
        self,
        audio_paths: List[str] = None,
        audios: List[Union[np.ndarray, torch.Tensor]] = None,
        labels: Union[List[int], List[List[int]]] = None,
        model: Model = None,
        sampling_rate: int = 16000,
        precompute_embeddings: bool = False,
        verbose: bool = False,
        mode: str = "linear",
        **kwargs,
    ):
        """
        :param audio_paths: list of audio paths
        :param audios: list of audio tensors
        :param labels: list of labels
        :param model: model that will be used to generate the embeddings
        :param sampling_rate: sampling rate of the audio
        :param precompute_embeddings: if True, the embeddings will be precomputed to avoid recomputing them for each epoch
        :param verbose: if True, print progress
        :param kwargs: additional parameters
        """
        if audio_paths is None and audios is None:
            raise ValueError("Either audio_paths or audios must be provided.")
        if audio_paths is not None and audios is not None:
            raise ValueError("Only one of audio_paths or audios must be provided.")
        
        # check if the other parameters are provided
        if labels is None:
            raise ValueError("labels must be provided.")
        if model is None:
            raise ValueError("model must be provided.")
        
        self.audio_paths = audio_paths
        self.audios = audios
        self.labels = labels
        self.model = model
        self.sampling_rate = sampling_rate
        self.precompute_embeddings = precompute_embeddings
        self.verbose = verbose
        self.mode = mode
        if self.precompute_embeddings:
            print("Precomputing embeddings...")
            self._precompute_embeddings()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        if self.audio_paths is not None:
            return len(self.audio_paths)
        else:
            return len(self.audios)

    def _get_embeddings_from_path(self, audio_path):
        '''
        Get the embeddings from a path
        '''
        # Load audio and resample it if necessary
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
        # if audio is 1, length - remove first dimension
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0)
        if audio.shape[0] == 1:
            audio = audio[0]

        # Generate embeddings
        if self.mode == "attention-pooling":
            embeddings = self.model.get_sequence_embeddings(audio)
        else:
            embeddings = self.model.get_embeddings(audio)
        return embeddings

    def _get_embeddings_from_audio(self, audio):
        '''
        Get the embeddings from an audio
        '''
        # if audio is 1, length - remove first dimension
        if audio.shape[0] == 1:
            audio = audio[0]
        # Generate embeddings
        if self.mode == "attention-pooling":
            embeddings = self.model.get_sequence_embeddings(audio)
        else:
            embeddings = self.model.get_embeddings(audio)
        return embeddings

    def _get_embeddings_shape(self):
        '''
        Get the shape of the embeddings
        '''
        if self.audio_paths is not None:
            audio_path = self.audio_paths[0]
            embeddings = self._get_embeddings_from_path(audio_path)
            shape = list(embeddings.shape)
        else:
            audio = self.audios[0]
            embeddings = self._get_embeddings_from_audio(audio)
            shape = list(embeddings.shape)
        return shape

    def _precompute_embeddings(self):
        '''
        Precompute embeddings for all the audio files in the dataset.
        This is done to avoid recomputing the embeddings for each epoch.
        '''
        indexes_to_remove = []

        # get the shape of the embeddings
        shape = self._get_embeddings_shape()

        # create an empty tensor to store the embeddings - independent of the input shape
        print(f"Shape of the embeddings: {shape}")
        print(f"Allocating memory for {len(self)} embeddings...")
        print(f"Total size: {len(self) * np.prod(shape) * 4 / 1024 / 1024 / 1024} GB")
        self.embeddings = torch.zeros((len(self), *shape))

        # compute the embeddings for all the audio files
        if self.audio_paths is not None:
            for audio_path in tqdm(self.audio_paths):
                try:
                    embeddings = self._get_embeddings_from_path(audio_path)
                except RuntimeError:
                    print(f"Error loading {audio_path}")
                    indexes_to_remove.append(self.audio_paths.index(audio_path))
                    continue
                self.embeddings[self.audio_paths.index(audio_path)] = embeddings
        else:
            index_embeddings = 0
            for audio in tqdm(self.audios):
                embeddings = self._get_embeddings_from_audio(audio)
                self.embeddings[index_embeddings] = embeddings
                index_embeddings += 1

        # remove audio paths and labels that could not be loaded
        if len(indexes_to_remove) > 0:
            for index in sorted(indexes_to_remove, reverse=True):
                del self.audio_paths[index]
                try:
                    del self.labels[index]
                    self.embeddings = torch.cat((self.embeddings[:index], self.embeddings[index+1:]))
                except TypeError: # if the labels are tensors
                    self.labels = torch.cat((self.labels[:index], self.labels[index+1:]))

        print(f"Successfully loaded {len(self)} audio files.")
        print(f"Shape of the final embeddings: {self.embeddings.shape}")


    def __getitem__(self, idx):

        if self.precompute_embeddings:
            embeddings = self.embeddings[idx]
            label = self.labels[idx]
            return embeddings, label
        
        if self.audio_paths is not None:

            audio_path = self.audio_paths[idx]
            label = self.labels[idx]

            # Load audio and resample it if necessary
            audio, sr = torchaudio.load(audio_path)

            if sr != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
        else:
            audio = self.audios[idx]
            label = self.labels[idx]

        # if audio is 1, length - remove first dimension
        if audio.shape[0] == 1:
            audio = audio[0]

        # Generate embeddings
        if self.mode == "attention-pooling":
            embeddings = self.model.get_sequence_embeddings(audio)
        else:
            embeddings = self.model.get_embeddings(audio)

        return embeddings, label
