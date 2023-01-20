import torch
import torchaudio
import numpy as np
from arch_eval.models.model import Model
from typing import List, Union
from tqdm import tqdm

class SequenceClassificationDataset(torch.utils.data.Dataset):
    """
    This class implements a PyTorch dataset for the sequence classification task.
    It also takes as input the model that will be used to generate the embeddings for each frame.
    """

    def __init__(
        self,
        audio_paths: List[str] = None,
        audios: List[Union[np.ndarray, torch.Tensor]] = None,
        labels: List[int] = None,
        model: Model = None,
        sampling_rate: int = 16000,
        precompute_embeddings: bool = False,
        verbose: bool = False,
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
        if self.precompute_embeddings:
            self._precompute_embeddings()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        if self.audio_paths is not None:
            return len(self.audio_paths)
        else:
            return len(self.audios)

    def _precompute_embeddings(self):
        '''
        Precompute embeddings for all the audio files in the dataset.
        This is done to avoid recomputing the embeddings for each epoch.
        '''
        self.embeddings = []
        indexes_to_remove = []
        if self.audio_paths is not None:
            for audio_path in tqdm(self.audio_paths):
                # Load audio and resample it if necessary
                try:
                    audio, sr = torchaudio.load(audio_path)
                except RuntimeError:
                    print(f"Error loading {audio_path}")
                    indexes_to_remove.append(self.audio_paths.index(audio_path))
                    continue
                # if stereo, convert to mono
                if audio.shape[0] == 2:
                    audio = torch.mean(audio, dim=0, keepdim=True)

                if sr != self.sampling_rate:
                    audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
                # if audio is 1, length - remove first dimension
                if audio.shape[0] == 1:
                    audio = audio[0]
                # Generate embeddings
                embeddings = self.model.get_token_embeddings(audio)
                # remove required_grad
                embeddings = embeddings.detach()
                self.embeddings.append(embeddings)
        else:
            for audio in tqdm(self.audios):
                # if audio is 1, length - remove first dimension
                if audio.shape[0] == 1:
                    audio = audio[0]
                # Generate embeddings
                embeddings = self.model.get_token_embeddings(audio)
                # remove required_grad
                embeddings = embeddings.detach()
                self.embeddings.append(embeddings)

        # remove audio paths and labels that could not be loaded
        if len(indexes_to_remove) > 0:
            for index in sorted(indexes_to_remove, reverse=True):
                del self.audio_paths[index]
                del self.labels[index]

        self.embeddings = torch.stack(self.embeddings)
        print (f"Embeddings shape: {self.embeddings.shape}")
        return


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
        embeddings = self.model.get_token_embeddings(audio)
        # remove required_grad
        embeddings = embeddings.detach()

        return embeddings, label
