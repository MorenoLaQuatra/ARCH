import torch
import torchaudio
import numpy as np
from arch_eval.models.model import Model
from typing import List, Union

class ClassificationDataset(torch.utils.data.Dataset):
    """
    This class implements a PyTorch dataset for classification tasks.
    It also takes as input the model that will be used to generate the embeddings.
    """

    def __init__(
        self,
        audio_paths: List[str] = None,
        audios: List[Union[np.ndarray, torch.Tensor]] = None,
        labels: List[int] = None,
        model: Model = None,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        """
        :param audio_paths: list of audio paths
        :param labels: list of labels
        :param model: model used to generate embeddings
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
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        if self.audio_paths is not None:
            return len(self.audio_paths)
        else:
            return len(self.audios)

    def __getitem__(self, idx):
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

        # Generate embeddings
        embeddings = self.model.get_embeddings(audio)

        return embeddings, label
