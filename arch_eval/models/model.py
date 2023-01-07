import torch
import numpy as np


class Model:
    """
    Model class: this is a base class for all models. It contains the basic methods
    for generating embeddings from audio files.
    """
    def __init__(
        self,
        model,
        **kwargs,
    ):
        self.model = model
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        
    def get_embeddings(
        self,
        audio: np.ndarray,
        **kwargs,
    ):
        """
        Get embeddings from audio. This method should be implemented by the child class.
        It generates embeddings for the whole audio.
        :param audio: audio array
        :return: a tensor of shape (embedding_size,)
        """
        raise NotImplementedError

    def get_token_embeddings(
        self,
        audio: np.ndarray,
        frame_length_ms: int = 20,
        **kwargs,
    ):
        """
        Get token embeddings from audio. This method should be implemented by the child class.
        It generates embeddings for each frame of the audio. The frame length is specified
        by the frame_length_ms parameter.
        :param audio: audio array
        :param frame_length_ms: frame length in milliseconds
        :return: a tensor of shape (n_frames, embedding_size)
        """
        raise NotImplementedError