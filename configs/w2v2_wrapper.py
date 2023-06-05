from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from arch_eval import Model, ClassificationModel

# implement a child class of Model
class Wav2Vec2ModelWrapper(Model):
    def __init__(self, model, feature_extractor, device, max_length, train_backbone=False):
        super().__init__(model)
        self.model = model
        # the model must not be trained
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_length = max_length
        self.train_backbone = train_backbone

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).input_values
        inputs = inputs.to(self.device)
        if self.train_backbone:
            token_embeddings = self.model(inputs).last_hidden_state
        else:
            with torch.no_grad():
                token_embeddings = self.model(inputs).last_hidden_state

        embeddings = token_embeddings.mean(dim=1).squeeze()
        # move the embeddings to the cpu
        embeddings = embeddings.cpu()
        return embeddings

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16_000, 
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).input_values
        
        inputs = inputs.to(self.device)
        if self.train_backbone:
            token_embeddings = self.model(inputs).last_hidden_state
        else:
            with torch.no_grad():
                token_embeddings = self.model(inputs).last_hidden_state

        # move the embeddings to the cpu
        if self.train_backbone:
            token_embeddings = token_embeddings.cpu()
        else:
            with torch.no_grad():
                token_embeddings = token_embeddings.detach().cpu()

        return token_embeddings.squeeze()


    def get_token_embeddings_old(self, audio: np.ndarray, **kwargs):
        chunks = []
        for i in range(0, len(audio), self.max_length):
            if i + self.max_length >= len(audio):
                chunk = audio[i:]
            else:
                # add overlap for approx problem - 20ms = 1 frame = 320 samples
                overlap = int(0.02*16_000)
                chunk = audio[i: i + self.max_length + overlap]
            inputs = self.feature_extractor(
                chunk, 
                sampling_rate=16_000, 
                return_tensors="pt",
            ).input_values
            inputs = inputs.to(self.device)
            if self.train_backbone:
                token_embeddings = self.model(inputs).last_hidden_state
            else:
                with torch.no_grad():
                    token_embeddings = self.model(inputs).last_hidden_state

            chunks.append(token_embeddings.squeeze().cpu())

        return torch.cat(chunks, dim=0)

    def get_classification_embedding_size(self):
        return self.model.config.hidden_size

    def get_token_embedding_size(self):
        return self.model.config.hidden_size

    def get_sampling_rate(self):
        return self.feature_extractor.sampling_rate

    def get_embedding_layer(self):
        # return the size of the embedding layer
        return self.model.config.hidden_size