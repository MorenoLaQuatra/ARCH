from typing import Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from arch_eval import Model, ClassificationModel

# implement a child class of Model
class Wav2Vec2ModelWrapper(Model):
    def __init__(self, model, feature_extractor, device, max_length, attentive_pooling=False, train_backbone=False):
        super().__init__(model)
        self.model = model
        # the model must not be trained
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_length = max_length
        self.attentive_pooling = attentive_pooling
        if attentive_pooling:
            self.attentive_pooling_head = AttentivePoolingHead(self.model.config.hidden_size)
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

        if self.attentive_pooling:
            return self.attentive_pooling_head(token_embeddings).squeeze()
        else:
            return token_embeddings.mean(dim=1).squeeze()


    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
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




class AttentivePoolingHead(nn.Module):
    '''
    This class implements the attentive pooling head.
    The attentive pooling head contains:
        - attentive pooling layer
        - fully connected layer with relu activation and batch normalization

    This layer is used to obtain a fixed size representation of the input sequence.
    '''

    def __init__(
            self,
            hidden_size: int = 768,
            ):
        '''
        Args:
            config: Model configuration.
        '''

        super().__init__()
        self.hidden_size = hidden_size
        # attentive pooling layer
        self.attentive_pooling = nn.Linear(self.hidden_size, 1)
        # fully connected layer with relu activation and batch normalization
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
        )


    def forward(
            self,
            x: torch.Tensor,
            ):
        '''
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        '''
        attentive_pooling = self.attentive_pooling(x)
        attentive_pooling = attentive_pooling.squeeze(-1)
        attentive_pooling = torch.softmax(attentive_pooling, dim=-1)
        attentive_pooling = attentive_pooling.unsqueeze(1)
        x = torch.bmm(attentive_pooling, x)
        x = x.squeeze(1)
        x = self.fc(x)

        return x