from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch
import numpy as np
import soundfile as sf

from arch_eval import Model, ClassificationModel, SequenceClassificationModel
from arch_eval import ClassificationDataset
from arch_eval import SequenceClassificationDataset

from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = model.to(DEVICE)

# implement a child class of Model
class Wav2Vec2ModelWrapper(Model):
    def __init__(self, model, feature_extractor, device):
        super().__init__(model)
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        inputs = self.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_values
        inputs = inputs.to(self.device)
        token_embeddings = self.model(inputs).last_hidden_state
        return token_embeddings.mean(dim=1).squeeze()


    def get_token_embeddings(self, audio: np.ndarray, frame_length_ms: int = 20, **kwargs):
        inputs = self.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt").input_values
        inputs = inputs.to(self.device)
        token_embeddings = self.model(inputs).last_hidden_state
        return token_embeddings.squeeze()

    def get_classification_embedding_size(self):
        return self.model.config.hidden_size

    def get_token_embedding_size(self):
        return self.model.config.hidden_size

    def get_sampling_rate(self):
        return self.feature_extractor.sampling_rate

# create a model instance
model = Wav2Vec2ModelWrapper(model, feature_extractor, DEVICE)

# create 8 random audio files
audios = torch.randn(8, 16_000 * 10)

# get embeddings for each audio
for audio in audios:
    embeddings = model.get_embeddings(audio)
    print ("Embeddings shape:", embeddings.shape)
    print ("Expected shape:", (768,))

# get token embeddings for each audio
for audio in audios:
    embeddings = model.get_token_embeddings(audio)
    print ("Embeddings shape:", embeddings.shape)
    print ("Expected shape:", (int(10_000 / 20.00001), 768))

'''
# create a dataset
dataset = ClassificationDataset(
    audios = audios,
    labels = torch.randint(0, 10, (8,)),
    model = model,
)

# validation dataset
dataset_val = ClassificationDataset(
    audios = torch.randn(8, 16_000 * 10),
    labels = torch.randint(0, 10, (8,)),
    model = model,
)

# create a dataloader
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=4, shuffle=False)

classification_model = ClassificationModel(
    layers = [], # linear layers
    input_embedding_size = 768, # size of the embeddings
    activation = "relu",
    dropout = 0.1,
    num_classes = 10, # number of classes
)

# train the model
classification_model.train(
    train_dataloader = train_dataloader,
    val_dataloader = val_dataloader,
    epochs = 10,
    device = DEVICE,
)
'''

'''
TOKEN EMBEDDINGS

'''

audios = torch.randn(8, 16_000 * 10)
# labels shape = (batch_size, number of tokens, number of classes)
labels = torch.randint(0, 10, (8, int(10 * 1000 / 20.00001),))



print ("Test audios shape:", audios.shape)
print ("Test labels shape:", labels.shape)

# create a dataset
dataset = SequenceClassificationDataset(
    audios = audios,
    labels = labels,
    model = model,
    sampling_rate = 16_000,
    precompute_embeddings = True,
    verbose = True,
)

# validation dataset

dataset_val = SequenceClassificationDataset(
    audios = torch.randn(8, 16_000 * 10),
    labels = torch.randint(0, 10, (8, int(10 * 1000 / 20.00001), )),
    model = model,
    sampling_rate = 16_000,
    precompute_embeddings = True,
    verbose = True,
)

# create a dataloader
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=4, shuffle=False)

sequence_classification_model = SequenceClassificationModel(
    layers = [], # linear layers
    input_embedding_size = model.get_token_embedding_size(), # size of the embeddings
    activation="relu",
    dropout=0.1,
    num_classes=10, # number of classes
)

# train the model
sequence_classification_model.train(
    train_dataloader = train_dataloader,
    val_dataloader = val_dataloader,
)

# evaluate the model
sequence_classification_model.evaluate(val_dataloader)



# Multi-class and multi-label classification
print("\n\n------ Multi-class and multi-label classification -------\n\n")

audios = torch.randn(8, 16_000 * 10)
# labels shape = (batch_size, number of tokens, number of classes)
labels = torch.randint(0, 2, (8, int(10 * 1000 / 20.00001), 10), dtype=torch.float32)

print ("Test audios shape:", audios.shape)
print ("Test labels shape:", labels.shape)

# create a dataset
dataset = SequenceClassificationDataset(
    audios = audios,
    labels = labels,
    model = model,
    sampling_rate = 16_000,
    precompute_embeddings = True,
    verbose = True,
)

# validation dataset

dataset_val = SequenceClassificationDataset(
    audios = torch.randn(8, 16_000 * 10),
    labels = torch.randint(0, 1, (8, int(10 * 1000 / 20.00001), 10), dtype=torch.float32),
    model = model,
    sampling_rate = 16_000,
    precompute_embeddings = True,
    verbose = True,
)

# create a dataloader
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=4, shuffle=False)

sequence_classification_model = SequenceClassificationModel(
    layers = [], # linear layers
    input_embedding_size = model.get_token_embedding_size(), # size of the embeddings
    activation="relu",
    dropout=0.1,
    num_classes=10, # number of classes
    is_multilabel = True,
)

# train the model
sequence_classification_model.train(
    train_dataloader = train_dataloader,
    val_dataloader = val_dataloader,
)

# evaluate the model
sequence_classification_model.evaluate(val_dataloader)


