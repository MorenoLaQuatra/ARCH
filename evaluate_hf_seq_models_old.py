from transformers import AutoModel, AutoFeatureExtractor
import torch
import numpy as np
import soundfile as sf

from arch_eval import Model, SequenceClassificationModel
from arch_eval import SequenceClassificationDataset
from arch_eval import MiviaRoad
from arch_eval import MIR1K
from arch_eval import Jamendo


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='microsoft/wavlm-base')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--tsv_logging_file', type=str, default='results/hf_seq_models.tsv')
parser.add_argument('--model_frame_length_ms', type=float, default=20.00001)
args = parser.parse_args()

print("------------------------------------")
print(f"Evaluating model: {args.model}")
print("------------------------------------")

'''
************************************************************************************************
*                                       Setting parameters                                     *
************************************************************************************************
'''

# Load model
audio_model = AutoModel.from_pretrained(args.model)
feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
audio_model = audio_model.to(args.device)
model_parameters = sum(p.numel() for p in audio_model.parameters())
tsv_lines = [] 

'''
************************************************************************************************
*                                         Model Wrapping                                       *
************************************************************************************************
'''

# implement a child class of Model
class Wav2Vec2ModelWrapper(Model):
    def __init__(self, model, feature_extractor, device, max_length):
        super().__init__(model)
        self.model = model
        # the model must not be trained
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.device = device
        self.max_length = max_length

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
        token_embeddings = self.model(inputs).last_hidden_state
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



'''
************************************************************************************************
*                                          MIVIA                                               *
************************************************************************************************
'''

model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=30*16_000)
MIVIA_ROAD_PATH = "/data1/mlaquatra/datasets/audio_datasets/MIVIA_ROAD_DB1/"
mivia_road_eval = MiviaRoad(
    path=MIVIA_ROAD_PATH,
    verbose=args.verbose
)

res_mivia = mivia_road_eval.evaluate(
    model = model,
    mode = "non-linear",
    device = args.device,
    batch_size = 32,
    max_num_epochs = args.max_epochs,
    model_frame_length_ms = args.model_frame_length_ms,
)

for k, v in res_mivia.items():
    print(f"{k}: {v}")


'''
************************************************************************************************
*                                          MIR1K                                               *
************************************************************************************************
'''

model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=30*16_000)
MIR1K_PATH = "/data1/mlaquatra/datasets/audio_datasets/MIR-1K/"
mir1k_eval = MIR1K(
    path=MIR1K_PATH,
    verbose=args.verbose
)

res_mir1k = mir1k_eval.evaluate(
    model = model,
    mode = "non-linear",
    device = args.device,
    batch_size = 32,
    max_num_epochs = args.max_epochs,
    model_frame_length_ms = args.model_frame_length_ms,
)



'''
************************************************************************************************
*                                          Jamendo                                            *
************************************************************************************************
'''

'''
model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=10*16_000)
JAMENDO_PATH = "/data1/mlaquatra/datasets/audio_datasets/jamendo/"
jamendo_eval = Jamendo(
    path=JAMENDO_PATH,
    verbose=args.verbose
)

res_jamendo = jamendo_eval.evaluate(
    model = model,
    mode = "linear",
    device = args.device,
    batch_size = 1,
    max_num_epochs = args.max_epochs,
    model_frame_length_ms = 20.00001, # 20ms ish (wav2vec2 default)
)
'''