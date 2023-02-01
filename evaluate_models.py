from transformers import AutoModel, AutoFeatureExtractor
import torch
import numpy as np
import soundfile as sf

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset
from arch_eval import RAVDESS
from arch_eval import US8K
from arch_eval import AudioMNIST

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/wav2vec2-base')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--tsv_logging_file', type=str, default='results/hf_models.tsv')
parser.add_argument('--n_iters', type=int, default=1)
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

        # TODO: manage long audio files in streaming mode (split in chunks)
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
        return token_embeddings.squeeze()

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
*                                        AudioMNIST                                            *
************************************************************************************************
'''

model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=5*16_000)
AUDIO_MNIST_DATASET_PATH = "/data1/mlaquatra/datasets/audio_datasets/AudioMNIST/"
evaluator_audio_mnist = AudioMNIST(path=AUDIO_MNIST_DATASET_PATH, verbose=args.verbose)

res = []
for i in range(args.n_iters):
    if args.verbose:
        print(f"Iteration {i+1}/{args.n_iters}")
        print ("----------------- AudioMNIST LINEAR -----------------")

    res_audio_mnist = evaluator_audio_mnist.evaluate(model, mode="linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)

    if args.verbose:
        print("Iteration: ", i+1)
        for metric, value in res_audio_mnist.items():
            print (f"{metric}: {value}")

    res.append(res_audio_mnist)

# compute mean and std of each metric over all iterations
res_mean = {}
res_std = {}
for metric in res[0].keys():
    res_mean[metric] = np.mean([r[metric] for r in res])
    res_std[metric] = np.std([r[metric] for r in res])

if args.verbose:
    print ("----------------- AudioMNIST LINEAR -----------------")
    for metric, value in res_mean.items():
        print (f"{metric}: {value} +- {res_std[metric]}")


# create a tsv line: model_tag, size, is_linear, dataset_name, mean_accuracy, std_accuracy, mean_f1, std_f1
tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\taudio_mnist\t{res_mean['accuracy']}\t{res_std['accuracy']}\t{res_mean['f1']}\t{res_std['f1']}")

res = []
for i in range(args.n_iters):
    if args.verbose:
        print(f"Iteration {i+1}/{args.n_iters}")
        print ("----------------- AudioMNIST NON-LINEAR -----------------")

    res_audio_mnist = evaluator_audio_mnist.evaluate(model, mode="non-linear", device=args.device, batch_size=32, max_num_epochs=args.max_epochs)

    if args.verbose:
        print("Iteration: ", i+1)
        for metric, value in res_audio_mnist.items():
            print (f"{metric}: {value}")

    res.append(res_audio_mnist)

# compute mean and std of each metric over all iterations
res_mean = {}
res_std = {}
for metric in res[0].keys():
    res_mean[metric] = np.mean([r[metric] for r in res])
    res_std[metric] = np.std([r[metric] for r in res])

if args.verbose:
    print ("----------------- AudioMNIST NON-LINEAR -----------------")
    for metric, value in res_mean.items():
        print (f"{metric}: {value} +- {res_std[metric]}")

# create a tsv line: model_tag, size, is_linear, dataset_name, mean_accuracy, std_accuracy, mean_f1, std_f1
tsv_lines.append(f"{args.model}\t{model_parameters}\tFalse\taudio_mnist\t{res_mean['accuracy']}\t{res_std['accuracy']}\t{res_mean['f1']}\t{res_std['f1']}")



