import torch
import numpy as np
import soundfile as sf
import json
from transformers import AutoModel, AutoFeatureExtractor

from arch_eval import Model, ClassificationModel
from arch_eval import ClassificationDataset

from arch_eval import ESC50
from arch_eval import US8K
from arch_eval import FSD50K
from arch_eval import VIVAE

from arch_eval import FMASmall
from arch_eval import MagnaTagATune
from arch_eval import IRMAS
from arch_eval import MedleyDB

from arch_eval import RAVDESS
from arch_eval import AudioMNIST
from arch_eval import SLURP
from arch_eval import EMOVO

from configs.w2v2_wrapper import Wav2Vec2ModelWrapper

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='facebook/wav2vec2-base')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--tsv_logging_file', type=str, default='results/hf_models.tsv')
parser.add_argument('--n_iters', type=int, default=1)
parser.add_argument('--data_config_file', type=str, default='configs/datasets_config.json')
parser.add_argument('--attentive_pooling', default=False, action = 'store_true')
parser.add_argument('--precompute_embeddings', default=False, action = 'store_true')
parser.add_argument('--enabled_datasets', type=str, nargs='+', default=["esc50", "us8k", "fsd50k", "vivae", 
                                                                        "fma_small", "magna_tag_a_tune", "irmas", "medleydb",
                                                                        "ravdess", "audio_mnist", "slurp", "emovo"])
args = parser.parse_args()

# example command:
# python evaluate_models.py --model facebook/wav2vec2-base --device cuda --max_epochs 200 --verbose --tsv_logging_file results/hf_models.tsv --n_iters 1 --data_config_file configs/datasets_config.json --enabled_datasets esc50 us8k fma_small magnatagatune irmas ravdess audio_mnist

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


# load datasets info
with open(args.data_config_file) as f:
    datasets_info = json.load(f)

enabled_datasets = args.enabled_datasets

for dataset_name in enabled_datasets:
    
    model = Wav2Vec2ModelWrapper(audio_model, feature_extractor, args.device, max_length=datasets_info[dataset_name]["max_length_seconds"]*16_000)
    
    if dataset_name == "esc50":
        evaluator = ESC50(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "us8k":
        evaluator = US8K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "fma_small":
        evaluator = FMASmall(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "magna_tag_a_tune":
        evaluator = MagnaTagATune(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "irmas":
        evaluator = IRMAS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "ravdess":
        evaluator = RAVDESS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "audio_mnist":
        evaluator = AudioMNIST(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "fsd50k":
        evaluator = FSD50K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "slurp":
        evaluator = SLURP(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "vivae":
        evaluator = VIVAE(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "medleydb":
        evaluator = MedleyDB(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "emovo":
        evaluator = EMOVO(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


    mode = "attention-pooling" if args.attentive_pooling else "linear"

    res = []
    for i in range(args.n_iters):
        if args.verbose:
            print(f"Iteration {i+1}/{args.n_iters}")
            print (f"----------------- {dataset_name} {mode} -----------------")

        res_dataset = evaluator.evaluate(
            model, 
            mode=mode, 
            device=args.device, 
            batch_size=32, 
            max_num_epochs=args.max_epochs, 
        )

        if args.verbose:
            print(f"Iteration {i+1}/{args.n_iters}")
            for metric, value in res_dataset.items():
                print(f"{metric}: {value}")

        res.append(res_dataset)

    res_mean = {}
    res_std = {}
    for metric in res[0].keys():
        res_mean[metric] = np.mean([r[metric] for r in res])
        res_std[metric] = np.std([r[metric] for r in res])

    if args.verbose:
        print(f"----------------- {dataset_name} {mode} -----------------")
        for metric, value in res_mean.items():
            print(f"{metric}: {value} +/- {res_std[metric]}")

    # create a tsv line: model_tag, size, is_linear, dataset_name, mean_map_macro, std_map_macro, mean_map_weighted, std_map_weighted
    if datasets_info[dataset_name]["is_multilabel"]:
        tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['map_macro']}\t{res_std['map_macro']}\t{res_mean['map_weighted']}\t{res_std['map_weighted']}\n")
    else:
        tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['accuracy']}\t{res_std['accuracy']}\t{res_mean['f1']}\t{res_std['f1']}\n")


if args.verbose:
    print("\n\nAll results:")
    for line in tsv_lines:
        print(line)

# append tsv lines in file
with open(args.tsv_logging_file, "a") as f:
    for line in tsv_lines:
        f.write(line)