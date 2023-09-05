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
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--data_config_file', type=str, default='configs/datasets_config.json')
parser.add_argument('--enabled_datasets', type=str, nargs='+', default=["esc50", "us8k", "fsd50k", "vivae", 
                                                                        "fma_small", "magna_tag_a_tune", "irmas", "medleydb",
                                                                        "ravdess", "audio_mnist", "slurp", "emovo"])
args = parser.parse_args()

print("------------------------------------")
print(f"Computing datasets average duration")
print("------------------------------------")

'''
************************************************************************************************
*                                       Setting parameters                                     *
************************************************************************************************
'''

# load datasets info
with open(args.data_config_file) as f:
    datasets_info = json.load(f)

enabled_datasets = args.enabled_datasets

for dataset_name in enabled_datasets:
    
    if dataset_name == "esc50":
        evaluator = ESC50(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "us8k":
        evaluator = US8K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "fma_small":
        evaluator = FMASmall(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "magna_tag_a_tune":
        evaluator = MagnaTagATune(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "irmas":
        evaluator = IRMAS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "ravdess":
        evaluator = RAVDESS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "audio_mnist":
        evaluator = AudioMNIST(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "fsd50k":
        evaluator = FSD50K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "slurp":
        evaluator = SLURP(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "vivae":
        evaluator = VIVAE(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "medleydb":
        evaluator = MedleyDB(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    elif dataset_name == "emovo":
        evaluator = EMOVO(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


    print(f"Computing {dataset_name} average duration")
    avg_duration_in_seconds = evaluator.get_average_duration()
    print(f"{dataset_name} average duration: {avg_duration_in_seconds} seconds")