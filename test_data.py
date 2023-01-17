from arch_eval import ESC50
from arch_eval import FMASmall
from arch_eval import RAVDESS

# Load the evaluator

#evaluator = ESC50(path="/data1/mlaquatra/datasets/audio_datasets/esc50/")
'''
evaluator = FMASmall(
    config_path="/data1/mlaquatra/datasets/audio_datasets/fma_metadata/",
    audio_files_path="/data1/mlaquatra/datasets/audio_datasets/fma_small/"
)
'''
evaluator = RAVDESS(
    path="/data1/mlaquatra/datasets/audio_datasets/ravdess/"
)
