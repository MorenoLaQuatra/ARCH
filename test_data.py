from arch_eval import ESC50
from arch_eval import FMASmall
from arch_eval import RAVDESS
from arch_eval import US8K
from arch_eval import AudioMNIST


from arch_eval import MiviaRoad
from arch_eval import MIR1K
from arch_eval import Jamendo


# Load the evaluator

#evaluator = ESC50(path="/data1/mlaquatra/datasets/audio_datasets/esc50/")
'''
evaluator = FMASmall(
    config_path="/data1/mlaquatra/datasets/audio_datasets/fma_metadata/",
    audio_files_path="/data1/mlaquatra/datasets/audio_datasets/fma_small/"
)

evaluator = RAVDESS(
    path="/data1/mlaquatra/datasets/audio_datasets/ravdess/"
)


evaluator = MiviaRoad(
    path="/data1/mlaquatra/datasets/audio_datasets/MIVIA_ROAD_DB1/"
)

evaluator = MIR1K(
    path="/data1/mlaquatra/datasets/audio_datasets/MIR-1K/",
)

evaluator = Jamendo(
    path="/data1/mlaquatra/datasets/audio_datasets/jamendo/"
)

evaluator = US8K(
    path="/data1/mlaquatra/datasets/audio_datasets/UrbanSound8K/",
    verbose=True
)
'''

evaluator = AudioMNIST(
    path="/data1/mlaquatra/datasets/audio_datasets/AudioMNIST/",
    verbose=True
)


