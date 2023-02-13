from .models.model import Model

from .models.classification_model import ClassificationModel
from .models.sequence_classification_model import SequenceClassificationModel

from .datasets.classification_dataset import ClassificationDataset
from .datasets.sequence_classification_dataset import SequenceClassificationDataset

from .evaluation.classification.sound.esc50 import ESC50
from .evaluation.classification.sound.us8k import US8K
from .evaluation.classification.sound.fsd50k import FSD50K
from .evaluation.classification.sound.vivae import VIVAE

from .evaluation.classification.music.fma_small import FMASmall
from .evaluation.classification.music.magnatagatune import MagnaTagATune
from .evaluation.classification.music.irmas import IRMAS
from .evaluation.classification.music.medleydb import MedleyDB

from .evaluation.classification.speech.ravdess import RAVDESS
from .evaluation.classification.speech.audio_mnist import AudioMNIST
from .evaluation.classification.speech.slurp import SLURP
from .evaluation.classification.speech.emovo import EMOVO


from .evaluation.sequence.mivia_road import MiviaRoad
from .evaluation.sequence.mir1k import MIR1K
from .evaluation.sequence.jamendo import Jamendo