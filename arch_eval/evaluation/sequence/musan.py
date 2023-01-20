import os
import glob
import pandas as pd
import numpy as np
import torch

from arch_eval import Model, SequenceClassificationModel
from arch_eval import SequenceClassificationDataset

from sklearn.model_selection import train_test_split

class MUSAN():
    '''
    This class implements the functionality to load the MUSAN dataset 
    and the recipe for its evaluation (sequence classification).
    It implements the fold-based evaluation, where each fold is a
    different speaker.
    '''