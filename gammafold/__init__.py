"""
GammaFold: Protein Sequence and Structure Modeling Framework

A multi-modal transformer for protein language modeling and structure prediction.
"""

__version__ = "0.1.0"
__author__ = "GammaFold Team"

# Data
from gammafold.data.tokenizer import GammaFoldTokenizer
from gammafold.data.vocabulary import ProteinVocabulary, VOCABULARY
from gammafold.data.collators import DataCollator, DataCollatorForMLM

# Models
from gammafold.models.gamma_fold import (
    GammaFoldConfig,
    GammaFoldFormer,
    GammaFoldMultiModal,
    create_model,
    create_multimodal_model,
)

# Training
from gammafold.training import (
    TrainingConfig,
    Trainer,
    CombinedLoss,
)

__all__ = [
    # Version
    "__version__",
    
    # Data
    "GammaFoldTokenizer",
    "ProteinVocabulary",
    "VOCABULARY",
    "DataCollator",
    "DataCollatorForMLM",
    
    # Models
    "GammaFoldConfig",
    "GammaFoldFormer",
    "GammaFoldMultiModal",
    "create_model",
    "create_multimodal_model",
    
    # Training
    "TrainingConfig",
    "Trainer",
    "CombinedLoss",
]

