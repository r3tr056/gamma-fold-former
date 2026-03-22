"""
GammaFold Data Module.

Provides data loading, tokenization, and preprocessing.
"""

from gammafold.data.tokenizer import GammaFoldTokenizer
from gammafold.data.vocabulary import ProteinVocabulary, VOCABULARY
from gammafold.data.properties import PHYSICOCHEMICAL_PROPERTIES
from gammafold.data.collators import DataCollator, DataCollatorForMLM
from gammafold.data.dataset import ProteinSequenceDataset, create_dataloader
from gammafold.data.filtering import FilterConfig, filter_sequence, prepare_dataset
from gammafold.data.splits import SplitConfig, split_fasta, random_split

__all__ = [
    # Tokenization
    "GammaFoldTokenizer",
    "ProteinVocabulary",
    "VOCABULARY",
    "PHYSICOCHEMICAL_PROPERTIES",
    
    # Data loading
    "DataCollator",
    "DataCollatorForMLM",
    "ProteinSequenceDataset",
    "create_dataloader",
    
    # Preprocessing
    "FilterConfig",
    "filter_sequence",
    "prepare_dataset",
    "SplitConfig",
    "split_fasta",
    "random_split",
]

