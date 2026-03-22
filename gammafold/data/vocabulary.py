"""
Protein vocabulary for GammaFold tokenizer.

Defines all token types:
- 20 standard amino acids
- Extended/ambiguous amino acids  
- Special tokens for model operations
- Secondary structure tokens

OPTIMIZED: Class-level constants (no mutable defaults)
"""

from typing import Dict


class ProteinVocabulary:
    """Complete protein vocabulary with all token types."""
    
    # Standard 20 amino acids (canonical)
    AMINO_ACIDS: Dict[str, int] = {
        'A': 1,   # Alanine
        'C': 2,   # Cysteine
        'D': 3,   # Aspartic acid
        'E': 4,   # Glutamic acid
        'F': 5,   # Phenylalanine
        'G': 6,   # Glycine
        'H': 7,   # Histidine
        'I': 8,   # Isoleucine
        'K': 9,   # Lysine
        'L': 10,  # Leucine
        'M': 11,  # Methionine
        'N': 12,  # Asparagine
        'P': 13,  # Proline
        'Q': 14,  # Glutamine
        'R': 15,  # Arginine
        'S': 16,  # Serine
        'T': 17,  # Threonine
        'V': 18,  # Valine
        'W': 19,  # Tryptophan
        'Y': 20,  # Tyrosine
    }
    
    # Extended amino acids (rare/ambiguous)
    EXTENDED_AA: Dict[str, int] = {
        'U': 21,  # Selenocysteine (rare)
        'O': 22,  # Pyrrolysine (rare)
        'B': 23,  # Asn or Asp (ambiguous)
        'Z': 24,  # Gln or Glu (ambiguous)
        'J': 25,  # Leu or Ile (ambiguous)
        'X': 26,  # Unknown/any amino acid
    }
    
    # Special tokens for model operations
    SPECIAL: Dict[str, int] = {
        '<pad>': 0,     # Padding token
        '<sos>': 27,    # Start of sequence
        '<eos>': 28,    # End of sequence
        '<mask>': 29,   # Masked token (for MLM training)
        '<cls>': 30,    # Classification token
        '<sep>': 31,    # Separator (for multi-sequence input)
        '<unk>': 32,    # Unknown token
        '<gap>': 33,    # Gap in alignment (MSA)
        '<insert>': 34, # Insertion position
        '<null>': 35,   # Null/empty position
    }
    
    # Secondary structure tokens (DSSP 8-class + special)
    SECONDARY_STRUCTURE: Dict[str, int] = {
        '<ss_pad>': 0,  # Padding
        'H': 1,         # Alpha helix
        'G': 2,         # 3-10 helix
        'I': 3,         # Pi helix  
        'E': 4,         # Extended strand (beta sheet)
        'B': 5,         # Isolated beta bridge
        'T': 6,         # Turn
        'S': 7,         # Bend
        'C': 8,         # Coil (loop/irregular)
        '<ss_unk>': 9,  # Unknown secondary structure
    }
    
    # Vocabulary sizes
    VOCAB_SIZE: int = 36
    SS_VOCAB_SIZE: int = 10
    
    # Token range constants (for masking)
    AMINO_ACID_MIN: int = 1
    AMINO_ACID_MAX: int = 20
    STANDARD_AA_COUNT: int = 20
    
    def get_all_tokens(self) -> Dict[str, int]:
        """Return combined vocabulary."""
        return {**self.AMINO_ACIDS, **self.EXTENDED_AA, **self.SPECIAL}
    
    def get_inverse_vocab(self) -> Dict[int, str]:
        """Return ID to token mapping."""
        all_tokens = self.get_all_tokens()
        return {v: k for k, v in all_tokens.items()}
    
    def is_amino_acid(self, token: str) -> bool:
        """Check if token is a standard amino acid."""
        return token in self.AMINO_ACIDS
    
    def is_extended_aa(self, token: str) -> bool:
        """Check if token is an extended/ambiguous amino acid."""
        return token in self.EXTENDED_AA
    
    def is_special(self, token: str) -> bool:
        """Check if token is a special token."""
        return token in self.SPECIAL
    
    def is_maskable(self, token_id: int) -> bool:
        """Check if a token ID can be masked for MLM."""
        return self.AMINO_ACID_MIN <= token_id <= self.AMINO_ACID_MAX


# Singleton instance
VOCABULARY = ProteinVocabulary()
