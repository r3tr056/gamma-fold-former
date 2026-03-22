"""
Expanded physicochemical properties for amino acids.

Properties sourced from AAindex database and literature:
- Hydrophobicity scales
- Charge and electrostatic properties
- Size/volume properties
- Structural propensities
- Biochemical properties

Each amino acid is represented by a vector of 24 normalized properties.
"""

import numpy as np
from typing import Dict, List, Tuple


# AAindex-derived and literature physicochemical properties
# Each amino acid has 24 properties capturing different aspects
# Values are from various AAindex scales and normalized
AMINO_ACID_PROPERTIES_RAW = {
    # Format: [hydrophobicity_kyte, hydrophobicity_hopp, charge_pH7, pI,
    #          volume, surface_area, mass, polarity_grantham,
    #          helix_propensity, sheet_propensity, turn_propensity, coil_propensity,
    #          flexibility, mutability, bulkiness, refractivity,
    #          H_bond_donor, H_bond_acceptor, polar_surface, nonpolar_surface,
    #          aromatic, aliphatic, sulfur, hydroxyl]
    
    'A': [ 1.8, -0.5,  0.0,  6.0,   88.6, 115.0,  89.1,  8.1,
           1.42, 0.83, 0.66, 1.01,  0.36, 100.0, 11.5, 0.046,
           0.0,  0.0,  0.0, 67.0,  0.0, 1.0, 0.0, 0.0],
    
    'C': [ 2.5, -1.0,  0.0,  5.1,  108.5, 135.0, 121.2,  5.5,
           0.70, 1.19, 1.19, 0.96,  0.35, 20.0, 13.5, 0.128,
           0.0,  0.0,  0.0, 104.0,  0.0, 0.0, 1.0, 0.0],
    
    'D': [-3.5,  3.0, -1.0,  2.8,  111.1, 150.0, 133.1, 13.0,
           1.01, 0.54, 1.46, 1.05,  0.51, 81.0, 11.7, 0.105,
           0.0,  2.0,  50.0, 59.0,  0.0, 0.0, 0.0, 0.0],
    
    'E': [-3.5,  3.0, -1.0,  3.2,  138.4, 190.0, 147.1, 12.3,
           1.51, 0.37, 0.74, 0.99,  0.50, 83.0, 13.6, 0.151,
           0.0,  2.0,  50.0, 67.0,  0.0, 0.0, 0.0, 0.0],
    
    'F': [ 2.8, -2.5,  0.0,  5.5,  189.9, 210.0, 165.2,  5.2,
           1.13, 1.38, 0.60, 0.89,  0.31, 41.0, 19.8, 0.290,
           0.0,  0.0,  0.0, 164.0,  1.0, 0.0, 0.0, 0.0],
    
    'G': [-0.4,  0.0,  0.0,  6.0,   60.1,  75.0,  75.1,  9.0,
           0.57, 0.75, 1.56, 1.14,  0.54, 49.0,  3.4, 0.000,
           0.0,  0.0,  0.0, 25.0,  0.0, 0.0, 0.0, 0.0],
    
    'H': [-3.2,  0.5,  0.5,  7.6,  153.2, 195.0, 155.2, 10.4,
           1.00, 0.87, 0.95, 1.05,  0.32, 66.0, 13.7, 0.230,
           1.0,  1.0,  25.0, 118.0,  0.5, 0.0, 0.0, 0.0],
    
    'I': [ 4.5, -1.8,  0.0,  6.0,  166.7, 175.0, 131.2,  5.2,
           1.08, 1.60, 0.47, 0.88,  0.46, 45.0, 21.4, 0.186,
           0.0,  0.0,  0.0, 140.0,  0.0, 1.0, 0.0, 0.0],
    
    'K': [-3.9,  3.0,  1.0,  9.7,  168.6, 200.0, 146.2, 11.3,
           1.16, 0.74, 1.01, 1.04,  0.47, 56.0, 15.7, 0.219,
           1.0,  0.0,  25.0, 115.0,  0.0, 0.0, 0.0, 0.0],
    
    'L': [ 3.8, -1.8,  0.0,  6.0,  166.7, 170.0, 131.2,  4.9,
           1.21, 1.30, 0.59, 0.89,  0.40, 40.0, 21.4, 0.186,
           0.0,  0.0,  0.0, 137.0,  0.0, 1.0, 0.0, 0.0],
    
    'M': [ 1.9, -1.3,  0.0,  5.7,  162.9, 185.0, 149.2,  5.7,
           1.45, 1.05, 0.60, 0.93,  0.29, 94.0, 16.3, 0.221,
           0.0,  0.0,  0.0, 136.0,  0.0, 0.0, 1.0, 0.0],
    
    'N': [-3.5,  0.2,  0.0,  5.4,  114.1, 160.0, 132.1, 11.6,
           0.67, 0.89, 1.56, 1.04,  0.46, 104.0, 12.8, 0.134,
           1.0,  1.0,  62.0, 44.0,  0.0, 0.0, 0.0, 0.0],
    
    'P': [-1.6,  0.0,  0.0,  6.3,  112.7, 145.0, 115.1,  8.0,
           0.57, 0.55, 1.52, 1.17,  0.51, 56.0, 17.4, 0.131,
           0.0,  0.0,  0.0, 105.0,  0.0, 0.0, 0.0, 0.0],
    
    'Q': [-3.5,  0.2,  0.0,  5.7,  143.8, 180.0, 146.2, 10.5,
           1.11, 1.10, 0.98, 0.93,  0.49, 93.0, 14.5, 0.180,
           1.0,  1.0,  62.0, 59.0,  0.0, 0.0, 0.0, 0.0],
    
    'R': [-4.5,  3.0,  1.0, 10.8,  173.4, 225.0, 174.2, 10.5,
           0.98, 0.93, 0.95, 1.03,  0.52, 65.0, 14.3, 0.291,
           2.0,  0.0,  75.0, 93.0,  0.0, 0.0, 0.0, 0.0],
    
    'S': [-0.8,  0.3,  0.0,  5.7,   89.0, 115.0, 105.1,  9.2,
           0.77, 0.75, 1.43, 1.07,  0.51, 117.0,  9.5, 0.062,
           1.0,  1.0,  36.0, 46.0,  0.0, 0.0, 0.0, 1.0],
    
    'T': [-0.7, -0.4,  0.0,  5.6,  116.1, 140.0, 119.1,  8.6,
           0.83, 1.19, 0.96, 1.00,  0.44, 107.0, 15.8, 0.108,
           1.0,  1.0,  36.0, 74.0,  0.0, 0.0, 0.0, 1.0],
    
    'V': [ 4.2, -1.5,  0.0,  6.0,  140.0, 155.0, 117.1,  5.9,
           1.06, 1.70, 0.50, 0.85,  0.39, 49.0, 21.6, 0.140,
           0.0,  0.0,  0.0, 117.0,  0.0, 1.0, 0.0, 0.0],
    
    'W': [-0.9, -3.4,  0.0,  5.9,  227.8, 255.0, 204.2,  5.4,
           1.08, 1.37, 0.96, 0.85,  0.30, 18.0, 21.7, 0.409,
           1.0,  0.0,  0.0, 190.0,  1.0, 0.0, 0.0, 0.0],
    
    'Y': [-1.3, -2.3,  0.0,  5.7,  193.6, 230.0, 181.2,  6.2,
           0.69, 1.47, 1.14, 0.94,  0.42, 41.0, 18.0, 0.298,
           1.0,  1.0,  36.0, 141.0,  1.0, 0.0, 0.0, 1.0],
    
    # For unknown/ambiguous amino acids, use average values
    'X': [ 0.0,  0.0,  0.0,  6.0,  140.0, 165.0, 137.0,  8.0,
           1.00, 1.00, 1.00, 1.00,  0.42, 65.0, 15.0, 0.170,
           0.4,  0.5,  25.0, 100.0,  0.2, 0.3, 0.1, 0.2],
    
    'B': [-3.5,  1.6, -0.5,  4.1,  112.6, 155.0, 132.6, 12.3,
           0.84, 0.72, 1.51, 1.05,  0.49, 93.0, 12.3, 0.120,
           0.5,  1.5,  56.0, 52.0,  0.0, 0.0, 0.0, 0.0],
    
    'Z': [-3.5,  1.6, -0.5,  4.5,  141.1, 185.0, 146.7, 11.4,
           1.31, 0.74, 0.86, 0.96,  0.50, 88.0, 14.1, 0.166,
           0.5,  1.5,  56.0, 63.0,  0.0, 0.0, 0.0, 0.0],
    
    'U': [ 2.5, -1.0,  0.0,  5.5,  108.5, 135.0, 168.1,  5.5,
           0.70, 1.19, 1.19, 0.96,  0.35, 20.0, 13.5, 0.128,
           0.0,  0.0,  0.0, 104.0,  0.0, 0.0, 1.0, 0.0],
    
    'O': [-3.9,  3.0,  1.0,  9.7,  255.0, 275.0, 255.3, 11.3,
           1.16, 0.74, 1.01, 1.04,  0.47, 56.0, 20.0, 0.350,
           1.0,  0.0,  25.0, 180.0,  0.0, 0.0, 0.0, 0.0],
    
    'J': [ 4.2, -1.7,  0.0,  6.0,  166.7, 172.0, 131.2,  5.1,
           1.15, 1.45, 0.53, 0.89,  0.43, 43.0, 21.4, 0.186,
           0.0,  0.0,  0.0, 139.0,  0.0, 1.0, 0.0, 0.0],
}

# Property names for reference
PROPERTY_NAMES = [
    'hydrophobicity_kyte',    # Kyte-Doolittle hydrophobicity scale
    'hydrophobicity_hopp',    # Hopp-Woods hydrophilicity scale
    'charge_pH7',             # Net charge at physiological pH
    'pI',                     # Isoelectric point
    'volume',                 # Residue volume (Å³)
    'surface_area',           # Accessible surface area
    'mass',                   # Molecular mass (Da)
    'polarity_grantham',      # Grantham polarity scale
    'helix_propensity',       # Alpha helix propensity
    'sheet_propensity',       # Beta sheet propensity
    'turn_propensity',        # Turn propensity
    'coil_propensity',        # Coil propensity
    'flexibility',            # B-factor derived flexibility
    'mutability',             # Relative mutability
    'bulkiness',              # Bulkiness index
    'refractivity',           # Refractivity
    'H_bond_donor',           # Hydrogen bond donor count
    'H_bond_acceptor',        # Hydrogen bond acceptor count
    'polar_surface',          # Polar surface area
    'nonpolar_surface',       # Nonpolar surface area
    'aromatic',               # Aromaticity (F, W, Y, H partial)
    'aliphatic',              # Aliphatic character (A, I, L, V)
    'sulfur',                 # Contains sulfur (C, M)
    'hydroxyl',               # Contains hydroxyl (S, T, Y)
]

NUM_PROPERTIES = len(PROPERTY_NAMES)


class PhysicochemicalEncoder:
    """
    Encodes amino acids into normalized physicochemical property vectors.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: If True, z-score normalize each property
        """
        self.normalize = normalize
        self._setup_properties()
    
    def _setup_properties(self):
        """Precompute normalized property matrix."""
        # Get properties for standard 20 amino acids
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        props = np.array([AMINO_ACID_PROPERTIES_RAW[aa] for aa in standard_aa])
        
        # Compute normalization statistics from standard amino acids
        self.mean = props.mean(axis=0)
        self.std = props.std(axis=0) + 1e-8
        
        # Create lookup table for all amino acids
        self.property_lookup = {}
        for aa, values in AMINO_ACID_PROPERTIES_RAW.items():
            arr = np.array(values)
            if self.normalize:
                arr = (arr - self.mean) / self.std
            self.property_lookup[aa] = arr
        
        # Default for unknown characters
        self.default_props = self.property_lookup['X']
    
    def encode(self, amino_acid: str) -> np.ndarray:
        """Get property vector for a single amino acid."""
        return self.property_lookup.get(amino_acid.upper(), self.default_props)
    
    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a protein sequence to property matrix.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Array of shape [seq_len, num_properties]
        """
        return np.array([self.encode(aa) for aa in sequence])
    
    @property
    def num_properties(self) -> int:
        return NUM_PROPERTIES
    
    @property
    def property_names(self) -> List[str]:
        return PROPERTY_NAMES


# Create singleton encoder
PHYSICOCHEMICAL_ENCODER = PhysicochemicalEncoder(normalize=True)

# For convenience, expose the raw and normalized properties
PHYSICOCHEMICAL_PROPERTIES = AMINO_ACID_PROPERTIES_RAW
