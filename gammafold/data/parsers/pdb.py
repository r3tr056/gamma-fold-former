"""
PDB file parser for GammaFold.

Extracts:
- Amino acid sequences
- Cα coordinates
- Secondary structure (from HELIX/SHEET records)
- B-factors for confidence

OPTIMIZED: O(1) secondary structure lookup using interval search
"""

import os
import bisect
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# Three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XLE': 'J',
    'UNK': 'X',
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


class SSIntervalTree:
    """
    Efficient secondary structure lookup using sorted intervals.
    
    Provides O(log n) lookup instead of O(n) per residue.
    """
    
    def __init__(self):
        # Separate lists for each chain
        self.helix_intervals: Dict[str, List[Tuple[int, int]]] = {}
        self.sheet_intervals: Dict[str, List[Tuple[int, int]]] = {}
        self._helix_starts: Dict[str, List[int]] = {}
        self._sheet_starts: Dict[str, List[int]] = {}
    
    def add_helix(self, chain_id: str, start: int, end: int):
        """Add a helix interval."""
        if chain_id not in self.helix_intervals:
            self.helix_intervals[chain_id] = []
        self.helix_intervals[chain_id].append((start, end))
    
    def add_sheet(self, chain_id: str, start: int, end: int):
        """Add a sheet interval."""
        if chain_id not in self.sheet_intervals:
            self.sheet_intervals[chain_id] = []
        self.sheet_intervals[chain_id].append((start, end))
    
    def build(self):
        """Sort intervals and build index for binary search."""
        for chain_id in self.helix_intervals:
            self.helix_intervals[chain_id].sort()
            self._helix_starts[chain_id] = [s for s, e in self.helix_intervals[chain_id]]
        
        for chain_id in self.sheet_intervals:
            self.sheet_intervals[chain_id].sort()
            self._sheet_starts[chain_id] = [s for s, e in self.sheet_intervals[chain_id]]
    
    def lookup(self, chain_id: str, res_id: int) -> str:
        """
        Look up secondary structure for a residue. O(log n).
        
        Returns: 'H' (helix), 'E' (sheet), or 'C' (coil)
        """
        # Check helices
        if chain_id in self._helix_starts:
            starts = self._helix_starts[chain_id]
            intervals = self.helix_intervals[chain_id]
            idx = bisect.bisect_right(starts, res_id) - 1
            if idx >= 0 and intervals[idx][0] <= res_id <= intervals[idx][1]:
                return 'H'
        
        # Check sheets
        if chain_id in self._sheet_starts:
            starts = self._sheet_starts[chain_id]
            intervals = self.sheet_intervals[chain_id]
            idx = bisect.bisect_right(starts, res_id) - 1
            if idx >= 0 and intervals[idx][0] <= res_id <= intervals[idx][1]:
                return 'E'
        
        return 'C'
    
    def batch_lookup(self, chain_id: str, res_ids: List[int]) -> str:
        """Batch lookup for a list of residue IDs. Returns SS string."""
        return ''.join(self.lookup(chain_id, rid) for rid in res_ids)


@dataclass
class PDBChain:
    """Container for a single PDB chain."""
    chain_id: str
    sequence: str
    ca_coords: np.ndarray  # [L, 3]
    residue_ids: List[int]
    b_factors: np.ndarray  # [L]
    secondary_structure: str  # DSSP-style SS string
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def to_tensors(self) -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return {
            'sequence': self.sequence,
            'ca_coords': torch.tensor(self.ca_coords, dtype=torch.float32),
            'b_factors': torch.tensor(self.b_factors, dtype=torch.float32),
        }


@dataclass
class PDBStructure:
    """Container for a complete PDB structure."""
    pdb_id: str
    chains: Dict[str, PDBChain] = field(default_factory=dict)
    resolution: Optional[float] = None
    method: Optional[str] = None
    
    def get_chain(self, chain_id: str) -> Optional[PDBChain]:
        return self.chains.get(chain_id)
    
    @property
    def all_chains(self) -> List[str]:
        return list(self.chains.keys())
    
    def __repr__(self):
        return f"PDBStructure(id='{self.pdb_id}', chains={self.all_chains})"


def parse_pdb(path: str, chains: Optional[List[str]] = None) -> PDBStructure:
    """
    Parse a PDB file and extract structure information.
    
    OPTIMIZED: Uses SSIntervalTree for O(log n) secondary structure lookup.
    
    Args:
        path: Path to PDB file
        chains: Optional list of chain IDs to extract (None = all)
        
    Returns:
        PDBStructure object with extracted data
    """
    pdb_id = os.path.basename(path).replace('.pdb', '').replace('.ent', '')
    
    # Data storage
    chain_data = {}  # chain_id -> {residue_id: {atom_data}}
    ss_tree = SSIntervalTree()
    resolution = None
    method = None
    
    with open(path, 'r') as f:
        for line in f:
            record = line[:6].strip()
            
            # Header records
            if record == 'EXPDTA':
                method = line[10:79].strip()
            
            elif record == 'REMARK':
                # Resolution is in REMARK 2
                if line[7:10].strip() == '2':
                    if 'RESOLUTION' in line:
                        try:
                            res_str = line[23:30].strip()
                            if res_str and 'NOT' not in res_str:
                                resolution = float(res_str)
                        except ValueError:
                            pass
            
            # Secondary structure - collect into interval tree
            elif record == 'HELIX':
                chain_id = line[19].strip()
                try:
                    start_res = int(line[21:25].strip())
                    end_res = int(line[33:37].strip())
                    ss_tree.add_helix(chain_id, start_res, end_res)
                except ValueError:
                    pass
            
            elif record == 'SHEET':
                chain_id = line[21].strip()
                try:
                    start_res = int(line[22:26].strip())
                    end_res = int(line[33:37].strip())
                    ss_tree.add_sheet(chain_id, start_res, end_res)
                except ValueError:
                    pass
            
            # Atom records
            elif record == 'ATOM':
                atom_name = line[12:16].strip()
                
                # Only keep Cα atoms
                if atom_name != 'CA':
                    continue
                
                chain_id = line[21].strip()
                
                # Filter by requested chains
                if chains is not None and chain_id not in chains:
                    continue
                
                try:
                    res_name = line[17:20].strip()
                    res_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    b_factor = float(line[60:66].strip()) if len(line) >= 66 else 0.0
                    
                    # Initialize chain data
                    if chain_id not in chain_data:
                        chain_data[chain_id] = {}
                    
                    # Store residue data (keep first occurrence)
                    if res_id not in chain_data[chain_id]:
                        chain_data[chain_id][res_id] = {
                            'res_name': res_name,
                            'coords': [x, y, z],
                            'b_factor': b_factor
                        }
                except ValueError:
                    continue
    
    # Build SS interval tree for O(log n) lookup
    ss_tree.build()
    
    # Build PDBStructure
    structure = PDBStructure(pdb_id=pdb_id, resolution=resolution, method=method)
    
    for chain_id, residues in chain_data.items():
        # Sort by residue ID
        sorted_ids = sorted(residues.keys())
        
        # Pre-allocate arrays for efficiency
        seq_len = len(sorted_ids)
        sequence = []
        coords = np.empty((seq_len, 3), dtype=np.float32)
        b_factors = np.empty(seq_len, dtype=np.float32)
        
        for i, res_id in enumerate(sorted_ids):
            res_data = residues[res_id]
            
            # Convert residue name
            aa = THREE_TO_ONE.get(res_data['res_name'], 'X')
            sequence.append(aa)
            coords[i] = res_data['coords']
            b_factors[i] = res_data['b_factor']
        
        # Get secondary structure using optimized interval tree
        ss_string = ss_tree.batch_lookup(chain_id, sorted_ids)
        
        if sequence:
            chain = PDBChain(
                chain_id=chain_id,
                sequence=''.join(sequence),
                ca_coords=coords,
                residue_ids=sorted_ids,
                b_factors=b_factors,
                secondary_structure=ss_string
            )
            structure.chains[chain_id] = chain
    
    return structure


def extract_ca_coordinates(pdb_path: str, chain_id: Optional[str] = None) -> Tuple[str, torch.Tensor]:
    """
    Simple helper to extract sequence and Cα coordinates from PDB.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Optional chain ID (uses first chain if None)
        
    Returns:
        Tuple of (sequence, coords tensor [L, 3])
    """
    structure = parse_pdb(pdb_path)
    
    if not structure.chains:
        raise ValueError(f"No chains found in {pdb_path}")
    
    if chain_id is None:
        chain_id = list(structure.chains.keys())[0]
    
    chain = structure.get_chain(chain_id)
    if chain is None:
        raise ValueError(f"Chain {chain_id} not found in {pdb_path}")
    
    return chain.sequence, torch.tensor(chain.ca_coords, dtype=torch.float32)


def write_pdb(
    sequence: str,
    coords: torch.Tensor,
    output_path: str,
    chain_id: str = 'A',
    b_factors: Optional[torch.Tensor] = None
):
    """
    Write a PDB file from sequence and coordinates.
    
    Args:
        sequence: Amino acid sequence
        coords: [L, 3] Cα coordinates
        output_path: Output file path
        chain_id: Chain identifier
        b_factors: Optional [L] B-factors
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    
    if b_factors is not None and isinstance(b_factors, torch.Tensor):
        b_factors = b_factors.detach().cpu().numpy()
    else:
        b_factors = np.zeros(len(sequence))
    
    with open(output_path, 'w') as f:
        atom_num = 1
        for i, (aa, coord, bf) in enumerate(zip(sequence, coords, b_factors)):
            res_name = ONE_TO_THREE.get(aa, 'UNK')
            res_id = i + 1
            x, y, z = coord
            
            # PDB ATOM record format
            line = (
                f"ATOM  {atom_num:5d}  CA  {res_name:3s} {chain_id}{res_id:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           C  \n"
            )
            f.write(line)
            atom_num += 1
        
        f.write("END\n")
