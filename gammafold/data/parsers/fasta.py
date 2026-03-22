"""
FASTA file parser for GammaFold.

Supports:
- Standard FASTA format
- Multi-sequence files
- Header parsing for metadata
"""

import os
import re
from typing import List, Dict, Tuple, Iterator, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FastaRecord:
    """Container for a single FASTA record."""
    id: str
    description: str
    sequence: str
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def __repr__(self):
        return f"FastaRecord(id='{self.id}', length={self.length})"


def parse_fasta(path: str) -> List[FastaRecord]:
    """
    Parse a FASTA file and return list of records.
    
    Args:
        path: Path to FASTA file
        
    Returns:
        List of FastaRecord objects
    """
    records = []
    current_id = None
    current_desc = ""
    current_seq = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous record
                if current_id is not None:
                    records.append(FastaRecord(
                        id=current_id,
                        description=current_desc,
                        sequence=''.join(current_seq)
                    ))
                
                # Parse header
                header = line[1:]
                parts = header.split(None, 1)
                current_id = parts[0] if parts else "unknown"
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line.upper())
    
    # Save last record
    if current_id is not None:
        records.append(FastaRecord(
            id=current_id,
            description=current_desc,
            sequence=''.join(current_seq)
        ))
    
    return records


def iter_fasta(path: str) -> Iterator[FastaRecord]:
    """
    Iterate over FASTA file records (memory efficient).
    
    Args:
        path: Path to FASTA file
        
    Yields:
        FastaRecord objects one at a time
    """
    current_id = None
    current_desc = ""
    current_seq = []
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_id is not None:
                    yield FastaRecord(
                        id=current_id,
                        description=current_desc,
                        sequence=''.join(current_seq)
                    )
                
                header = line[1:]
                parts = header.split(None, 1)
                current_id = parts[0] if parts else "unknown"
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq = []
            else:
                current_seq.append(line.upper())
    
    if current_id is not None:
        yield FastaRecord(
            id=current_id,
            description=current_desc,
            sequence=''.join(current_seq)
        )


def write_fasta(records: List[FastaRecord], path: str, line_width: int = 80):
    """
    Write FASTA records to file.
    
    Args:
        records: List of FastaRecord objects
        path: Output file path
        line_width: Characters per line for sequence
    """
    with open(path, 'w') as f:
        for record in records:
            # Write header
            if record.description:
                f.write(f">{record.id} {record.description}\n")
            else:
                f.write(f">{record.id}\n")
            
            # Write sequence with line wrapping
            seq = record.sequence
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + '\n')


def parse_uniprot_header(header: str) -> Dict[str, str]:
    """
    Parse UniProt FASTA header format.
    
    Format: >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier ...
    
    Args:
        header: Full header line (without >)
        
    Returns:
        Dictionary with parsed fields
    """
    result = {}
    
    # Parse identifier part
    parts = header.split(None, 1)
    if not parts:
        return result
    
    id_parts = parts[0].split('|')
    if len(id_parts) >= 3:
        result['db'] = id_parts[0]
        result['accession'] = id_parts[1]
        result['entry_name'] = id_parts[2]
    else:
        result['accession'] = parts[0]
    
    # Parse description and metadata
    if len(parts) > 1:
        desc = parts[1]
        
        # Extract protein name (before OS=)
        os_match = re.search(r'\s+OS=', desc)
        if os_match:
            result['protein_name'] = desc[:os_match.start()]
            
            # Parse key=value pairs
            for match in re.finditer(r'(\w+)=([^=]+?)(?=\s+\w+=|\s*$)', desc[os_match.start():]):
                result[match.group(1).lower()] = match.group(2).strip()
        else:
            result['protein_name'] = desc
    
    return result
