"""
Data download utilities for GammaFold.

Supports downloading from:
- UniProt (SwissProt, TrEMBL)
- RCSB PDB
- AlphaFold Database
"""

import os
import gzip
import shutil
import logging
import hashlib
import requests
from pathlib import Path
from typing import Optional, List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    """Configuration for downloads."""
    output_dir: str = "data/raw"
    max_workers: int = 4
    chunk_size: int = 8192
    timeout: int = 30
    retry_count: int = 3
    verify_checksum: bool = True


class BaseDownloader:
    """Base class for data downloaders."""
    
    def __init__(self, config: Optional[DownloadConfig] = None):
        self.config = config or DownloadConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _download_file(
        self,
        url: str,
        output_path: str,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """Download a single file with retry logic."""
        for attempt in range(self.config.retry_count):
            try:
                response = requests.get(
                    url,
                    stream=True,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)
                
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_count - 1:
                    logger.error(f"Failed to download {url} after {self.config.retry_count} attempts")
                    return False
        
        return False
    
    def _decompress_gzip(self, input_path: str, output_path: str) -> bool:
        """Decompress a gzip file."""
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            logger.error(f"Failed to decompress {input_path}: {e}")
            return False


class UniProtDownloader(BaseDownloader):
    """
    Download protein sequences from UniProt.
    
    Supports:
    - SwissProt (reviewed, curated)
    - TrEMBL (unreviewed)
    - Organism-specific downloads
    """
    
    BASE_URL = "https://rest.uniprot.org/uniprotkb"
    FTP_BASE = "https://ftp.uniprot.org/pub/databases/uniprot"
    
    # Common organism taxonomy IDs
    ORGANISMS = {
        # Mammals
        'human': 9606,
        'mouse': 10090,
        'rat': 10116,
        'bovine': 9913,
        'pig': 9823,
        'dog': 9615,
        'rabbit': 9986,
        # Model organisms
        'yeast': 559292,
        'ecoli': 83333,
        'drosophila': 7227,
        'zebrafish': 7955,
        'xenopus': 8364,
        'celegans': 6239,
        # Plants
        'arabidopsis': 3702,
        'rice': 39947,
        'maize': 4577,
        # Bacteria
        'bacillus': 224308,
        'pseudomonas': 208964,
        'salmonella': 99287,
        'streptococcus': 1311,
        # Other
        'chicken': 9031,
        'chimpanzee': 9598,
    }
    
    def download_swissprot(self, output_name: str = "swissprot.fasta.gz") -> str:
        """
        Download complete SwissProt database.
        
        Returns:
            Path to downloaded file
        """
        url = f"{self.FTP_BASE}/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
        output_path = os.path.join(self.config.output_dir, output_name)
        
        logger.info(f"Downloading SwissProt to {output_path}")
        
        if self._download_file(url, output_path):
            logger.info(f"SwissProt downloaded: {output_path}")
            return output_path
        
        raise RuntimeError("Failed to download SwissProt")
    
    def download_by_organism(
        self,
        organism: str,
        reviewed_only: bool = True,
        max_sequences: Optional[int] = None
    ) -> str:
        """
        Download sequences for a specific organism.
        
        Args:
            organism: Organism name or taxonomy ID
            reviewed_only: Only download reviewed (SwissProt) entries
            max_sequences: Maximum number of sequences to download
            
        Returns:
            Path to downloaded FASTA file
        """
        # Get taxonomy ID
        if organism.lower() in self.ORGANISMS:
            tax_id = self.ORGANISMS[organism.lower()]
        else:
            try:
                tax_id = int(organism)
            except ValueError:
                raise ValueError(f"Unknown organism: {organism}")
        
        # Build query
        query = f"organism_id:{tax_id}"
        if reviewed_only:
            query += " AND reviewed:true"
        
        # Build URL
        params = {
            "query": query,
            "format": "fasta",
            "compressed": "true",
        }
        if max_sequences:
            params["size"] = max_sequences
        
        url = f"{self.BASE_URL}/stream?" + "&".join(f"{k}={v}" for k, v in params.items())
        
        output_name = f"{organism.lower()}_{'swissprot' if reviewed_only else 'all'}.fasta.gz"
        output_path = os.path.join(self.config.output_dir, output_name)
        
        logger.info(f"Downloading {organism} sequences to {output_path}")
        
        if self._download_file(url, output_path):
            # Decompress
            decompressed_path = output_path.replace('.gz', '')
            self._decompress_gzip(output_path, decompressed_path)
            logger.info(f"Downloaded and decompressed: {decompressed_path}")
            return decompressed_path
        
        raise RuntimeError(f"Failed to download sequences for {organism}")
    
    def download_by_keywords(
        self,
        keywords: List[str],
        output_name: str,
        reviewed_only: bool = True
    ) -> str:
        """
        Download sequences matching keywords.
        
        Args:
            keywords: List of keywords to search
            output_name: Output filename
            reviewed_only: Only reviewed entries
            
        Returns:
            Path to downloaded file
        """
        keyword_query = " OR ".join(f'keyword:"{kw}"' for kw in keywords)
        query = f"({keyword_query})"
        if reviewed_only:
            query += " AND reviewed:true"
        
        params = {
            "query": query,
            "format": "fasta",
            "compressed": "true",
        }
        
        url = f"{self.BASE_URL}/stream?" + "&".join(f"{k}={v}" for k, v in params.items())
        output_path = os.path.join(self.config.output_dir, output_name)
        
        if self._download_file(url, output_path):
            decompressed_path = output_path.replace('.gz', '')
            if output_path.endswith('.gz'):
                self._decompress_gzip(output_path, decompressed_path)
                return decompressed_path
            return output_path
        
        raise RuntimeError(f"Failed to download sequences for keywords: {keywords}")


class PDBDownloader(BaseDownloader):
    """
    Download protein structures from RCSB PDB.
    
    Supports:
    - Individual PDB files
    - Bulk downloads by criteria
    - Resolution/method filtering
    """
    
    BASE_URL = "https://files.rcsb.org/download"
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    def download_pdb(self, pdb_id: str) -> str:
        """
        Download a single PDB file.
        
        Args:
            pdb_id: 4-character PDB ID
            
        Returns:
            Path to downloaded file
        """
        pdb_id = pdb_id.upper()
        url = f"{self.BASE_URL}/{pdb_id}.pdb"
        output_path = os.path.join(self.config.output_dir, f"{pdb_id}.pdb")
        
        if self._download_file(url, output_path):
            return output_path
        
        raise RuntimeError(f"Failed to download PDB {pdb_id}")
    
    def download_pdbs(self, pdb_ids: List[str], show_progress: bool = True) -> List[str]:
        """
        Download multiple PDB files in parallel.
        
        Args:
            pdb_ids: List of PDB IDs
            show_progress: Show progress bar
            
        Returns:
            List of downloaded file paths
        """
        downloaded = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.download_pdb, pdb_id): pdb_id for pdb_id in pdb_ids}
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(pdb_ids), desc="Downloading PDBs")
            
            for future in iterator:
                try:
                    path = future.result()
                    downloaded.append(path)
                except Exception as e:
                    pdb_id = futures[future]
                    logger.warning(f"Failed to download {pdb_id}: {e}")
        
        return downloaded
    
    def search_by_resolution(
        self,
        max_resolution: float = 2.5,
        method: str = "X-RAY DIFFRACTION",
        limit: int = 1000
    ) -> List[str]:
        """
        Search for PDB IDs by resolution and method.
        
        Args:
            max_resolution: Maximum resolution in Angstroms
            method: Experimental method
            limit: Maximum number of results
            
        Returns:
            List of PDB IDs matching criteria
        """
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": max_resolution
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": method
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "paginate": {
                    "start": 0,
                    "rows": limit
                }
            }
        }
        
        try:
            response = requests.post(self.SEARCH_URL, json=query, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [entry["identifier"] for entry in data.get("result_set", [])]
        except Exception as e:
            logger.error(f"PDB search failed: {e}")
            return []
    
    def download_by_resolution(
        self,
        max_resolution: float = 2.5,
        method: str = "X-RAY DIFFRACTION",
        limit: int = 100
    ) -> List[str]:
        """
        Download PDB files by resolution criteria.
        
        Args:
            max_resolution: Maximum resolution
            method: Experimental method
            limit: Maximum number to download
            
        Returns:
            List of downloaded file paths
        """
        pdb_ids = self.search_by_resolution(max_resolution, method, limit)
        logger.info(f"Found {len(pdb_ids)} PDBs with resolution <= {max_resolution}Å")
        return self.download_pdbs(pdb_ids)


class AlphaFoldDownloader(BaseDownloader):
    """
    Download predicted structures from AlphaFold Database.
    """
    
    BASE_URL = "https://alphafold.ebi.ac.uk/files"
    
    def download_by_uniprot(self, uniprot_id: str) -> str:
        """
        Download AlphaFold prediction for a UniProt ID.
        
        Args:
            uniprot_id: UniProt accession
            
        Returns:
            Path to downloaded PDB file
        """
        url = f"{self.BASE_URL}/AF-{uniprot_id}-F1-model_v4.pdb"
        output_path = os.path.join(self.config.output_dir, f"AF-{uniprot_id}.pdb")
        
        if self._download_file(url, output_path):
            return output_path
        
        raise RuntimeError(f"Failed to download AlphaFold prediction for {uniprot_id}")
    
    def download_proteome(self, organism: str = "human") -> str:
        """
        Download complete proteome predictions.
        
        Args:
            organism: Organism name (human, mouse, etc.)
            
        Returns:
            Path to downloaded archive
        """
        # Proteome IDs
        proteomes = {
            'human': 'UP000005640',
            'mouse': 'UP000000589',
            'yeast': 'UP000002311',
            'ecoli': 'UP000000625',
        }
        
        if organism.lower() not in proteomes:
            raise ValueError(f"Unknown organism: {organism}")
        
        proteome_id = proteomes[organism.lower()]
        url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{proteome_id}_HUMAN_v4.tar"
        output_path = os.path.join(self.config.output_dir, f"alphafold_{organism}.tar")
        
        logger.info(f"Downloading AlphaFold proteome for {organism}")
        
        if self._download_file(url, output_path):
            return output_path
        
        raise RuntimeError(f"Failed to download AlphaFold proteome for {organism}")


def download_training_data(
    output_dir: str = "data/raw",
    organisms: List[str] = None,
    include_pdb: bool = True,
    pdb_resolution: float = 2.5,
    pdb_limit: int = 1000
) -> Dict[str, List[str]]:
    """
    Download a complete training dataset.
    
    Args:
        output_dir: Output directory
        organisms: List of organisms to download (default: human)
        include_pdb: Whether to download PDB structures
        pdb_resolution: Maximum PDB resolution
        pdb_limit: Maximum number of PDBs
        
    Returns:
        Dictionary with paths to downloaded files
    """
    if organisms is None:
        organisms = ['human']
    
    config = DownloadConfig(output_dir=output_dir)
    result = {'sequences': [], 'structures': []}
    
    # Download sequences
    uniprot = UniProtDownloader(config)
    for organism in organisms:
        try:
            path = uniprot.download_by_organism(organism, reviewed_only=True)
            result['sequences'].append(path)
            logger.info(f"Downloaded sequences for {organism}")
        except Exception as e:
            logger.error(f"Failed to download {organism}: {e}")
    
    # Download structures
    if include_pdb:
        pdb = PDBDownloader(config)
        paths = pdb.download_by_resolution(pdb_resolution, limit=pdb_limit)
        result['structures'] = paths
        logger.info(f"Downloaded {len(paths)} PDB structures")
    
    return result
