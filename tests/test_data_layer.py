"""
Comprehensive unit tests for GammaFold data layer.
Tests all outputs thoroughly against expected values.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVocabulary(unittest.TestCase):
    """Test vocabulary correctness."""
    
    def setUp(self):
        from gammafold.data.vocabulary import VOCABULARY
        self.vocab = VOCABULARY
    
    def test_vocab_sizes(self):
        """Test vocabulary size constants."""
        self.assertEqual(self.vocab.VOCAB_SIZE, 36)
        self.assertEqual(self.vocab.SS_VOCAB_SIZE, 10)
    
    def test_standard_amino_acids(self):
        """Test all 20 standard amino acids are present with correct IDs."""
        expected = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        }
        self.assertEqual(self.vocab.AMINO_ACIDS, expected)
        self.assertEqual(len(self.vocab.AMINO_ACIDS), 20)
    
    def test_extended_amino_acids(self):
        """Test extended/ambiguous amino acids."""
        self.assertEqual(self.vocab.EXTENDED_AA['U'], 21)  # Selenocysteine
        self.assertEqual(self.vocab.EXTENDED_AA['X'], 26)  # Unknown
        self.assertEqual(len(self.vocab.EXTENDED_AA), 6)
    
    def test_special_tokens(self):
        """Test special tokens have correct IDs."""
        self.assertEqual(self.vocab.SPECIAL['<pad>'], 0)
        self.assertEqual(self.vocab.SPECIAL['<sos>'], 27)
        self.assertEqual(self.vocab.SPECIAL['<eos>'], 28)
        self.assertEqual(self.vocab.SPECIAL['<mask>'], 29)
        self.assertEqual(len(self.vocab.SPECIAL), 10)
    
    def test_secondary_structure_tokens(self):
        """Test DSSP secondary structure tokens."""
        ss = self.vocab.SECONDARY_STRUCTURE
        self.assertEqual(ss['<ss_pad>'], 0)
        self.assertEqual(ss['H'], 1)  # Alpha helix
        self.assertEqual(ss['E'], 4)  # Beta sheet
        self.assertEqual(ss['C'], 8)  # Coil
        self.assertEqual(len(ss), 10)
    
    def test_no_token_id_collisions(self):
        """Ensure no token IDs are duplicated."""
        all_tokens = self.vocab.get_all_tokens()
        ids = list(all_tokens.values())
        self.assertEqual(len(ids), len(set(ids)), "Duplicate token IDs found")


class TestPhysicochemicalProperties(unittest.TestCase):
    """Test physicochemical property encoding."""
    
    def setUp(self):
        from gammafold.data.properties import PHYSICOCHEMICAL_ENCODER, NUM_PROPERTIES, PROPERTY_NAMES
        self.encoder = PHYSICOCHEMICAL_ENCODER
        self.num_props = NUM_PROPERTIES
        self.prop_names = PROPERTY_NAMES
    
    def test_num_properties(self):
        """Test we have 24 properties."""
        self.assertEqual(self.num_props, 24)
        self.assertEqual(len(self.prop_names), 24)
    
    def test_property_names(self):
        """Test property names are as expected."""
        expected_first_five = [
            'hydrophobicity_kyte', 'hydrophobicity_hopp', 'charge_pH7', 'pI', 'volume'
        ]
        self.assertEqual(self.prop_names[:5], expected_first_five)
    
    def test_single_aa_encoding(self):
        """Test encoding individual amino acids."""
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            props = self.encoder.encode(aa)
            self.assertEqual(props.shape, (24,), f"AA {aa} should have 24 properties")
            self.assertFalse(np.isnan(props).any(), f"AA {aa} has NaN values")
    
    def test_sequence_encoding(self):
        """Test encoding a full sequence."""
        seq = "MKFLILLFN"
        encoded = self.encoder.encode_sequence(seq)
        self.assertEqual(encoded.shape, (9, 24))
        self.assertFalse(np.isnan(encoded).any())
    
    def test_normalization(self):
        """Test properties are normalized (mean ~0, std ~1)."""
        all_props = self.encoder.encode_sequence("ACDEFGHIKLMNPQRSTVWY")
        means = all_props.mean(axis=0)
        stds = all_props.std(axis=0)
        
        # Means should be close to 0 (within ±0.5)
        self.assertTrue(np.abs(means).max() < 0.5, 
                       f"Means not centered: max abs mean = {np.abs(means).max():.3f}")
    
    def test_hydrophobicity_ordering(self):
        """Test hydrophobic amino acids have higher hydrophobicity values."""
        # Kyte-Doolittle: I, V, L are hydrophobic (positive), D, E, K are hydrophilic (negative)
        ile = self.encoder.encode('I')[0]  # hydrophobicity_kyte
        val = self.encoder.encode('V')[0]
        asp = self.encoder.encode('D')[0]
        
        self.assertGreater(ile, asp, "Ile should be more hydrophobic than Asp")
        self.assertGreater(val, asp, "Val should be more hydrophobic than Asp")
    
    def test_charge_values(self):
        """Test charged amino acids have correct charge signs."""
        # Charge at pH 7 (property index 2)
        lys = self.encoder.encode('K')[2]  # +1 charge
        asp = self.encoder.encode('D')[2]  # -1 charge
        ala = self.encoder.encode('A')[2]  # 0 charge
        
        self.assertGreater(lys, ala, "Lys should have positive charge relative to Ala")
        self.assertLess(asp, ala, "Asp should have negative charge relative to Ala")


class TestFastaParser(unittest.TestCase):
    """Test FASTA file parsing."""
    
    def setUp(self):
        from gammafold.data.parsers.fasta import parse_fasta, parse_uniprot_header
        self.parse_fasta = parse_fasta
        self.parse_uniprot_header = parse_uniprot_header
        self.test_file = 'data/test_samples/P00533.fasta'
    
    def test_parse_egfr(self):
        """Test parsing EGFR FASTA file."""
        records = self.parse_fasta(self.test_file)
        
        self.assertEqual(len(records), 1)
        rec = records[0]
        
        self.assertEqual(rec.id, "sp|P00533|EGFR_HUMAN")
        self.assertEqual(rec.length, 1210)
        self.assertTrue(rec.sequence.startswith("MRPSGTAGAALLALLAALCPASRA"))
        self.assertTrue(rec.sequence.endswith("TAENAEYLRVAPQSSEFIGA"))
        
        # Check sequence contains only valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        seq_aa = set(rec.sequence)
        self.assertTrue(seq_aa.issubset(valid_aa), 
                       f"Invalid AA found: {seq_aa - valid_aa}")
    
    def test_uniprot_header_parsing(self):
        """Test parsing UniProt header format."""
        header = "sp|P00533|EGFR_HUMAN Epidermal growth factor receptor OS=Homo sapiens OX=9606 GN=EGFR PE=1 SV=2"
        parsed = self.parse_uniprot_header(header)
        
        self.assertEqual(parsed['db'], 'sp')
        self.assertEqual(parsed['accession'], 'P00533')
        self.assertEqual(parsed['entry_name'], 'EGFR_HUMAN')
        self.assertEqual(parsed['protein_name'], 'Epidermal growth factor receptor')
        self.assertEqual(parsed['os'], 'Homo sapiens')
        self.assertEqual(parsed['ox'], '9606')
        self.assertEqual(parsed['gn'], 'EGFR')
        self.assertEqual(parsed['pe'], '1')
        self.assertEqual(parsed['sv'], '2')


class TestPdbParser(unittest.TestCase):
    """Test PDB file parsing."""
    
    def setUp(self):
        from gammafold.data.parsers.pdb import parse_pdb, extract_ca_coordinates
        self.parse_pdb = parse_pdb
        self.extract_ca_coordinates = extract_ca_coordinates
        self.crambin_file = 'data/test_samples/1CRN.pdb'
        self.hemoglobin_file = 'data/test_samples/4HHB.pdb'
    
    def test_parse_crambin_metadata(self):
        """Test parsing Crambin structure metadata."""
        structure = self.parse_pdb(self.crambin_file)
        
        self.assertEqual(structure.pdb_id, "1CRN")
        self.assertEqual(structure.resolution, 1.5)
        self.assertEqual(structure.method, "X-RAY DIFFRACTION")
        self.assertEqual(structure.all_chains, ['A'])
    
    def test_parse_crambin_chain(self):
        """Test parsing Crambin chain A."""
        structure = self.parse_pdb(self.crambin_file)
        chain = structure.get_chain('A')
        
        self.assertIsNotNone(chain)
        self.assertEqual(chain.length, 46)
        self.assertEqual(chain.sequence, "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN")
    
    def test_crambin_coordinates(self):
        """Test Crambin Cα coordinates."""
        structure = self.parse_pdb(self.crambin_file)
        chain = structure.get_chain('A')
        coords = chain.ca_coords
        
        self.assertEqual(coords.shape, (46, 3))
        
        # First Cα should be at approximately (16.967, 12.784, 4.338)
        np.testing.assert_array_almost_equal(
            coords[0], [16.967, 12.784, 4.338], decimal=2
        )
        
        # Check coordinates are reasonable (no extreme values)
        self.assertTrue(np.abs(coords).max() < 100, "Coordinates out of range")
    
    def test_crambin_bfactors(self):
        """Test B-factor extraction."""
        structure = self.parse_pdb(self.crambin_file)
        chain = structure.get_chain('A')
        
        self.assertEqual(chain.b_factors.shape, (46,))
        self.assertGreater(chain.b_factors.min(), 0)
        self.assertLess(chain.b_factors.max(), 100)
    
    def test_crambin_secondary_structure(self):
        """Test secondary structure extraction from HELIX/SHEET records."""
        structure = self.parse_pdb(self.crambin_file)
        chain = structure.get_chain('A')
        
        ss = chain.secondary_structure
        self.assertEqual(len(ss), 46)
        
        # Crambin has beta sheet at start (E), helix in middle (H), coil (C)
        self.assertTrue(ss.startswith("EEEE"), f"Expected EEEE at start, got {ss[:4]}")
        self.assertIn('H', ss, "Should contain helix")
        self.assertIn('C', ss, "Should contain coil")
    
    def test_parse_hemoglobin_multichain(self):
        """Test parsing multi-chain structure."""
        structure = self.parse_pdb(self.hemoglobin_file)
        
        self.assertEqual(structure.pdb_id, "4HHB")
        self.assertEqual(len(structure.chains), 4)
        self.assertEqual(sorted(structure.all_chains), ['A', 'B', 'C', 'D'])
        
        # Alpha subunits (A, C) should have 141 residues
        self.assertEqual(structure.get_chain('A').length, 141)
        self.assertEqual(structure.get_chain('C').length, 141)
        
        # Beta subunits (B, D) should have 146 residues
        self.assertEqual(structure.get_chain('B').length, 146)
        self.assertEqual(structure.get_chain('D').length, 146)
    
    def test_extract_ca_helper(self):
        """Test the extract_ca_coordinates helper function."""
        import torch
        seq, coords = self.extract_ca_coordinates(self.crambin_file)
        
        self.assertEqual(seq, "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN")
        self.assertIsInstance(coords, torch.Tensor)
        self.assertEqual(coords.shape, (46, 3))


class TestTokenizer(unittest.TestCase):
    """Test tokenizer functionality."""
    
    def setUp(self):
        from gammafold.data.tokenizer import GammaFoldTokenizer
        self.tokenizer = GammaFoldTokenizer(max_length=64)
    
    def test_encode_simple_sequence(self):
        """Test encoding a simple sequence."""
        seq = "MKFL"
        encoded = self.tokenizer.encode(seq)
        
        # Should be: <sos>=27, M=11, K=9, F=5, L=10, <eos>=28, then padding
        expected_tokens = [27, 11, 9, 5, 10, 28] + [0] * 58
        self.assertEqual(encoded.token_ids.tolist(), expected_tokens)
    
    def test_encode_token_ids_shape(self):
        """Test token IDs have correct shape."""
        encoded = self.tokenizer.encode("ACDEFGHIK")
        self.assertEqual(encoded.token_ids.shape[0], 64)  # max_length
    
    def test_encode_properties_shape(self):
        """Test properties have correct shape."""
        encoded = self.tokenizer.encode("ACDEFGHIK")
        self.assertEqual(encoded.properties.shape, (64, 24))
    
    def test_encode_attention_mask(self):
        """Test attention mask is correct."""
        encoded = self.tokenizer.encode("MKFL")
        
        # 6 real tokens (<sos>, M, K, F, L, <eos>), rest padding
        self.assertEqual(encoded.attention_mask.sum().item(), 6)
        self.assertEqual(encoded.attention_mask[:6].tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(encoded.attention_mask[6:10].tolist(), [0, 0, 0, 0])
    
    def test_encode_with_secondary_structure(self):
        """Test encoding with secondary structure."""
        seq = "MKFL"
        ss = "CHHE"  # Coil, Helix, Helix, Sheet
        encoded = self.tokenizer.encode(seq, secondary_structure=ss)
        
        # SS tokens: <ss_pad>=0, C=8, H=1, H=1, E=4, <ss_pad>=0, then 0s
        expected_ss = [0, 8, 1, 1, 4, 0] + [0] * 58
        self.assertEqual(encoded.ss_tokens.tolist(), expected_ss)
    
    def test_decode(self):
        """Test decoding back to sequence."""
        original = "MKFLILLFN"
        encoded = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(encoded.token_ids)
        
        self.assertEqual(decoded, original)
    
    def test_batch_encode(self):
        """Test batch encoding multiple sequences."""
        sequences = ["MKF", "ACDE", "GHI"]
        batch = self.tokenizer.batch_encode(sequences)
        
        self.assertEqual(batch['token_ids'].shape, (3, 64))
        self.assertEqual(batch['properties'].shape, (3, 64, 24))
        self.assertEqual(batch['attention_mask'].shape, (3, 64))
        self.assertEqual(batch['lengths'].tolist(), [5, 6, 5])  # includes <sos> and <eos>
    
    def test_mlm_targets_only_masks_amino_acids(self):
        """Test MLM only masks amino acid tokens, not special tokens."""
        import torch
        torch.manual_seed(42)
        
        encoded = self.tokenizer.encode("MKFLILLFN")
        masked, labels = self.tokenizer.create_mlm_targets(encoded.token_ids, mask_prob=0.5)
        
        # Check that special tokens (<sos>=27, <eos>=28, <pad>=0) are never masked
        for i, (orig, label) in enumerate(zip(encoded.token_ids, labels)):
            if orig.item() in [0, 27, 28]:  # special tokens
                self.assertEqual(label.item(), -100, 
                               f"Special token at {i} should not be masked")
    
    def test_properties_for_special_tokens(self):
        """Test that special tokens have zero properties."""
        encoded = self.tokenizer.encode("MK")
        
        # <sos> at position 0 should have zero properties
        sos_props = encoded.properties[0]
        self.assertTrue((sos_props == 0).all(), "SOS token should have zero properties")
        
        # <pad> tokens should have zero properties
        pad_props = encoded.properties[-1]
        self.assertTrue((pad_props == 0).all(), "PAD token should have zero properties")
        
        # Amino acids should have non-zero properties
        m_props = encoded.properties[1]  # M
        self.assertFalse((m_props == 0).all(), "M should have non-zero properties")


class TestEmbeddings(unittest.TestCase):
    """Test multi-channel embedding layer."""
    
    def setUp(self):
        import torch
        from gammafold.data.tokenizer import GammaFoldTokenizer
        from gammafold.models.embeddings import MultiChannelEmbedding
        
        self.torch = torch
        self.tokenizer = GammaFoldTokenizer(max_length=32)
        self.embedding = MultiChannelEmbedding(
            vocab_size=36,
            embed_dim=128,
            num_properties=24,
            use_structure=True,
            use_secondary_structure=True
        )
    
    def test_channel_dimensions(self):
        """Test channel dimension allocation."""
        dims = self.embedding.get_channel_dims()
        
        self.assertEqual(dims['total'], 128)
        self.assertGreater(dims['sequence'], 0)
        self.assertGreater(dims['properties'], 0)
        self.assertGreater(dims['structure'], 0)
        self.assertGreater(dims['secondary_structure'], 0)
        
        # Sum should equal total
        channel_sum = dims['sequence'] + dims['properties'] + dims['structure'] + dims['secondary_structure']
        self.assertEqual(channel_sum, 128)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        batch = self.tokenizer.batch_encode(["MKFL", "ACDE"])
        
        with self.torch.no_grad():
            output = self.embedding(
                token_ids=batch['token_ids'],
                properties=batch['properties'],
                ss_tokens=batch['ss_tokens']
            )
        
        self.assertEqual(output.shape, (2, 32, 128))
    
    def test_forward_pass_no_nan(self):
        """Test forward pass produces no NaN values."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN"])
        
        with self.torch.no_grad():
            output = self.embedding(
                token_ids=batch['token_ids'],
                properties=batch['properties']
            )
        
        self.assertFalse(self.torch.isnan(output).any(), "Output contains NaN")
    
    def test_padding_embeddings(self):
        """Test that padding positions have different embeddings than real tokens."""
        batch = self.tokenizer.batch_encode(["MK"])  # Short sequence
        
        with self.torch.no_grad():
            output = self.embedding(
                token_ids=batch['token_ids'],
                properties=batch['properties']
            )
        
        # Token position (M at index 1) should differ from padding (index 10+)
        m_embedding = output[0, 1]  # M
        pad_embedding = output[0, 10]  # padding
        
        self.assertFalse(
            self.torch.allclose(m_embedding, pad_embedding),
            "M embedding should differ from padding"
        )


if __name__ == '__main__':
    # Change to project directory
    os.chdir('/home/retro/Projects/gamma-fold')
    
    # Run tests with verbosity
    unittest.main(verbosity=2)
