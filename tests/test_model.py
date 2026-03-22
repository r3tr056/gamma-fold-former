"""
Unit tests for GammaFold model architecture.
"""

import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRotaryPositionEmbedding(unittest.TestCase):
    """Test RoPE implementation."""
    
    def setUp(self):
        from gammafold.models.attention import RotaryPositionEmbedding
        self.rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
    
    def test_rope_output_shape(self):
        """Test RoPE preserves shape."""
        q = torch.randn(2, 32, 8, 64)
        k = torch.randn(2, 32, 8, 64)
        q_rot, k_rot = self.rope(q, k)
        
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
    
    def test_rope_modifies_values(self):
        """Test RoPE actually rotates vectors."""
        q = torch.randn(2, 32, 8, 64)
        k = torch.randn(2, 32, 8, 64)
        q_rot, k_rot = self.rope(q, k)
        
        self.assertFalse(torch.allclose(q, q_rot))
        self.assertFalse(torch.allclose(k, k_rot))
    
    def test_rope_different_positions(self):
        """Test different positions get different rotations."""
        q = torch.ones(1, 4, 1, 64)  # Same value at all positions
        k = torch.ones(1, 4, 1, 64)
        q_rot, k_rot = self.rope(q, k)
        
        # Different positions should have different outputs
        self.assertFalse(torch.allclose(q_rot[0, 0], q_rot[0, 1]))


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention."""
    
    def setUp(self):
        from gammafold.models.attention import MultiHeadAttention
        self.attn = MultiHeadAttention(embed_dim=256, num_heads=4, use_rope=True)
    
    def test_attention_output_shape(self):
        """Test attention output shape."""
        x = torch.randn(2, 32, 256)
        output, _ = self.attn(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        x = torch.randn(2, 32, 256)
        mask = torch.ones(2, 32)
        mask[:, 16:] = 0  # Mask out second half
        
        output, _ = self.attn(x, attention_mask=mask)
        self.assertEqual(output.shape, x.shape)
    
    def test_attention_return_weights(self):
        """Test returning attention weights."""
        x = torch.randn(2, 32, 256)
        output, attn_weights = self.attn(x, return_attention=True)
        
        self.assertEqual(attn_weights.shape, (2, 4, 32, 32))


class TestFeedForward(unittest.TestCase):
    """Test feed-forward network."""
    
    def setUp(self):
        from gammafold.models.attention import FeedForward
        self.ffn = FeedForward(embed_dim=256, hidden_dim=1024)
    
    def test_ffn_output_shape(self):
        """Test FFN output shape."""
        x = torch.randn(2, 32, 256)
        output = self.ffn(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_ffn_no_nan(self):
        """Test FFN produces no NaN."""
        x = torch.randn(2, 32, 256)
        output = self.ffn(x)
        self.assertFalse(torch.isnan(output).any())


class TestTransformerBlock(unittest.TestCase):
    """Test transformer block."""
    
    def setUp(self):
        from gammafold.models.attention import TransformerBlock
        self.block = TransformerBlock(embed_dim=256, num_heads=4)
    
    def test_block_output_shape(self):
        """Test block output shape."""
        x = torch.randn(2, 32, 256)
        output, _ = self.block(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_block_residual(self):
        """Test block has residual connection."""
        x = torch.randn(2, 32, 256)
        output, _ = self.block(x)
        
        # Output should be different from input (transformation applied)
        self.assertFalse(torch.allclose(x, output))
        
        # But correlation should exist due to residual
        correlation = (x * output).mean()
        self.assertGreater(correlation.abs(), 0.1)


class TestGammaFoldConfig(unittest.TestCase):
    """Test model configurations."""
    
    def test_config_small(self):
        from gammafold.models.gamma_fold import GammaFoldConfig
        config = GammaFoldConfig.small()
        
        self.assertEqual(config.embed_dim, 320)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 8)
    
    def test_config_medium(self):
        from gammafold.models.gamma_fold import GammaFoldConfig
        config = GammaFoldConfig.medium()
        
        self.assertEqual(config.embed_dim, 640)
        self.assertEqual(config.num_layers, 12)
    
    def test_config_large(self):
        from gammafold.models.gamma_fold import GammaFoldConfig
        config = GammaFoldConfig.large()
        
        self.assertEqual(config.embed_dim, 1280)
        self.assertEqual(config.num_layers, 24)


class TestGammaFoldFormer(unittest.TestCase):
    """Test main model."""
    
    def setUp(self):
        from gammafold.models.gamma_fold import GammaFoldFormer, GammaFoldConfig
        from gammafold.data.tokenizer import GammaFoldTokenizer
        
        self.config = GammaFoldConfig.small()
        self.config.max_seq_len = 64
        self.model = GammaFoldFormer(self.config)
        self.tokenizer = GammaFoldTokenizer(max_length=64)
    
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN", "ACDEFGHIK"])
        
        with torch.no_grad():
            outputs = self.model(
                token_ids=batch['token_ids'],
                properties=batch['properties'],
                attention_mask=batch['attention_mask']
            )
        
        self.assertEqual(outputs['last_hidden_state'].shape, (2, 64, 320))
        self.assertEqual(outputs['mlm_logits'].shape, (2, 64, 36))
        self.assertEqual(outputs['coord_pred'].shape, (2, 64, 3))
    
    def test_mlm_loss(self):
        """Test MLM loss computation."""
        # Use longer sequence and fixed seed for reproducibility
        torch.manual_seed(42)
        batch = self.tokenizer.batch_encode(["MKFLILLFNACDEFGHIKLMNPQ"])
        masked_ids, labels = self.tokenizer.create_mlm_targets(
            batch['token_ids'], mask_prob=0.3  # Higher mask prob
        )
        
        # Skip test if no tokens were masked (edge case)
        if (labels == -100).all():
            self.skipTest("No tokens masked in this random run")
        
        outputs = self.model(
            token_ids=masked_ids,
            properties=batch['properties'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        
        self.assertIsNotNone(outputs['mlm_loss'])
        self.assertFalse(torch.isnan(outputs['mlm_loss']))
        self.assertGreater(outputs['mlm_loss'].item(), 0)
    
    def test_coord_loss(self):
        """Test coordinate loss computation."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN"])
        coords = torch.randn(1, 64, 3)
        coord_mask = batch['attention_mask'].float()
        
        outputs = self.model(
            token_ids=batch['token_ids'],
            properties=batch['properties'],
            attention_mask=batch['attention_mask'],
            coords=coords,
            coord_mask=coord_mask
        )
        
        self.assertIsNotNone(outputs['coord_loss'])
        self.assertGreater(outputs['coord_loss'].item(), 0)
    
    def test_backward_pass(self):
        """Test gradients flow correctly."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN"])
        masked_ids, labels = self.tokenizer.create_mlm_targets(batch['token_ids'])
        
        self.model.train()
        outputs = self.model(
            token_ids=masked_ids,
            properties=batch['properties'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        
        outputs['loss'].backward()
        
        # Check gradients exist
        params_with_grad = sum(1 for p in self.model.parameters() if p.grad is not None)
        self.assertGreater(params_with_grad, 0)
    
    def test_predict_structure(self):
        """Test structure prediction convenience method."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN"])
        
        coords = self.model.predict_structure(
            token_ids=batch['token_ids'],
            properties=batch['properties'],
            attention_mask=batch['attention_mask']
        )
        
        self.assertEqual(coords.shape, (1, 64, 3))
    
    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        num_params = self.model.num_parameters_millions
        
        # Small model should be around 10M
        self.assertGreater(num_params, 5)
        self.assertLess(num_params, 20)


class TestOutputHeads(unittest.TestCase):
    """Test output prediction heads."""
    
    def test_mlm_head(self):
        from gammafold.models.gamma_fold import MLMHead
        
        head = MLMHead(embed_dim=256, vocab_size=36)
        x = torch.randn(2, 32, 256)
        logits = head(x)
        
        self.assertEqual(logits.shape, (2, 32, 36))
    
    def test_structure_head(self):
        from gammafold.models.gamma_fold import StructurePredictionHead
        
        head = StructurePredictionHead(embed_dim=256, hidden_dim=128)
        x = torch.randn(2, 32, 256)
        coords = head(x)
        
        self.assertEqual(coords.shape, (2, 32, 3))
    
    def test_distance_head(self):
        from gammafold.models.gamma_fold import DistanceMatrixHead
        
        head = DistanceMatrixHead(embed_dim=256, num_bins=64)
        x = torch.randn(2, 32, 256)
        logits = head(x)
        
        self.assertEqual(logits.shape, (2, 32, 32, 64))


class TestCrossModalAttention(unittest.TestCase):
    """Test cross-modal attention components."""
    
    def setUp(self):
        from gammafold.models.cross_attention import CrossModalAttention
        self.cross_attn = CrossModalAttention(query_dim=256, kv_dim=128, num_heads=8)
    
    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        query = torch.randn(2, 32, 256)
        kv = torch.randn(2, 32, 128)
        
        output, _ = self.cross_attn(query, kv)
        self.assertEqual(output.shape, query.shape)
    
    def test_cross_attention_weights(self):
        """Test attention weights shape."""
        query = torch.randn(2, 32, 256)
        kv = torch.randn(2, 32, 128)
        
        output, attn_weights = self.cross_attn(query, kv, return_attention=True)
        self.assertEqual(attn_weights.shape, (2, 8, 32, 32))


class TestMultiStreamEmbedding(unittest.TestCase):
    """Test multi-stream embedding layer."""
    
    def setUp(self):
        from gammafold.models.embeddings import MultiStreamEmbedding
        self.embed = MultiStreamEmbedding(
            vocab_size=36, seq_dim=256, prop_dim=96, struct_dim=128, ss_dim=48
        )
    
    def test_output_streams(self):
        """Test that all streams are returned."""
        token_ids = torch.randint(0, 36, (2, 32))
        properties = torch.randn(2, 32, 24)
        
        seq, prop, struct, ss = self.embed(token_ids, properties)
        
        self.assertEqual(seq.shape, (2, 32, 256))
        self.assertEqual(prop.shape, (2, 32, 96))
        self.assertEqual(struct.shape, (2, 32, 128))
        self.assertEqual(ss.shape, (2, 32, 48))


class TestCrossModalFusionBlock(unittest.TestCase):
    """Test cross-modal fusion block."""
    
    def setUp(self):
        from gammafold.models.cross_attention import CrossModalFusionBlock
        self.block = CrossModalFusionBlock(
            seq_dim=256, prop_dim=96, struct_dim=128, ss_dim=48, num_heads=8
        )
    
    def test_fusion_output(self):
        """Test fusion block processes all streams."""
        seq = torch.randn(2, 32, 256)
        prop = torch.randn(2, 32, 96)
        struct = torch.randn(2, 32, 128)
        ss = torch.randn(2, 32, 48)
        mask = torch.ones(2, 32)
        
        seq_out, prop_out, struct_out, ss_out = self.block(seq, prop, struct, ss, mask)
        
        self.assertEqual(seq_out.shape, seq.shape)
        self.assertEqual(prop_out.shape, prop.shape)
        self.assertEqual(struct_out.shape, struct.shape)
        self.assertEqual(ss_out.shape, ss.shape)


class TestGammaFoldMultiModal(unittest.TestCase):
    """Test multi-modal cross-attention model."""
    
    def setUp(self):
        from gammafold.models.gamma_fold import GammaFoldMultiModal, MultiModalConfig
        from gammafold.data.tokenizer import GammaFoldTokenizer
        
        self.config = MultiModalConfig.small()
        self.config.max_seq_len = 64
        self.model = GammaFoldMultiModal(self.config)
        self.tokenizer = GammaFoldTokenizer(max_length=64)
    
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch = self.tokenizer.batch_encode(["MKFLILLFN", "ACDEFGHIK"])
        
        with torch.no_grad():
            outputs = self.model(
                token_ids=batch['token_ids'],
                properties=batch['properties'],
                attention_mask=batch['attention_mask']
            )
        
        self.assertEqual(outputs['last_hidden_state'].shape, (2, 64, 384))
        self.assertEqual(outputs['mlm_logits'].shape, (2, 64, 36))
        self.assertEqual(outputs['coord_pred'].shape, (2, 64, 3))
    
    def test_training_step(self):
        """Test backward pass with gradients."""
        torch.manual_seed(42)
        batch = self.tokenizer.batch_encode(["MKFLILLFNACDEFGHIK"])
        masked_ids, labels = self.tokenizer.create_mlm_targets(
            batch['token_ids'], mask_prob=0.3
        )
        
        self.model.train()
        outputs = self.model(
            token_ids=masked_ids,
            properties=batch['properties'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        
        if outputs['loss'] is not None:
            outputs['loss'].backward()
            params_with_grad = sum(1 for p in self.model.parameters() if p.grad is not None)
            self.assertGreater(params_with_grad, 0)
    
    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        num_params = self.model.num_parameters_millions
        # Small multi-modal should be around 17M
        self.assertGreater(num_params, 10)
        self.assertLess(num_params, 30)


if __name__ == '__main__':
    os.chdir('/home/retro/Projects/gamma-fold')
    unittest.main(verbosity=2)
