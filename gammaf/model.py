
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.checkpoint import checkpoint

class RotaryPositionalEmbedding(nn.Module):
	"""Rotary Position Embedding (RoPE) for transformers."""
	def __init__(self, dim):
		super().__init__()
		self.dim = dim
		inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("inv_freq", inv_freq)

	def _rotate_half(self, x):
		x1, x2 = x.chunk(2, dim=-1)
		return torch.cat((-x2, x1), dim=-1)

	def forward(self, seq_len, device):
		t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
		freqs = torch.einsum('i,j->ij', t, self.inv_freq)
		emb = torch.cat((freqs, freqs), dim=-1)
		return emb[None, :, None, :]
	
	def apply_rotary_emb(self, x, pos_emb):
		return (x * pos_emb.cos()) + (self._rotate_half(x) * pos_emb.sin())
	
class RMSNorm(nn.Module):
	"""Root Mean Square Layer Normalization."""
	def __init__(self, dim, eps=1e-8):
		super().__init__()
		self.scale = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x):
		norm_x = torch.norm(x, dim=-1, keepdim=True) * (1 / math.sqrt(x.size(-1)))
		return self.scale * (x / (norm_x + self.eps))

class SwiGLUFFN(nn.Module):
	"""Feed-forward network with SwiGLU activation."""
	def __init__(self, dim, hidden_dim, dropout=0.1):
		super().__init__()
		self.w1 = nn.Linear(dim, hidden_dim)
		self.w2 = nn.Linear(dim, hidden_dim)
		self.w3 = nn.Linear(hidden_dim, dim)
		self.dropout = nn.Dropout(dropout)
		self.activation = nn.SiLU()

	def forward(self, x):
		return self.w3(self.dropout(self.activation(self.w1(x)) * self.w2(x)))

class FlashAttention(nn.Module):
	"""Memory-efficient multi-head attention with RoPE."""
	def __init__(self, embed_dim, num_heads, dropout=0.1):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.rope = RotaryPositionalEmbedding(self.head_dim)
		
		self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		self.dropout = dropout

	def forward(self, x, attention_mask=None):
		batch_size, seq_len, _ = x.shape
		qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
		q, k, v = qkv.unbind(2)
		
		# Apply RoPE
		pos_emb = self.rope(seq_len, x.device)
		q = self.rope.apply_rotary_emb(q, pos_emb)
		k = self.rope.apply_rotary_emb(k, pos_emb)
		
		# Flash attention via PyTorch 2.0's optimized function
		x = F.scaled_dot_product_attention(
			q.permute(0, 2, 1, 3),   # [bs, heads, seq_len, dim]
			k.permute(0, 2, 1, 3), 
			v.permute(0, 2, 1, 3),
			dropout_p=self.dropout if self.training else 0,
			attn_mask=attention_mask
		).transpose(1, 2).reshape(batch_size, seq_len, -1)
		
		return self.out_proj(x)

class TransformerBlock(nn.Module):
	"""Transformer block with stochastic depth and checkpointing."""
	def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
		super().__init__()
		self.norm1 = RMSNorm(embed_dim)
		self.attn = FlashAttention(embed_dim, num_heads, dropout)
		self.norm2 = RMSNorm(embed_dim)
		self.ffn = SwiGLUFFN(embed_dim, ff_dim, dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, attention_mask=None):
		# Stochastic depth and checkpointing
		if self.training and torch.rand(1) < 0.1:  # 10% stochastic depth probability
			return x
		
		# Self-attention with residual
		attn_out = checkpoint(self._attn_block, x, attention_mask)
		x = x + self.dropout(attn_out)
		
		# FFN with residual
		ffn_out = checkpoint(self._ffn_block, x)
		return x + self.dropout(ffn_out)

	def _attn_block(self, x, attention_mask):
		return self.attn(self.norm1(x), attention_mask)

	def _ffn_block(self, x):
		return self.ffn(self.norm2(x))

class GammaFoldFormer(nn.Module):
	"""GammaFold Former Protein Transformer."""
	def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=12, 
				 ff_dim=1024, dropout=0.1, max_len=1024, num_classes=None):
		super().__init__()
		self.embed_dim = embed_dim
		self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
		self.embed_dropout = nn.Dropout(dropout)
		self.scale = math.sqrt(embed_dim)
		
		self.blocks = nn.ModuleList([
			TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
			for _ in range(num_layers)
		])
		
		self.final_norm = RMSNorm(embed_dim)
		self.head = nn.Linear(embed_dim, num_classes if num_classes else vocab_size)
		
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.xavier_normal_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0, std=0.02)

	def forward(self, x, attention_mask=None):
		x = self.token_emb(x) * self.scale
		x = self.embed_dropout(x)
		
		for block in self.blocks:
			x = block(x, attention_mask)
			
		x = self.final_norm(x)
		return self.head(x)

	def configure_sharded_model(self):
		"""Configure for FSDP (multi-GPU training)."""
		from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
		return FSDP(self)