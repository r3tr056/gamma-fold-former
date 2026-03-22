# GammaFold Framework Architecture

> **GammaFold** is a comprehensive protein modeling framework built around the GammaFold Former transformer architecture. It provides end-to-end tools for protein sequence modeling, structure prediction, and dataset management.

---

## Table of Contents

- [Vision & Goals](#vision--goals)
- [Framework Overview](#framework-overview)
- [Core Components](#core-components)
  - [GammaFold Former Model](#1-gammafold-former-model)
  - [Inference Server](#2-inference-server)
  - [Sequence Modeling Tools](#3-sequence-modeling-tools)
  - [Structure Prediction Engine](#4-structure-prediction-engine)
  - [Dataset Utilities](#5-dataset-utilities)
  - [User Interface](#6-user-interface)
- [Directory Structure](#directory-structure)
- [Implementation Roadmap](#implementation-roadmap)
- [API Reference](#api-reference)

---

## Vision & Goals

### Mission
Democratize protein modeling by providing a production-ready, open-source framework that enables researchers and developers to:
- Model and generate protein sequences
- Predict 3D protein structures from amino acid sequences
- Work with standardized protein datasets from trusted sources

### End Goals & Deliverables

| Deliverable | Description | Status |
|-------------|-------------|--------|
| GammaFold Former S/M/L | Transformer models in 3 size variants | 🔧 In Progress |
| Inference Server | REST/gRPC API for model serving | ❌ Not Started |
| Sequence Generation Tool | Generate novel protein sequences | ❌ Not Started |
| Structure Prediction Tool | Predict 3D coordinates from sequence | 🔧 In Progress |
| Dataset Import/Export | UniProt, PDB, AlphaFold DB, Pfam support | 🔧 In Progress |
| Material UI Dashboard | Web interface for all framework features | ❌ Not Started |
| CLI Tools | Command-line utilities for all operations | ❌ Not Started |
| Python SDK | Programmatic access to all features | 🔧 In Progress |

---

## Framework Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GammaFold Framework                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    User Interface Layer                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Material UI │  │    CLI      │  │      Python SDK         │  │   │
│  │  │  Dashboard  │  │   Tools     │  │  (gammafold package)    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    Inference Server                              │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  REST API   │  │  gRPC API   │  │   WebSocket Streaming   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                    Core Engine Layer                             │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │   │
│  │  │ Sequence Modeler │  │Structure Predictor│  │ Dataset Utils │  │   │
│  │  │  - Generation    │  │  - Coordinate     │  │  - Import     │  │   │
│  │  │  - Prediction    │  │    Prediction     │  │  - Export     │  │   │
│  │  │  - Embedding     │  │  - PDB Export     │  │  - Transform  │  │   │
│  │  └──────────────────┘  └──────────────────┘  └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌─────────────────────────────────▼───────────────────────────────┐   │
│  │                 GammaFold Former Models                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │   Small     │  │   Medium    │  │         Large           │  │   │
│  │  │  (30M)      │  │   (150M)    │  │        (650M)           │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. GammaFold Former Model

The core transformer architecture for protein sequence and structure modeling.

#### Model Variants

| Variant | Parameters | Layers | Heads | Embed Dim | FF Dim | Max Length | Use Case |
|---------|------------|--------|-------|-----------|--------|------------|----------|
| **Small** | ~30M | 6 | 8 | 256 | 1024 | 1024 | Development, quick inference |
| **Medium** | ~150M | 12 | 12 | 512 | 2048 | 2048 | Balanced performance |
| **Large** | ~650M | 24 | 16 | 1024 | 4096 | 4096 | Maximum accuracy |

#### Architecture Components

```python
GammaFoldFormer
├── TokenEmbedding          # Amino acid to vector embedding
├── RotaryPositionalEmb     # RoPE for position encoding
├── TransformerBlocks[]     # N transformer layers
│   ├── RMSNorm             # Pre-normalization
│   ├── FlashAttention      # Memory-efficient self-attention
│   │   └── RoPE            # Rotary position embeddings
│   ├── SwiGLUFFN           # Gated feed-forward network
│   └── StochasticDepth     # Training regularization
├── FinalNorm               # Output normalization
├── SequenceHead            # Token prediction logits
└── StructureHead           # 3D coordinate prediction
```

#### Implementation Location
```
gammafold/
├── models/
│   ├── __init__.py
│   ├── config.py           # Model configuration dataclasses
│   ├── former.py           # GammaFoldFormer main class
│   ├── attention.py        # FlashAttention with RoPE
│   ├── ffn.py              # SwiGLU feed-forward
│   ├── normalization.py    # RMSNorm implementation
│   └── embeddings.py       # Token + positional embeddings
```

#### API

```python
from gammafold.models import GammaFoldFormer, ModelConfig

# Load pretrained model
model = GammaFoldFormer.from_pretrained("gammafold-medium")

# Or create custom configuration
config = ModelConfig(
    variant="medium",
    vocab_size=21,
    embed_dim=512,
    num_heads=12,
    num_layers=12,
    ff_dim=2048,
    max_len=2048,
    dropout=0.1
)
model = GammaFoldFormer(config)

# Forward pass
token_logits, coordinates = model(sequence_tokens)
```

---

### 2. Inference Server

Production-ready server for model deployment with multiple API protocols.

#### Features
- REST API for simple HTTP requests
- gRPC API for high-performance RPC
- WebSocket for streaming predictions
- Model hot-reloading
- Request batching for efficiency
- GPU memory management
- Health checks and metrics

#### Implementation Location
```
gammafold/
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   ├── grpc_server.py      # gRPC service implementation
│   ├── websocket.py        # WebSocket handlers
│   ├── model_manager.py    # Model loading and caching
│   ├── batching.py         # Request batching logic
│   └── config.py           # Server configuration
```

#### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health status |
| `/models` | GET | List available models |
| `/predict/sequence` | POST | Generate/complete protein sequence |
| `/predict/structure` | POST | Predict 3D coordinates |
| `/predict/stream` | POST | Stream structure prediction |
| `/embed` | POST | Get sequence embeddings |

#### Example Request

```bash
# Predict protein structure
curl -X POST http://localhost:8000/predict/structure \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAADNH...", "model": "gammafold-medium"}'
```

#### Response Format

```json
{
  "sequence": "MKFLILLFN...",
  "length": 256,
  "coordinates": [
    {"residue": 1, "aa": "M", "ca": [0.0, 0.0, 0.0], "confidence": 0.95},
    {"residue": 2, "aa": "K", "ca": [3.8, 0.2, 0.1], "confidence": 0.93}
  ],
  "pdb_string": "ATOM      1  CA  MET A   1       0.000   0.000   0.000...",
  "plddt_score": 87.5,
  "inference_time_ms": 234
}
```

---

### 3. Sequence Modeling Tools

Tools for protein sequence generation, prediction, and analysis.

#### Features

| Tool | Description |
|------|-------------|
| **Sequence Generation** | Generate novel protein sequences from scratch or prompts |
| **Sequence Completion** | Complete partial sequences (fill masked regions) |
| **Next Token Prediction** | Predict most likely next amino acid |
| **Sequence Embedding** | Extract learned representations for downstream tasks |
| **Sequence Scoring** | Compute likelihood/perplexity of sequences |

#### Implementation Location
```
gammafold/
├── sequence/
│   ├── __init__.py
│   ├── generator.py        # Sequence generation with sampling
│   ├── completer.py        # Masked sequence completion
│   ├── embedder.py         # Extract sequence embeddings
│   ├── scorer.py           # Sequence likelihood scoring
│   └── sampler.py          # Sampling strategies (greedy, beam, nucleus)
```

#### API

```python
from gammafold.sequence import SequenceGenerator, SamplingConfig

# Initialize generator
generator = SequenceGenerator.from_pretrained("gammafold-medium")

# Generate novel sequence
config = SamplingConfig(
    max_length=256,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
sequence = generator.generate(config)

# Complete masked sequence
partial = "MKFLILLFN<mask><mask><mask>LPVLAADNH"
completed = generator.complete(partial, num_predictions=5)

# Get embeddings
embeddings = generator.embed("MKFLILLFNILCLFPVLAADNH")  # Shape: [seq_len, embed_dim]
```

---

### 4. Structure Prediction Engine

Continuous 3D coordinate prediction for protein backbone atoms.

#### Features

| Feature | Description |
|---------|-------------|
| **Backbone Prediction** | Predict Cα (alpha carbon) coordinates |
| **Full Atom Prediction** | Predict N, Cα, C, O coordinates (future) |
| **Confidence Scores** | Per-residue pLDDT-like confidence |
| **PDB Export** | Save predictions as PDB/mmCIF files |
| **Visualization** | Built-in 3D visualization tools |

#### Output Format

For each residue, the structure head predicts:
- **Cα coordinates**: (x, y, z) in Ångströms
- **Confidence score**: 0-100 pLDDT-like metric

#### Implementation Location
```
gammafold/
├── structure/
│   ├── __init__.py
│   ├── predictor.py        # Main structure prediction class
│   ├── coordinates.py      # Coordinate processing utilities
│   ├── refinement.py       # Post-prediction refinement
│   ├── confidence.py       # Confidence estimation
│   └── export/
│       ├── pdb.py          # PDB file writer
│       ├── mmcif.py        # mmCIF file writer
│       └── visualization.py # 3D visualization (PyMOL, py3Dmol)
```

#### API

```python
from gammafold.structure import StructurePredictor

# Initialize predictor
predictor = StructurePredictor.from_pretrained("gammafold-large")

# Predict structure
result = predictor.predict("MKFLILLFNILCLFPVLAADNH")

# Access results
print(result.coordinates)      # Shape: [seq_len, 3]
print(result.confidence)       # Shape: [seq_len]
print(result.mean_plddt)       # Scalar

# Export to PDB
result.to_pdb("prediction.pdb")

# Stream continuous prediction for long sequences
for chunk in predictor.predict_stream(long_sequence, chunk_size=256):
    print(f"Predicted residues {chunk.start}-{chunk.end}")
```

---

### 5. Dataset Utilities

Comprehensive tools for importing, exporting, and transforming protein data.

#### Supported Data Sources (Import)

| Source | Format | Description |
|--------|--------|-------------|
| **UniProt** | FASTA, XML | Swiss-Prot and TrEMBL sequences |
| **PDB** | PDB, mmCIF | Experimental structures |
| **AlphaFold DB** | mmCIF | Predicted structures |
| **Pfam** | Stockholm, FASTA | Protein families |
| **ProteinNet** | Custom | Training/validation splits |
| **ESMAtlas** | FASTA | Metagenomic sequences |

#### Supported Export Formats

| Format | Use Case |
|--------|----------|
| **FASTA** | Sequence-only datasets |
| **PDB** | Single structure files |
| **mmCIF** | Modern structure format |
| **Parquet** | Efficient columnar storage |
| **HuggingFace Datasets** | ML-ready format |
| **TFRecord** | TensorFlow training |

#### Implementation Location
```
gammafold/
├── data/
│   ├── __init__.py
│   ├── sources/
│   │   ├── uniprot.py      # UniProt REST API client
│   │   ├── pdb.py          # RCSB PDB client
│   │   ├── alphafold_db.py # AlphaFold Database client
│   │   ├── pfam.py         # Pfam database client
│   │   └── proteinnet.py   # ProteinNet parser
│   ├── exporters/
│   │   ├── fasta.py        # FASTA writer
│   │   ├── pdb.py          # PDB writer
│   │   ├── parquet.py      # Parquet writer
│   │   └── huggingface.py  # HF Datasets format
│   ├── transforms/
│   │   ├── tokenizer.py    # Sequence tokenization
│   │   ├── filter.py       # Length/quality filtering
│   │   ├── cluster.py      # Sequence clustering (MMseqs2)
│   │   └── augment.py      # Data augmentation
│   └── dataset.py          # PyTorch Dataset classes
```

#### API

```python
from gammafold.data import DatasetBuilder, UniProtSource, PDBSource

# Build dataset from multiple sources
builder = DatasetBuilder()

# Add sequences from UniProt
builder.add_source(
    UniProtSource(
        query="(reviewed:true) AND (organism_id:9606)",  # Human proteins
        max_sequences=100000
    )
)

# Add structures from PDB
builder.add_source(
    PDBSource(
        resolution_max=2.5,  # High quality structures only
        method=["X-RAY DIFFRACTION", "CRYO-EM"]
    )
)

# Apply transforms
builder.filter(min_length=50, max_length=1024)
builder.deduplicate(identity_threshold=90)
builder.cluster(identity_threshold=30)  # Reduce redundancy

# Build and export
dataset = builder.build()
dataset.to_parquet("protein_dataset.parquet")
dataset.to_fasta("sequences.fasta")

# Create PyTorch DataLoader
from gammafold.data import ProteinDataset
train_dataset = ProteinDataset("protein_dataset.parquet", split="train")
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

### 6. User Interface

Material Design web dashboard for accessing all framework features.

#### Features

| Feature | Description |
|---------|-------------|
| **Sequence Input** | Paste or upload protein sequences |
| **Structure Viewer** | Interactive 3D visualization (NGL/Mol*) |
| **Job Management** | Submit, track, and download predictions |
| **Dataset Browser** | Explore and download datasets |
| **Model Comparison** | Compare S/M/L model predictions |
| **API Playground** | Test API endpoints interactively |

#### Technology Stack

- **Frontend**: React + Material UI (MUI)
- **3D Viewer**: Mol* or NGL Viewer
- **State Management**: React Query / Zustand
- **API Client**: Generated from OpenAPI spec

#### Implementation Location
```
gammafold/
├── ui/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── SequenceInput/
│   │   │   ├── StructureViewer/
│   │   │   ├── JobTracker/
│   │   │   ├── DatasetBrowser/
│   │   │   └── Navigation/
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Predict.tsx
│   │   │   ├── Datasets.tsx
│   │   │   └── Api.tsx
│   │   ├── api/
│   │   │   └── client.ts   # Auto-generated API client
│   │   └── hooks/
│   └── public/
```

#### Wireframe

```
┌────────────────────────────────────────────────────────────────────────┐
│  🧬 GammaFold                              [Predict] [Datasets] [API]  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────┐  ┌────────────────────────────────┐  │
│  │ Sequence Input              │  │ 3D Structure Viewer            │  │
│  │ ┌─────────────────────────┐ │  │                                │  │
│  │ │ MKFLILLFNILCLFPVLAADNH  │ │  │       [Interactive 3D]         │  │
│  │ │ ...                     │ │  │                                │  │
│  │ └─────────────────────────┘ │  │     🔄 Rotate  🔍 Zoom         │  │
│  │                             │  │                                │  │
│  │ Model: [Small ▼]            │  │  pLDDT: 87.5  Length: 256      │  │
│  │                             │  │                                │  │
│  │ [▶ Predict Structure]       │  │  [Download PDB] [Copy FASTA]   │  │
│  └─────────────────────────────┘  └────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ Confidence Plot                                                  │  │
│  │  100 ┤ ████████████████  ████████  ██████████████████████████   │  │
│  │   75 ┤                 ██        ██                              │  │
│  │   50 ┤                                                           │  │
│  │    0 ┴─────────────────────────────────────────────────────────  │  │
│  │         1        50       100       150       200       256      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
gamma-fold/
├── docs/
│   ├── ARCHITECTURE.md         # This document
│   ├── API.md                  # API reference
│   ├── TRAINING.md             # Training guide
│   └── DEPLOYMENT.md           # Deployment guide
│
├── gammafold/                  # Main Python package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── models/                 # GammaFold Former models
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── former.py
│   │   ├── attention.py
│   │   ├── ffn.py
│   │   ├── normalization.py
│   │   └── embeddings.py
│   │
│   ├── sequence/               # Sequence modeling tools
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── completer.py
│   │   ├── embedder.py
│   │   ├── scorer.py
│   │   └── sampler.py
│   │
│   ├── structure/              # Structure prediction
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── coordinates.py
│   │   ├── refinement.py
│   │   ├── confidence.py
│   │   └── export/
│   │       ├── pdb.py
│   │       ├── mmcif.py
│   │       └── visualization.py
│   │
│   ├── data/                   # Dataset utilities
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── sources/
│   │   │   ├── uniprot.py
│   │   │   ├── pdb.py
│   │   │   ├── alphafold_db.py
│   │   │   └── pfam.py
│   │   ├── exporters/
│   │   │   ├── fasta.py
│   │   │   ├── pdb.py
│   │   │   └── parquet.py
│   │   └── transforms/
│   │       ├── tokenizer.py
│   │       ├── filter.py
│   │       └── cluster.py
│   │
│   ├── server/                 # Inference server
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── grpc_server.py
│   │   ├── websocket.py
│   │   └── config.py
│   │
│   ├── training/               # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── schedulers.py
│   │   └── callbacks.py
│   │
│   └── cli/                    # Command-line tools
│       ├── __init__.py
│       ├── main.py
│       ├── predict.py
│       ├── train.py
│       └── data.py
│
├── ui/                         # Material Design frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── pages/
│   └── public/
│
├── scripts/                    # Development scripts
│   ├── train.sh
│   ├── evaluate.sh
│   └── download_checkpoints.sh
│
├── tests/                      # Test suite
│   ├── test_models.py
│   ├── test_sequence.py
│   ├── test_structure.py
│   ├── test_data.py
│   └── test_server.py
│
├── checkpoints/                # Model weights (gitignored)
├── pyproject.toml              # Python package config
├── requirements.txt            # Dependencies
└── README.md                   # Project overview
```

---

## Implementation Roadmap

### Phase 1: Core Foundation (Weeks 1-4)
- [ ] Refactor model code into clean `gammafold/models/` structure
- [ ] Fix existing bugs (typos, imports)
- [ ] Implement model variants (S, M, L) with config system
- [ ] Create comprehensive test suite
- [ ] Set up CI/CD pipeline

### Phase 2: Data Pipeline (Weeks 5-8)
- [ ] Implement UniProt client with proper parsing
- [ ] Implement PDB client with structure extraction
- [ ] Create real coordinate parsing (not fake random data)
- [ ] Build dataset transforms (filtering, clustering)
- [ ] Create PyTorch Dataset classes with proper batching

### Phase 3: Inference Tools (Weeks 9-12)
- [ ] Sequence generation with multiple sampling strategies
- [ ] Structure prediction with confidence scores
- [ ] PDB/mmCIF export functionality
- [ ] Inference optimization (TorchScript, ONNX)

### Phase 4: Server & API (Weeks 13-16)
- [ ] FastAPI REST server implementation
- [ ] gRPC service definition and implementation
- [ ] WebSocket streaming for long predictions
- [ ] Model management and hot-reloading
- [ ] Request batching and GPU management

### Phase 5: User Interface (Weeks 17-20)
- [ ] React + Material UI setup
- [ ] Sequence input and validation
- [ ] 3D structure viewer integration (Mol*)
- [ ] Job submission and tracking
- [ ] API playground

### Phase 6: Polish & Release (Weeks 21-24)
- [ ] Documentation completion
- [ ] Performance optimization
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Model zoo with pretrained weights

---

## API Reference

### CLI Commands

```bash
# Predict structure
gammafold predict structure --sequence "MKFLILLFN..." --model medium --output result.pdb

# Generate sequence
gammafold predict sequence --length 256 --temperature 0.8 --output generated.fasta

# Download dataset
gammafold data download --source uniprot --query "reviewed:true" --output dataset.fasta

# Train model
gammafold train --config config.yaml --data dataset.parquet --output checkpoints/

# Start server
gammafold serve --model medium --port 8000 --workers 4
```

### Python Package

```python
import gammafold

# Quick prediction
structure = gammafold.predict_structure("MKFLILLFNILCLFPVLAADNH")
structure.to_pdb("output.pdb")

# Quick generation
sequence = gammafold.generate_sequence(length=256)

# Access models directly
model = gammafold.GammaFoldFormer.from_pretrained("gammafold-medium")
```

---

## Dependencies

### Core
- Python >= 3.10
- PyTorch >= 2.0
- Transformers (for tokenizers)
- BioPython (for FASTA/PDB parsing)

### Server
- FastAPI
- Uvicorn
- gRPCio
- Pydantic

### Data
- Requests
- Pandas
- PyArrow (Parquet)
- MMseqs2 (clustering)

### UI
- React 18+
- Material UI (MUI) 5+
- Mol* Viewer
- React Query

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

---

## References

1. Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
2. Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." Nature.
3. Lin, Z. et al. (2022). "Language models of protein sequences at the scale of evolution." bioRxiv.
4. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv.
