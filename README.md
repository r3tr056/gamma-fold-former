
# GammaFold Former

**GammaFold Former** is a state-of-the-art, Transformer‑based protein sequence modeling framework designed for scientific research and real‑world applications. Leveraging advanced deep learning innovations—including mixed-precision training, dynamic learning rate scheduling, and robust data preprocessing—GammaFold Former enables accurate protein language modeling, structure prediction, and function annotation. The pipeline includes an automated dataset builder that retrieves, cleans, deduplicates, and merges protein sequence data from verified sources such as UniProt, PDB, and Pfam.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Building](#dataset-building)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Scientific Background](#scientific-background)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

**GammaFold Former** is a comprehensive framework for protein modeling that integrates advanced Transformer architectures with robust data collection and processing methods. It aims to democratize protein modeling by providing an end-to-end pipeline—from dataset curation using UniProt's REST API to production-ready training and inference routines—thereby facilitating rapid scientific discovery in fields such as protein structure prediction, drug design, and functional annotation.

---

## Features

- **Advanced Transformer Architecture:**  
  - Multi-layer Transformer encoder with positional encoding.
  - Production-ready design with state-of-the-art innovations (e.g., mixed precision training, Noam scheduler, gradient clipping).
  
- **Robust Dataset Builder:**  
  - Programmatic access to UniProt (and other trusted sources) using REST API and pagination.
  - Data cleaning, deduplication, and filtering to ensure high-quality protein sequences.
  
- **Efficient Training Pipeline:**  
  - Utilizes multi-worker DataLoader, AMP for mixed precision, and a Noam-style learning rate scheduler.
  - Logs training metrics to TensorBoard with periodic checkpointing.
  
- **Flexible Inference Module:**  
  - Greedy decoding for generating protein sequences.
  - Ready for extension to beam search or other decoding strategies.
  
- **Modular Design:**  
  - Easily extendable for classification, structure prediction, or integration with external data sources.

---

## Installation

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.8+
- [Biopython](https://biopython.org/)
- [Requests](https://docs.python-requests.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard) (optional, for logging)
- Other standard Python libraries (see `requirements.txt`)

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/gammafold-former.git
cd gammafold-former
pip install -r requirements.txt
```

---

## Usage

### Dataset Building

Use the dataset builder to compile a unified FASTA file from UniProt and local sources.

```bash
python dataset_builder.py \
  --uniprot_query "(reviewed:true) AND (organism_id:9606)" \
  --target_count 100000 \
  --local_files extra1.fasta extra2.fasta \
  --output merged_dataset.fasta \
  --min_length 50 \
  --max_length_filter 1000
```

This command retrieves 100,000 reviewed human protein sequences from UniProt, merges them with additional local FASTA files, cleans, deduplicates, and filters the data, then saves the result as `merged_dataset.fasta`.

### Training

Train the GammaFold Former model on the prepared dataset:

```bash
python collect_and_train.py \
  --uniprot_query "(reviewed:true) AND (organism_id:9606)" \
  --target_count 100000 \
  --local_files extra1.fasta extra2.fasta \
  --output merged_dataset.fasta \
  --train \
  --batch_size 64 \
  --max_length_dataset 256 \
  --vocab_size 24 \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 6 \
  --dropout 0.1 \
  --num_epochs 50 \
  --learning_rate 1e-4
```

This script will:
- Build the dataset if it is not already available.
- Train the Transformer-based protein model using advanced training routines (mixed precision, Noam scheduler, etc.).
- Save checkpoints and final model state.

### Inference

Once trained, use the inference module to generate new protein sequences or analyze the output for downstream tasks:

```python
# Example Python snippet for inference:
from part4_production import infer  # Greedy decoding function from production code

# Example tokenized input sequence: [SOS, token1, token2, ...]
example_sequence = [1, 5, 12, 8, 3]  # Replace with actual token IDs
generated_tokens = infer(model, example_sequence, max_new_tokens=50)
print("Generated sequence:", generated_tokens)
```

---

## Model Architecture

GammaFold Former is built around a standard "decoder-only" Transformer architecture:
- **Embedding Layer & Positional Encoding:**  
  Converts token IDs to embeddings and adds sinusoidal positional encodings.
- **Transformer Encoder Stack:**  
  A configurable number of encoder layers with multi-head self-attention and dropout.
- **Output Decoder:**  
  Projects the Transformer output to the vocabulary (or classification) space.

The model is optimized using a Noam-style learning rate scheduler, mixed precision training, and gradient clipping to ensure stable convergence.

---

## Scientific Background

Protein structure and function prediction has long been a cornerstone of computational biology. Traditional methods required manual curation and intensive laboratory work. GammaFold Former leverages deep learning innovations—specifically, Transformer-based language models—to learn rich representations directly from protein sequences. By integrating advanced dataset curation techniques, robust training pipelines, and state-of-the-art inference methods, GammaFold Former aims to bridge the gap between raw sequence data and actionable biological insights.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, improvements, or new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems*.
- UniProt Consortium. "UniProt: a worldwide hub of protein knowledge." *Nucleic Acids Research*, 47(D1), D506–D515.
- [Additional relevant literature and documentation as needed]

---

GammaFold Former strives to empower researchers with an open, transparent, and production-ready tool for protein sequence modeling. Happy modeling!
