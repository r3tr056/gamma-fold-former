
import argparse, logging, os, re, sys, time
from io import StringIO
import requests
from requests.adapters import HTTPAdapter, Retry
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


LOG_DIR = 'runs/gamma_fold_transformer_experiment'
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

############################################
# UniProt Data Functions
############################################

# Regex pattern to parse Link header for next cursor.
RE_NEXT_LINK = re.compile(r'<(.+)>; rel="next"')

def get_next_link(headers):
	"""Extract the next link (cursor) from the Link header, if present."""
	link = headers.get("Link")
	if link:
		match = RE_NEXT_LINK.search(link)
		if match:
			return match.group(1)
	return None

def get_batches(url):
	session = requests.Session()
	retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
	session.mount("https://", HTTPAdapter(max_retries=retries))
	batch_url = url
	while batch_url:
		logging.info(f"Fetching batch: {batch_url}")
		response = session.get(batch_url)
		response.raise_for_status()
		total = response.headers.get("x-total-results", "unknown")
		yield response.text, total
		batch_url = get_next_link(response.headers)
		# Be courteous to the API.
		time.sleep(0.5)


def load_local_fasta_files(file_list):
	"""
	Load sequences from each local FASTA file.
	"""
	records = []
	for filepath in file_list:
		if not os.path.exists(filepath):
			logging.warning(f"Local file not found: {filepath}")
			continue
		logging.info(f"Loading local file: {filepath}")
		with open(filepath, "r") as f:
			recs = list(SeqIO.parse(f, "fasta"))
			logging.info(f"Loaded {len(recs)} records from {filepath}")
			records.extend(recs)
	return records


def merge_datasets(uni_records, local_records, min_length=50, max_length=1000):
	"""
	Merge UniProt records and local records; clean, deduplicate, and filter by length.
	"""
	all_records = uni_records + local_records
	cleaned = [clean_sequence(rec) for rec in all_records]
	unique = deduplicate_records(cleaned)
	filtered = [rec for rec in unique if min_length <= len(rec.seq) <= max_length]
	logging.info(f"Filtered records by length: {len(unique)} -> {len(filtered)}")
	return filtered

# Data Cleaning and Merging
def clean_sequence(record):
	seq_str = str(record.seq).upper()
	seq_str = re.sub(r'[^A-Z]', '', seq_str)
	record.seq = type(record.seq)(seq_str)
	return record

def deduplicate_records(records):
	"""Deduplicate SeqRecords based on their sequence string."""
	unique = {}
	for rec in records:
		seq_str = str(rec.seq)
		if seq_str not in unique:
			unique[seq_str] = rec
	deduped = list(unique.values())
	logging.info(f"Deduplicated: {len(records)} -> {len(deduped)}")
	return deduped

def write_fasta(records, output_file):
	"""Write SeqRecords to a FASTA file."""
	with open(output_file, "w") as f:
		SeqIO.write(records, f, "fasta")
	logging.info(f"Wrote {len(records)} sequences to {output_file}")


def build_dataset_from_uniprot(query, target_count, local_files, output_file, min_length, max_length_filter):
	logging.info(f"Starting UniProt collection with query: {query}")
	collected_records = []
	base_url = "https://rest.uniprot.org/uniprotkb/search"
	params = {
		"query": query,
		"format": "fasta",
		"size": 500
	}

	initial_url = base_url + "?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())

	for batch_text, total in get_batches(initial_url):
		# Parse FASTA records from the batch text using StringIO
		fasta_io = StringIO(batch_text)
		batch_records = list(SeqIO.parse(fasta_io, "fasta"))
		collected_records.extend(batch_records)
		logging.info(f"Collected {len(collected_records)} / target {target_count} records (Total available: {total})")
		if len(collected_records) >= target_count:
			break
	
	# Trim to target_count if over-collected
	if len(collected_records) > target_count:
		collected_records = collected_records[:target_count]

	# Load local FASTA files if provided
	local_records = load_local_fasta_files(local_files) if local_files else []
	
	# Merge, clean, deduplicate and filter by length
	merged = merge_datasets(collected_records, local_records, min_length=min_length, max_length=max_length_filter)
	
	# Write merged dataset to output FASTA file
	write_fasta(merged, output_file)
	logging.info("Dataset building complete.")


def train_model(
	fasta_file,
	batch_size=32,
	max_length=256,
	vocab_size=24,
	embed_dim=256,
	num_heads=8,
	num_layers=6,
	dropout=0.1,
	num_epochs=10,
	learning_rate=1e-4,
	warmup_steps=4000
):
	from torch.utils.tensorboard import SummaryWriter
	
	from model import GammaFoldFormer
	from preprocessing import ProteinDataset, collate_fn
	from gammaf.learning import generate_square_subsequent_mask, noam_lr, save_checkpoint

	writer = SummaryWriter(log_dir=LOG_DIR)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Dataset and dataloader
	dataset = ProteinDataset(fasta_file, max_length=max_length)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)


	# GammaFold Former
	model = GammaFoldFormer(
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		num_heads=num_heads,
		num_layers=num_layers,
		dropout=dropout,
		max_len=max_length,
		num_classes=None
	)
	model.to(device)

	if torch.cuda.device_count() > 1:
		logging.info(f"Using {torch.cuda.device_count()} GPUs!")
		model = nn.DataParallel(model)

	## LOSS, Optimizer and Scheduler
	criterion = nn.CrossEntropyLoss(ignore_index=0)
	optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

	scheduler = optim.lr_scheduler.LambdaLR(
		optimizer, lr_lambda=lambda step: noam_lr(step+1, embed_dim, warmup=warmup_steps)
	)

	## Mixed precision training setup
	scalar = torch.amp.GradScaler()
	global_step = 0
	best_loss = float('inf')
	model.train()
	logging.info("Starting training...")

	for epoch in range(1, num_epochs + 1):
		epoch_loss = 0
		for batch_idx, (batch_seqs, seq_lengths) in enumerate(dataloader):
			batch_seqs = batch_seqs.to(device)

			input_seq = batch_seqs[:, :-1]
			target_seq = batch_seqs[:, 1:]
			seq_len = input_seq.size(1)
			src_mask = generate_square_subsequent_mask(seq_len).to(device)

			optimizer.zero_grad()

			with torch.autocast(device_type='cuda', dtype=torch.float16):
				outputs = model(input_seq, attention_mask=src_mask)
				output_flat = outputs.reshape(-1, vocab_size)
				target_flat = target_seq.reshape(-1)
				loss = criterion(output_flat, target_flat)
			
			scalar.scale(loss).backward()
			scalar.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scalar.step(optimizer)
			scalar.update()
			scheduler.step()

			epoch_loss += loss.item()
			writer.add_scalar('Train/StepLoss', loss.item(), global_step)
			writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
			global_step += 1

			if batch_idx % 100 == 0:
				logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
		
		avg_loss = epoch_loss / len(dataloader)
		logging.info(f"Epoch {epoch} complete, Avg Loss: {avg_loss:.4f}")
		writer.add_scalar('Epoch/AverageLoss', avg_loss, epoch)

		if avg_loss < best_loss:
			best_loss = avg_loss
			checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_checkpoint_epoch_{epoch}.pt")
			save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)

	writer.close()
	print("Training complete. Checkpoints saved.")


def main(args):
	logging.info("Building UniProt dataset...")

	build_dataset_from_uniprot(query=args.uniprot_query, target_count=args.target_count, local_files=args.local_files, output_file=args.output, min_length=args.min_length, max_length_filter=args.max_length_filter)
	
	if args.train:
		logging.info("Starting model training...")
		train_model(fasta_file=args.output,
					batch_size=args.batch_size,
					max_length=args.max_length_dataset,
					vocab_size=args.vocab_size,
					embed_dim=args.embed_dim,
					num_heads=args.num_heads,
					num_layers=args.num_layers,
					dropout=args.dropout,
					num_epochs=args.num_epochs,
					learning_rate=args.learning_rate,
					warmup_steps=args.warmup_steps)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Collect protein data and train a Transformer model.")
	parser.add_argument("--uniprot_query", type=str, default="(reviewed:true) AND (organism_id:9606)", help="UniProt query string. Example: '(reviewed:true) AND (organism_id:9606)'")
	parser.add_argument("--target_count", type=int, default=100000, help="Total number of UniProt records to fetch (e.g., 100000)")
	parser.add_argument("--local_files", type=str, nargs='*', default=[], help="List of local FASTA files to include")
	parser.add_argument("--output", type=str, required=True, help="Output FASTA file for the merged dataset")
	parser.add_argument("--min_length", type=int, default=50, help="Minimum sequence length to include")
	parser.add_argument("--max_length_filter", type=int, default=1000, help="Maximum sequence length to include")
	
	# Training parameters
	parser.add_argument("--train", action="store_true", help="Train the model after building dataset")
	parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
	parser.add_argument("--max_length_dataset", type=int, default=256,
						help="Maximum sequence length for training (including SOS/EOS)")
	parser.add_argument("--vocab_size", type=int, default=24, help="Vocabulary size (should match preprocessing)")
	parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
	parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
	parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer encoder layers")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
	parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
	parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
	parser.add_argument('--warmup_steps', type=int, default=4000, help="Scheduler Warmup steps")
	
	args = parser.parse_args()
	main(args)