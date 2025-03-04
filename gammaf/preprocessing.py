import torch
from torch.utils.data import Dataset
from Bio import SeqIO

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'


vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
for i, aa in enumerate(AMINO_ACIDS):
    vocab[aa] = i + 4

inv_vocab = {idx: token for token, idx in vocab.items()}

def tokenize_sequence(seq):
    tokens = [SOS_TOKEN] + list(seq.upper()) + [EOS_TOKEN]
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    return token_ids

class ProteinDataset(Dataset):
    def __init__(self, fasta_file, max_length=512):
        """
        Reads a FASTA file and tokenizes protein sequences
        NOTE: If the seq is too long, it will be truncated
        and SOS and EOS will be appended
        """
        self.tokenized_seqs = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq)
            if len(seq) + 2 > max_length:
                seq = seq[:max_length - 2]
            token_ids = tokenize_sequence(seq)
            self.tokenized_seqs.append(token_ids)

    def __len__(self):
        return len(self.tokenized_seqs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_seqs[idx], dtype=torch.long)
    
def collate_fn(batch):
    """
    Pads sequences in the batch to the length of the longest sequence.
    Returns the padded tensor and a tensor of original lengths.
    """
    lengths = [len(x) for x in batch]
    max_len = max(lengths)
    padded_batch = []
    for seq in batch:
        padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
        padded_batch.append(padded)
    return torch.stack(padded_batch), torch.tensor(lengths, dtype=torch.long)


