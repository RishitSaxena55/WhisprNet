from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer


class ASRDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing data for Automatic Speech Recognition (ASR).

    Supports:
    - Partitioning (train-clean-100, dev-clean, test-clean)
    - Feature normalization: global mean-variance, cepstral, or none
    - Tokenization using H4Tokenizer
    - SpecAugment (time and frequency masking)
    - Proper alignment and batching of features and transcripts
    """

    def __init__(
        self,
        partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
        config: dict,
        tokenizer: H4Tokenizer,
        isTrainPartition: bool,
        global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Initialize the ASRDataset.

        Args:
            partition (str): One of ['train-clean-100', 'dev-clean', 'test-clean']
            config (dict): Configuration for loading and preprocessing
            tokenizer (H4Tokenizer): Tokenizer for text transcripts
            isTrainPartition (bool): Whether this partition is used for training (enables SpecAugment)
            global_stats (tuple, optional): (mean, std) tensors for global MVN normalization
        """
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Special token ids
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Feature file paths
        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')
        self.fbank_files = sorted(os.listdir(self.fbank_dir), key=lambda x: x.split('.')[0])

        # Subset size control (for debugging or efficiency)
        subset_size = int(config.get('subset') * len(self.fbank_files))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # Transcript file paths (skip for test-clean)
        if partition != "test-clean":
            self.text_dir = os.path.join(config['root'], partition, 'text')
            self.text_files = sorted(os.listdir(self.text_dir), key=lambda x: x.split('.')[0])
            self.text_files = self.text_files[:subset_size]
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize storage
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        # Statistics
        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # Welford's method for global MVN stats
        if config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training sets")
            count = 0
            mean = torch.zeros(config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load and truncate feature to num_feats
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)[:config['num_feats'], :]
            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update global MVN stats
            if config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)
                batch_count = feat_tensor.shape[1]
                count += batch_count
                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            if partition != "test-clean":
                # Load transcript and tokenize
                text_path = os.path.join(self.text_dir, self.text_files[i])
                transcript = "".join(np.load(text_path))
                self.total_chars += len(transcript)
                tokenized = tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                # Add SOS and EOS tokens
                shifted = [self.sos_token] + tokenized
                golden = tokenized + [self.eos_token]
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # Compute average characters per token
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Finalize global stats
        if config['norm'] == 'global_mvn':
            if global_stats:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def __len__(self) -> int:
        """
        Return the number of samples.
        """
        return self.length

    def get_avg_chars_per_token(self) -> float:
        """
        Return average characters per token (used for character-level perplexity).
        """
        return self.avg_chars_per_token

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load and preprocess a single item.

        Returns:
            feat (FloatTensor): shape (num_feats, time)
            shifted (LongTensor or None): tokenized input (with SOS)
            golden (LongTensor or None): target sequence (with EOS)
        """
        feat = torch.FloatTensor(self.feats[idx])

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(
        self,
        batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Collate function for DataLoader. Pads features and transcripts to uniform size.

        Returns:
            padded_feats (Tensor): (B, T, F)
            padded_shifted (Tensor or None): (B, L)
            padded_golden (Tensor or None): (B, L)
            feat_lengths (Tensor): (B,)
            transcript_lengths (Tensor or None): (B,)
        """
        batch_feats, batch_shifted, batch_golden = zip(*batch)

        # Transpose feats to (time, feat) for padding
        batch_feats = [feat.T for feat in batch_feats]
        feat_lengths = torch.IntTensor([feat.size(0) for feat in batch_feats])
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)

        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            transcript_lengths = torch.IntTensor([t.size(0) for t in batch_shifted])
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # Apply SpecAugment if training
        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, F, T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
