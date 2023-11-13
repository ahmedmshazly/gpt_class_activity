import torch
import mmap
import random
import torch

class DataLoader:
    """
    DataLoader handles the loading and preprocessing of data.

    It reads the text data, creates a vocabulary, and provides utility functions for encoding and decoding text,
    as well as fetching random chunks of data from the dataset.
    """

    def __init__(self, vocab_file, train_file, val_file, block_size, batch_size):
        """
        Initializes the DataLoader with file paths and training parameters.

        Args:
            vocab_file (str): Path to the file containing the vocabulary.
            train_file (str): Path to the training data file.
            val_file (str): Path to the validation data file.
            block_size (int): Size of a block of data.
            batch_size (int): Number of samples in a batch.
        """
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_file = vocab_file
        self.train_file = train_file
        self.val_file = val_file
        self.chars = self._load_vocab()
        self.vocab_size = len(self.chars)
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}

    def _load_vocab(self):
        """
        Loads the vocabulary from the specified file.

        Returns:
            List[str]: A list of unique characters in the vocabulary.
        """
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            text = f.read()
            return sorted(list(set(text)))

    def encode(self, s):
        """
        Encodes a string into a list of integers based on the vocabulary.

        Args:
            s (str): The string to encode.

        Returns:
            List[int]: The encoded list of integers.
        """
        return [self.string_to_int[c] for c in s]

    def decode(self, l):
        """
        Decodes a list of integers back into a string.

        Args:
            l (List[int]): The list of integers to decode.

        Returns:
            str: The decoded string.
        """
        return ''.join([self.int_to_string[i] for i in l])

    def get_random_chunk(self, split):
        """
        Gets a random chunk of data from the specified split (train or validation).

        Args:
            split (str): The data split to use ('train' or 'val').

        Returns:
            torch.Tensor: A tensor containing encoded data.
        """
        filename = self.train_file if split == 'train' else self.val_file
        with open(filename, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                start_pos = random.randint(0, file_size - self.block_size * self.batch_size)

                mm.seek(start_pos)
                block = mm.read(self.block_size * self.batch_size - 1)
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

                data = torch.tensor(self.encode(decoded_block), dtype=torch.long)
                return data

    def get_batch(self, split, device):
        """
        Generates a batch of data for training or validation.

        Args:
            split (str): The data split to use ('train' or 'val').
            device (torch.device): The device to use for the data (e.g., 'cpu', 'cuda').

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors (x, y) for training.
        """
        data = self.get_random_chunk(split)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)
