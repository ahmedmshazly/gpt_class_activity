import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_components import Block

class GPTLanguageModel(nn.Module):
    """
    Implements a GPT language model.

    This model is based on the transformer architecture, specifically designed for generative pre-training
    of language models. It consists of token and position embedding layers, followed by a sequence of transformer
    blocks, and a final layer to generate predictions for the next token in the sequence.

    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer for tokens.
        position_embedding_table (nn.Embedding): Embedding layer for token positions.
        blocks (nn.Sequential): Sequential container of transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        lm_head (nn.Linear): Linear layer to map the output to the vocabulary size.
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        """
        Initializes the GPTLanguageModel instance.

        Args:
            vocab_size (int): Size of the vocabulary.
            n_embd (int): The size of each embedding vector.
            n_head (int): The number of attention heads in each transformer block.
            n_layer (int): The number of transformer blocks in the model.
            block_size (int): Size of the sequence block considered in attention.
            dropout (float): Dropout rate for regularization in the network.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights of the model's layers.

        This method is applied to each module in the model. It initializes the weights of linear and embedding
        layers following a normal distribution, which is a common practice in training deep learning models.

        Args:
            module (nn.Module): A module in the model.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None, device='cpu'):
        """
        Forward pass of the GPTLanguageModel.

        Processes an input sequence (index) and computes the logits for each token in the sequence.
        If targets are provided, it also computes the loss, which can be used for training.

        Args:
            index (torch.Tensor): A tensor of token indices with shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): A tensor of target token indices with the same shape as 'index'.
            device (str, optional): The device ('cpu' or 'cuda') to perform computations on.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing logits and, if targets are provided, the loss.
        """
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # Token embeddings (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # Positional embeddings (T, C)
        x = tok_emb + pos_emb  # Combine token and position embeddings (B, T, C)
        x = self.blocks(x)  # Pass through transformer blocks (B, T, C)
        x = self.ln_f(x)  # Apply final layer normalization (B, T, C)
        logits = self.lm_head(x)  # Project to vocabulary size (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens, device='cpu'):

        """
        Generates new tokens given a context (index).

        This function autoregressively generates new tokens based on the provided context.
        It predicts the next token, appends it to the context, and repeats this process.

        Args:
            index (torch.Tensor): A tensor of token indices with shape (batch_size, current_sequence_length).
            max_new_tokens (int): The maximum number of new tokens to generate.
            device (str, optional): The device ('cpu' or 'cuda') to perform computations on.

        Returns:
            torch.Tensor: The tensor containing the original and newly generated token indices.
        """
        max_seq_length = 64  # Assuming this is your model's maximum sequence length
        for _ in range(max_new_tokens):
            if index.size(1) >= max_seq_length:
                index = index[:, -max_seq_length + 1:]  # Keep the most recent tokens
            logits, _ = self.forward(index, device=device)  # Predict next token
            logits = logits[:, -1, :]  # Focus on the last time step
            probs = F.softmax(logits, dim=-1)  # Softmax to get probabilities
            index_next = torch.multinomial(probs, num_samples=1)  # Sample next token
            index = torch.cat((index, index_next), dim=1)  # Append to the sequence

        return index

