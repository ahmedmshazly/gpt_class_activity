import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
class Head(nn.Module):
    """
    Represents a single head in a multi-head self-attention mechanism.

    In the context of transformers, self-attention is a mechanism that allows each position in a sequence
    to attend to all positions in the previous layer of the sequence. Each head in multi-head attention
    can potentially focus on different parts of the input sequence, enabling the model to capture various
    types of dependencies.

    Attributes:
        key (nn.Linear): Linear layer to compute the 'key' in attention mechanism.
        query (nn.Linear): Linear layer to compute the 'query' in attention mechanism.
        value (nn.Linear): Linear layer to compute the 'value' in attention mechanism.
        tril (torch.Tensor): A lower triangular matrix used for masking in attention.
        dropout (nn.Dropout): Dropout layer to avoid overfitting in the attention mechanism.
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        Initializes the Head instance.

        Args:
            head_size (int): Size of this attention head.
            n_embd (int): The size of each embedding vector.
            block_size (int): Size of the sequence block considered in attention.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Defines the forward pass of the Head.

        During the forward pass, the query, key, and value vectors are computed. The attention scores
        are calculated and then used to weight the values. The attention scores are masked to ensure
        that the prediction for a particular position only depends on the known outputs at positions before it.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, time step, channels).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, of shape (batch size, time step, head size).
        """
        B, T, C = x.shape
        k = self.key(x)   # Compute the keys (B, T, head_size)
        q = self.query(x) # Compute the queries (B, T, head_size)
        v = self.value(x) # Compute the values (B, T, head_size)

        # Calculating attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking to ensure causality
        wei = F.softmax(wei, dim=-1) # Normalizing the scores
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v # (B, T, head_size)
        return out



class MultiHeadAttention(nn.Module):
    """
    Implements multiple heads of self-attention in parallel.

    Multi-head attention allows the model to jointly attend to information from different representation subspaces
    at different positions. This is achieved by having multiple attention heads, each performing self-attention 
    independently, and then concatenating their outputs.

    Attributes:
        heads (nn.ModuleList): A list of 'Head' instances representing individual attention heads.
        proj (nn.Linear): Linear layer to project concatenated head outputs back to the model dimension.
        dropout (nn.Dropout): Dropout layer applied after the projection.
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        """
        Initializes the MultiHeadAttention instance.

        Args:
            num_heads (int): Number of attention heads.
            head_size (int): Size of each attention head.
            n_embd (int): The size of each embedding vector.
            block_size (int): Size of the sequence block considered in attention.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Defines the forward pass of the MultiHeadAttention.

        Each head in the module list performs self-attention independently, and their outputs are concatenated.
        This concatenated output is then projected back to the dimensionality of the model and passed through a
        dropout layer for regularization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, time step, channels).

        Returns:
            torch.Tensor: Output tensor after applying multi-head self-attention, with the same shape as the input tensor.
        """
        # Applying self-attention in each head and concatenating the results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, F)

        # Projecting the concatenated output back to the model dimension and applying dropout
        out = self.dropout(self.proj(out))  # (B, T, n_embd)
        return out


class FeedFoward(nn.Module):
    """
    Implements a FeedFoward neural network layer within the transformer block.

    This class represents the FeedFoward layer used in each transformer block, consisting of two linear transformations
    with a ReLU activation in between. Such layers are applied after the multi-head attention in each block of the transformer.

    Attributes:
        net (nn.Sequential): A sequential container of layers forming the FeedFoward network.
    """

    def __init__(self, n_embd, dropout):
        """
        Initializes the FeedFoward instance.

        Args:
            n_embd (int): The size of each embedding vector.
            dropout (float): Dropout rate for regularization in the network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expanding the input 4 times its size
            nn.ReLU(),                      # Applying non-linearity (ReLU)
            nn.Linear(4 * n_embd, n_embd),  # Projecting back to the original dimension
            nn.Dropout(dropout)             # Applying dropout for regularization
        )

    def forward(self, x):
        """
        Defines the forward pass of the FeedFoward layer.

        The input passes through two linear transformations with a ReLU activation and dropout applied
        in between. This process adds non-linearity to the model and enables it to learn complex patterns.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, time step, channels).

        Returns:
            torch.Tensor: Output tensor after passing through the FeedFoward network.
        """
        return self.net(x)


class Block(nn.Module):
    """
    Represents a single transformer block.

    A transformer block consists of two main parts: a multi-head self-attention mechanism and a FeedFoward network.
    Each of these parts is followed by a residual connection and layer normalization. This structure allows the
    transformer to effectively process sequential data by enabling both communication (through attention) and
    complex computation (through FeedFoward layers).

    Attributes:
        sa (MultiHeadAttention): The multi-head self-attention component of the block.
        ffwd (FeedFoward): The FeedFoward network component of the block.
        ln1 (nn.LayerNorm): Layer normalization applied after the self-attention and residual connection.
        ln2 (nn.LayerNorm): Layer normalization applied after the FeedFoward network and residual connection.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        """
        Initializes the Block instance.

        Args:
            n_embd (int): The size of each embedding vector.
            n_head (int): The number of attention heads in the multi-head attention.
            dropout (float): Dropout rate for regularization in the network.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Defines the forward pass of the Block.

        The input first passes through the self-attention mechanism, followed by a residual connection and layer normalization.
        Then, it goes through the FeedFoward network, followed again by a residual connection and layer normalization.
        This process allows the block to both distribute attention across different parts of the input and apply
        complex transformations, while maintaining stability in training through normalization and residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (batch size, time step, channels).

        Returns:
            torch.Tensor: The output tensor of the block, with the same shape as the input tensor.
        """
        y = self.sa(x)
        x = self.ln1(x + y)  # Apply self-attention, followed by residual connection and normalization
        y = self.ffwd(x)
        x = self.ln2(x + y)  # Apply FeedFoward network, followed by residual connection and normalization
        return x
