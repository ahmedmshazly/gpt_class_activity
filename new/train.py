import argparse
import torch
from gpt_model import GPTLanguageModel
from data_handling import DataLoader
from train_eval import ModelTrainer

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='GPT Language Model Training Program')
    parser.add_argument('-batch_size', type=int, required=True, help='Batch size for training')
    return parser.parse_args()

batch_size = 4
block_size = 64  # Smaller block size to reduce memory usage
max_iters = 200  # Adjust as needed
learning_rate = 4e-4  # Slightly higher learning rate
eval_iters = 200  # Evaluate less frequently
n_embd = 256  # Reduced embedding size
n_head = 2  # Fewer attention heads
n_layer = 2  # Fewer layers
dropout = 0.2  # Dropout stays the same


def main():
    args = parse_arguments()

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    print(f"Using device: {device}")

    # Initialize data loader
    data_loader = DataLoader('./vocab.txt', './train_split.txt', './val_split.txt',
                             block_size, args.batch_size)

    # Initialize GPT language model
    model = GPTLanguageModel(data_loader.vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize model trainer
    trainer = ModelTrainer(model, optimizer, data_loader, device)

    # Training loop
    trainer.train(max_iters, eval_iters)

    # Save the trained model
    trainer.save_model('./model.pkl')
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
