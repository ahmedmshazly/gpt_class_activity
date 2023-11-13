import torch
import argparse
from gpt_model import GPTLanguageModel
from model_components import *

# Function to load and process the vocabulary
def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    return chars

# Encoding and decoding functions
def encode(s, string_to_int):
    return [string_to_int[c] for c in s]

def decode(l, int_to_string):
    return ''.join([int_to_string[i] for i in l])

# Argument Parsing
parser = argparse.ArgumentParser(description='Chatbot Demonstration Program')
parser.add_argument('-batch_size', type=int, required=True, help='Please provide a batch_size')
# args = parser.parse_args()

# print(f'Batch size: {args.batch_size}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Constants
block_size = 64
n_embd = 256
n_head = 2
n_layer = 2
dropout = 0.2

# Load and process vocabulary
vocab_file_path = "./vocab.txt"
chars = load_vocab(vocab_file_path)
vocab_size = len(chars)
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}

# Model Initialization and Loading
model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
model_path = './model.pkl'
print('Loading model parameters...')
try:
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    print('Model loaded successfully!')
except FileNotFoundError:
    print(f'Warning: Model file {model_path} not found. Using uninitialized model.')

def filter_input(text, allowed_chars):
    return ''.join([c for c in text if c in allowed_chars])

# Inside the interactive chat loop
while True:
    prompt = input("Prompt:\n")
    prompt = filter_input(prompt, chars)  # Filter out-of-vocabulary characters
    context = torch.tensor(encode(prompt, string_to_int), dtype=torch.long, device=device)

    # print("Token Embedding Size:", model.token_embedding_table.weight.size())
    # print("Position Embedding Size:", model.position_embedding_table.weight.size())
    # encoded_input = encode(prompt, string_to_int)
    # print("Encoded Input:", encoded_input)


    if len(context) > block_size:
        context = context[:block_size]  # Truncate to block size

    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=64)[0].tolist(), int_to_string)
    print(f'Completion:\n{generated_chars}')