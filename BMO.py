# Source: https://www.youtube.com/watch?v=kCc8FmEb1nY
# Run on model-08.pkl (ModelV9-JSON)
# Implementation of RL (Reinforcement learning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sentencepiece as spm  # For BPE tokenization
import pickle

start_time = time.time()

# Hyperparameters
batch_size = 8
block_size = 16
max_iters = 5000
eval_interval = 1000
learning_rate = 2e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embed = 32
n_head = 4
n_layer = 2
dropout = 0.2
temperature = 1.0  # Temperature for sampling
top_k = 10  # Number of top tokens to sample from

torch.manual_seed(1337)

# Load the trained BPE model
sp = spm.SentencePieceProcessor()
sp.load('bpe.model')

# BPE tokenization
vocab_size = sp.get_piece_size()

# Encoding and decoding functions using BPE
encode = lambda s: sp.encode(s, out_type=int)
decode = lambda l: sp.decode(l)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class EnglishEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_length):
        super(EnglishEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads) for _ in range(num_layers)])

    def create_positional_encoding(self, max_length, d_model):
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros(max_length, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, memory_output=None, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        if memory_output is not None:
            x = x + memory_output[:, -T:, :]

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class MemoryModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MemoryModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x, hidden_state=None):
        if x.size(0) == 0:
            raise ValueError("Input to LSTM is empty.")

        # Use a fixed hidden state for demonstration
        hidden_state = (torch.zeros(1, x.size(0), self.hidden_size, device=x.device),
                        torch.zeros(1, x.size(0), self.hidden_size, device=x.device))

        x, (h_n, c_n) = self.lstm(x, hidden_state)
        return x, (h_n, c_n)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EnglishEncoder(vocab_size, n_embed, n_head, n_layer, block_size)
        self.decoder = Decoder(vocab_size, n_embed, n_head, n_layer, dropout)
        self.memory = MemoryModule(n_embed, n_embed)
        self.temperature = temperature  # Set temperature

        # Define special tokens (e.g., EOS token)
        self.eos_token_id = sp.piece_to_id('</s>')  # Replace with the actual EOS token if different

    def forward(self, idx, targets=None):
        encoder_output = self.encoder(idx)
        memory_output, hidden_state = self.memory(encoder_output)
        logits, loss = self.decoder(idx, memory_output, targets)
        return logits, loss

    def top_k_sampling(self, logits, k):
        # Apply temperature to logits
        logits = logits / self.temperature
        # Get the top k logits
        top_k_values, top_k_indices = torch.topk(logits, k)
        top_k_probs = F.softmax(top_k_values, dim=-1)

        # Sample from the top k
        idx_next = torch.multinomial(top_k_probs, num_samples=1)

        return top_k_indices[torch.arange(top_k_indices.size(0)), idx_next].squeeze().item()

    def generate(self, idx, max_new_tokens=500, p=0.9, max_length_limit=1000):
        hidden_state = (None, None)
        total_generated_length = 0

        for _ in range(max_new_tokens):
            # Check length limit
            if total_generated_length >= max_length_limit:
                break

            idx_cond = idx[:, -block_size:].to(device)  # Move to device if not already
            encoder_output = self.encoder(idx_cond)

            if hidden_state[0] is None:
                hidden_state = (
                    torch.zeros(1, idx_cond.size(0), n_embed, device=device),
                    torch.zeros(1, idx_cond.size(0), n_embed, device=device)
                )

            memory_output, hidden_state = self.memory(encoder_output, hidden_state)  # Pass hidden_state here
            logits, _ = self.decoder(idx_cond, memory_output)
            logits = logits[:, -1, :]

            # Use top-k sampling to get the next token index
            idx_next = self.top_k_sampling(logits, top_k)

            # Check if the generated token is an end token or an incomplete word token
            if idx_next == self.eos_token_id or self.is_incomplete_word(idx_next):
                continue  # Skip this token and regenerate

            # Convert idx_next to a tensor before concatenating
            idx_next_tensor = torch.tensor([[idx_next]], dtype=torch.long, device=device)
            idx = torch.cat((idx, idx_next_tensor), dim=1)

            # Update the total generated length
            total_generated_length += 1

            # Early stopping on punctuation (adjust as needed)
            generated_text = decode(idx[0].tolist())
            if generated_text[-1] in {'.', '!', '?'}:
                break

        return idx

    def is_incomplete_word(self, token_id):
        # Use SentencePiece to check if token is an incomplete word or special token
        token = sp.id_to_piece(token_id)
        return token.startswith('▁') is False and token != '</s>'  # Adjust condition as needed

    def format_generated_text(self, generated_text):
        # Break text into lines without cutting words
        max_line_length = 150
        lines = []
        start = 0

        while start < len(generated_text):
            # Find the optimal break point within the max length
            end = min(start + max_line_length, len(generated_text))

            # Check if we need to break at a space or punctuation
            if end < len(generated_text) and not generated_text[end].isspace():
                # Move backwards to find a space or punctuation
                while end > start and not generated_text[end].isspace():
                    end -= 1

            # If no space or punctuation is found, just break at the max length
            if end == start:
                end = start + max_line_length

            # Append the line
            lines.append(generated_text[start:end].strip())
            start = end

        return "\n".join(lines)

# Load trained model from file if available
try:
    with open('model-08.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Trained model not found. Please train the model first.")

while True:
    prompt = input("Enter a prompt: ")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # Generate text with flexible length
    generated_idx = model.generate(context, max_new_tokens=500).tolist()[0]
    generated_text = decode(generated_idx)

    # Format the generated text with line breaks
    formatted_text = model.format_generated_text(generated_text)
    print(f"Generated text:\n{formatted_text}")