# Source: https://www.youtube.com/watch?v=UU1WVnMk4E8
# Run on model-07.pkl (ModelV8_TXT.py)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import nltk
import pickle

# Ensure nltk is installed and download necessary resources if needed
# nltk.download('punkt')

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
lstm_hidden_size = n_embed  # Size of LSTM hidden state

torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Word-level tokenization
words = nltk.word_tokenize(text.lower())
vocab = sorted(set(words))

# Add <UNK> token to the vocabulary
vocab.insert(0, '<unk>')
vocab_size = len(vocab)

# Create mapping from word to index and index to word
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}

# Encoding and decoding functions
encode = lambda s: [stoi.get(word, stoi['<unk>']) for word in s]
decode = lambda l: ' '.join([itos[i] for i in l])

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

    def forward(self, x, hidden_state=None):
        output, hidden_state = self.lstm(x, hidden_state)
        return output, hidden_state

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EnglishEncoder(vocab_size, n_embed, n_head, n_layer, block_size)
        self.decoder = Decoder(vocab_size, n_embed, n_head, n_layer, dropout)
        self.memory = MemoryModule(n_embed, n_embed)
        self.temperature = temperature  # Set temperature

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

        # Ensure idx_next is within the valid vocabulary range
        idx_next_token = top_k_indices[torch.arange(top_k_indices.size(0)), idx_next].squeeze().item()

        if idx_next_token >= vocab_size:  # Handle out-of-range values
            idx_next_token = vocab_size - 1  # Set to the last valid token in the vocabulary

        return idx_next_token
    def generate(self, idx, max_new_tokens, p=0.9):
        hidden_state = (None, None)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:].to(device)  # Move to device if not already
            encoder_output = self.encoder(idx_cond)

            if hidden_state[0] is None:
                hidden_state = (
                    torch.zeros(1, idx_cond.size(0), n_embed, device=device),
                    torch.zeros(1, idx_cond.size(0), n_embed, device=device)
                )

            memory_output, hidden_state = self.memory(encoder_output, hidden_state)
            logits, _ = self.decoder(idx_cond, memory_output)
            logits = logits[:, -1, :]

            # Use top-k sampling to get the next token index
            idx_next = self.top_k_sampling(logits, top_k)

            # Check if the generated token is the same as the last token of the input
            if idx_next == idx_cond[0, -1].item():
                continue  # Skip this token and regenerate

            # Convert idx_next to a tensor before concatenating
            idx_next_tensor = torch.tensor([[idx_next]], dtype=torch.long, device=device)
            idx = torch.cat((idx, idx_next_tensor), dim=1)
        return idx

# Initialize model
model = BigramLanguageModel()
model = model.to(device)

# Load trained model from file if available
try:
    with open('model-07.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Trained model not found. Please train the model first.")

# Allow user to input a prompt and generate text
while True:
    prompt = input("Enter a prompt: ")
    context = torch.tensor(encode(nltk.word_tokenize(prompt.lower())), dtype=torch.long, device=device).unsqueeze(0)
    generated_idx = model.generate(context, max_new_tokens=500).tolist()[0]
    generated_text = decode(generated_idx)
    print(f"Generated text:\n{generated_text}")
