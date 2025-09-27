# Source: https://www.youtube.com/watch?v=kCc8FmEb1nY
# Decoder + Encoder with model creation
# Works on word-level
# Handle unknown words
# Remove GNN Layer/Model because it doesn't fit the chatbot New*
# Implementation of Memory mechanism (LSTM)
# Implementation of Top-k Sampling (New param: temperature; Range: 0.7 - 1.2; 0.7 being less random but more coherent; 1.2 being more random but less coherent; 1.0 a bit of both)
# Optimisation (Should now works on GPU)
# Implementation of EOS token New*

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
vocab.insert(0, '</s>')
vocab.insert(0, '<unk>')
vocab_size = len(vocab)

# Create mapping from word to index and index to word
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}

# Encoding and decoding functions
encode = lambda s: [stoi.get(word, stoi['<unk>']) for word in s]
decode = lambda l: ' '.join([itos[i] for i in l]).replace(' < /s >', '</s>').replace('< /s >', '</s>').strip()

# Encode the entire dataset
data = torch.tensor(encode(words), dtype=torch.long)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
        self.eos_token = stoi['</s>']  # EOS token index

    def forward(self, idx, targets=None):
        # Encoding phase
        enc_output = self.encoder(idx)

        # Memory mechanism
        enc_output, _ = self.memory(enc_output)

        # Decoding phase
        logits, loss = self.decoder(idx, enc_output, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            # Apply temperature and top-k sampling
            logits = logits[:, -1, :] / self.temperature
            logits = F.softmax(logits, dim=-1)
            top_logits, top_indices = torch.topk(logits, top_k)

            # Sample from top-k tokens
            next_token = torch.multinomial(top_logits, num_samples=1)
            next_token = top_indices.gather(1, next_token)

            # Debugging output
            print(f"Generated token: {next_token.item()}, EOS token: {self.eos_token}")

            idx = torch.cat((idx, next_token), dim=1)

            # Check if the EOS token is reached
            if next_token.item() == self.eos_token:
                print("EOS token reached.")
                break

        # Decode and format output to remove spaces around EOS token
        return decode(idx[0].tolist()).strip()

# Initialize model
model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open('model-08.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved")

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=500)
print(decode(generated_text[0].tolist()))

print("--- %s seconds ---" % (time.time() - start_time))
