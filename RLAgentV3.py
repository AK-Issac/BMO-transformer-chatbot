# Creation of the Training for the RLAgent
# The core of this agent is the model-08.pkl (ModelV9_TXT.py)
# Implementation of the environment for RL
# Implementation of Greedy Policy New*
# Implementation of Q-Learning New*
# Reward function is Implemented BUT need to work on the function "is_coherent" of the RL New*

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sentencepiece as spm  # For BPE tokenization
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re

# Ensure you have downloaded the necessary NLTK resources:
# nltk.download('punkt')
# nltk.download('stopwords')

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
        total_generated_length = 0

        for _ in range(max_new_tokens):
            # Check length limit
            if total_generated_length >= max_length_limit:
                break

            idx_cond = idx[:, -block_size:].to(device)  # Move to device if not already
            encoder_output = self.encoder(idx_cond)

            # Use a fixed hidden state
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


# REINFORCEMENT LEARNING SECTION
class QLearningAgent:
    def __init__(self, action_space, state_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = torch.zeros(state_size, action_space)  # Initialize Q-table with zeros
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = action_space

    def choose_action(self, state, greedy=False):
        if greedy or random.random() > self.epsilon:
            # Choose action based on the highest Q-value (greedy)
            return torch.argmax(self.q_table[state]).item()
        else:
            # Explore: Choose a random action
            return random.randint(0, self.action_space - 1)

    def update(self, state, action, reward, next_state):
        # Update Q-table using the Q-learning formula
        best_next_action = torch.argmax(self.q_table[next_state]).item()
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]

        # Update Q-value using the learning rate
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Update Chatbot Environment to work with Q-learning
class ChatbotEnvironment:
    def __init__(self, model, eos_token_id, max_turns=5):
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        self.current_turn = 0
        self.prompt = ""  # Initialize the prompt
        return self.get_state()

    def get_state(self):
        encoded_prompt = encode(self.prompt)
        if len(encoded_prompt) == 0:
            return torch.tensor([0], dtype=torch.long)  # Handle empty state case
        return torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0)

    def step(self, action):
        response_idx = torch.tensor([[action]], dtype=torch.long)
        self.prompt += decode(response_idx.tolist()[0])  # Update the prompt

        # Get current state
        current_state = self.get_state()

        # Check if the current state is empty
        if current_state.size(1) == 0:  # Sequence length is zero
            return current_state, 0, True  # Return immediately

        # Generate text based on the updated prompt
        try:
            generated_idx = self.model.generate(current_state, max_new_tokens=1)
        except Exception as e:
            print("Error during text generation:", str(e))
            raise  # Re-raise to get the full stack trace

        generated_text = decode(generated_idx[0].tolist())

        # Update the prompt with the generated text
        self.prompt += generated_text

        # Calculate reward (based on response length, coherence, user feedback, etc.)
        reward = self.calculate_reward(generated_text)

        self.current_turn += 1
        done = self.current_turn >= self.max_turns or action == self.eos_token_id  # End if max turns reached or EOS token

        return self.get_state(), reward, done

    def calculate_reward(self, generated_text):
        """
        Reward the chatbot based on how well the generated text fits the context.
        """
        reward = 0

        # Reward for coherence
        if self.is_incoherent(generated_text):
            return -2  # Strong penalty for incoherence

        # Reward for appropriate length
        length = len(generated_text)
        if length < 5:
            reward -= 0.5  # Penalize for being too short
        elif length > 100:
            reward -= 0.5  # Penalize for being too long
        else:
            reward += 1  # Reward for appropriate length

        # Reward for good endings
        if generated_text.strip().endswith(('.', '!', '?')):
            reward += 1  # Stronger reward for a proper ending

        # Additional rewards for maintaining context or user satisfaction
        if self.context_is_maintained(generated_text):
            reward += 1  # Reward for maintaining context

        return reward

    def is_incoherent(self, text):
        """
        Check if the generated text is incoherent based on the provided context.
        """
        # Check for empty text
        if len(text.strip()) == 0:
            return True

        # Check for nonsensical phrases
        nonsensical_phrases = ['asdlkfjas', 'qwerty', '123456', 'blah', 'foo', 'bar']
        if any(phrase in text for phrase in nonsensical_phrases):
            return True

        # Check for excessive length
        if len(text.split()) < 5 or len(text.split()) > 100:
            return True

        # Check for context relevance
        context_words = set(word_tokenize(self.prompt.lower()))  # Current context words
        generated_words = set(word_tokenize(text.lower()))  # Words in the generated text

        # Check if the generated text shares any keywords with the context
        common_words = context_words.intersection(generated_words)
        if len(common_words) == 0:
            return True  # No overlap means incoherent

        # Further checks for semantic consistency (can be improved with a language model)
        relevant_keywords = ['context', 'user', 'response', 'chatbot', 'conversation']  # Modify as needed
        if not any(keyword in text.lower() for keyword in relevant_keywords):
            return True

        # Optional: Stopword removal for more focused keyword checks
        stop_words = set(stopwords.words('english'))
        filtered_context = [word for word in context_words if word not in stop_words]
        filtered_generated = [word for word in generated_words if word not in stop_words]

        # Count the frequency of words in the filtered context
        context_word_counts = Counter(filtered_context)

        # Check if the generated text includes relevant words from the context
        for word in filtered_generated:
            if word in context_word_counts and context_word_counts[word] > 0:
                return False  # Found relevant word in generated text

        return True  # If all checks fail, return incoherent

    def context_is_maintained(self, generated_text):
        """
        Check if the generated text maintains the context of the conversation.
        This is a placeholder function; implement your logic here.
        """
        # Implement logic to check if the context is maintained
        return True  # Example placeholder


# Load trained model from file if available
try:
    with open('model-08.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Trained model not found. Please train the model first.")

# Initialize Q-learning agent
state_size = vocab_size  # Example: the size of your state space (number of unique states)
action_space = vocab_size  # Number of possible actions (vocabulary size)
agent = QLearningAgent(action_space, state_size)

# RL Training Loop
num_episodes = 1000  # Number of training episodes

# Initialize the environment
env = ChatbotEnvironment(model, model.eos_token_id)

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Choose action using epsilon-greedy policy
        action = agent.choose_action(state)

        # Step in the environment with the chosen action
        next_state, reward, done = env.step(action)
        total_reward += reward

        # Update the Q-table based on the action taken
        agent.update(state, action, reward, next_state)

        # Transition to the next state
        state = next_state

    agent.decay_epsilon()  # Decay epsilon after each episode
    print(f"Episode {episode + 1}/{num_episodes} finished with total reward: {total_reward}")

# Save the trained model after training
with open('rlagent-03.pkl', 'wb') as f:
    pickle.dump(model, f)
print("RLAgent saved successfully.")
