
# BMO Transformer RL Chatbot  

**BMO** is a hands-on experiment in building a conversational AI **from the ground up**.  
Instead of relying on massive pre-trained black boxes, this project dives into the **fundamentals**:  

- Tokenization with SentencePiece  
- LSTMs and Transformer architectures  
- Reinforcement learning for chatbot optimization  
- Training neural networks from scratch  

BMO is both a **learning journey** and a **working chatbot**—every line of code represents progress in understanding how modern language models really work.  

---

## Important Notes  
- The included checkpoint **`model.pkl` only runs with CUDA**.  
- If CUDA isn’t available, you’ll need to **retrain the model** on your machine.  

---

## Requirements  

Install the following dependencies (make sure your CUDA version matches):  


numpy==1.26.3
sentencepiece==0.2.0
torch==2.5.1+cuXXX       # Replace XXX with your CUDA version (e.g. cu118)
torchaudio==2.5.1+cuXXX
torchvision==0.20.1+cuXXX
transformers==4.46.2


---

## Project Structure

* **`model.py`** — Model architecture, dataset loader, loss functions, training loop
* **`dataset.txt`** — Raw training data (dialogues or text samples)
* **`BMO.py`** — Interactive chatbot interface
* **`model.pkl`** — Pretrained weights (CUDA required)
* **`bpe.model` / `bpe.vocab`** — SentencePiece BPE tokenizer files

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/bmo-transformer-rl-chatbot.git
   cd bmo-transformer-rl-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Retrain the model**
   If CUDA differs or you want a fresh model:

   ```bash
   python model.py
   ```

4. **Run the chatbot**

   ```bash
   python BMO.py
   ```

---

## Why BMO?

This isn’t just another chatbot.
BMO is a **living lab notebook** that shows the evolution of an AI through:

* Subword tokenization with BPE
* Hybrid LSTM + Transformer design
* Reinforcement learning feedback loops
* Training and experimenting with custom datasets

If you’ve ever wanted to peek under the hood of large language models, BMO is a **practical guide** to the inner workings.

---


