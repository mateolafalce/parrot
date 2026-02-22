<div align="center">

<img src="./parrot.jpg" width="300" height="300" alt="Description">   

# Parrot

</div>

A transformer language model (MiniGPT) trained from scratch on real WhatsApp conversations to mimic how I (Mateo) respond.

## What is it?

Parrot parses a WhatsApp chat export, extracts conversation pairs (input message -> response), and trains a decoder-only transformer to learn how to respond like me given a message from X user.

The model is lightweight (~4.2M parameters) and designed to run on Google Colab with a GPU.

## Architecture

```
MiniGPT (decoder-only transformer)
├── Token Embedding (vocab_size x 256)
├── Positional Embedding (128 x 256)
├── 4x TransformerBlock
│   ├── Pre-LayerNorm
│   ├── Multi-Head Self-Attention (4 heads)
│   ├── Residual + Dropout
│   ├── Pre-LayerNorm
│   ├── Feed-Forward (256 → 1024 → 256, GELU)
│   └── Residual + Dropout
├── Final LayerNorm
└── Linear Head (weight tying with embedding)
```

| Hyperparameter | Value |
|---|---|
| `D_MODEL` | 256 |
| `N_HEADS` | 4 |
| `N_LAYERS` | 4 |
| `D_FF` | 1024 |
| `MAX_LEN` | 128 |
| `DROPOUT` | 0.1 |

## Training

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Cosine Annealing |
| Loss | CrossEntropyLoss (ignores padding) |
| Gradient clipping | 1.0 |

## Pipeline

1. **Parsing**: Upload a WhatsApp export (`.txt`). Messages are extracted with regex and consecutive messages from the same sender are grouped.
2. **Pairing**: `{input, output}` pairs are built where the input is what Javi said and the output is my response. Multimedia messages are filtered out.
3. **Tokenization**: A BPE tokenizer is used with special tokens (`SOS`, `EOS`, `PAD`).
4. **Training**: The transformer is trained with teacher forcing on concatenated sequences (input + output).
5. **Generation**: Autoregressive decoding with temperature scaling and top-k sampling (k=40).

## Pre-trained weights

The trained model weights are available on Hugging Face:

[lafalce/parrot/minigpt_chat.pt](https://huggingface.co/lafalce/parrot/blob/main/minigpt_chat.pt)

## How to use it

The project is contained in a single notebook: `parrot.ipynb`. It's designed to run on Google Colab.

1. Open `parrot.ipynb` in Google Colab
2. Upload a WhatsApp chat export when prompted
3. Run the cells in order to parse, tokenize, and train
4. Use the interactive loop in the last cell to chat with the model

To load the pre-trained weights instead of training from scratch:

```python
checkpoint = torch.load("minigpt_chat.pt")
model.load_state_dict(checkpoint['model_state'])
```
