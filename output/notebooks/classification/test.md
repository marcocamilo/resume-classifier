---
title: Encoder Model
execute:
  eval: false
---

### Encoder Model

Our next model utilizes the encoder component of the Transformer architecture. Unlike decoders, which generate sequences from dense representations, encoders create gense representations of input sequences. Their ability to extract sequence information makes them particularly useful for tasks such as sentiment analysis, named entity recognition, and text classification. The encoder model here is based on the Transformer architecture described in [Attention is All You Need](https://arxiv.org/abs/1706.03762) and is used as a baseline for transformer-based models.

The packages as well as data preparation steps for this model are the same as those for the Feedforward Neural Network model (see [packages](https://marcocamilo.com/resume-analyzer#import-packages-and-data-1) and [data preparation](https://marcocamilo.com/resume-analyzer#dataloader). Starting from initialized data loaders, I construct the encoder architecture with an embedding layer, a stack of encoder layers, and a feed-forward neural network for classification. I then initialize the hyperparameters for training and train the model using the Adam optimizer and CrossEntropyLoss criterion. The model's performance is evaluated on the test set using accuracy as the evaluation metric.

> #### Takeaways
>
> -   The Transformer Encoder model achieves an accuracy of 75% on the test set.
> -   This model serves as a robust baseline for transformer-based models in text classification tasks.

#### Import Packages and Initialize DataLoader

The only package we import in addition to those imported in the previous model is the `math` package, a native Python package that offers straightforward and readible functions for mathematical operations like root, logarithms, or exponential functions.

``` python
import math
```

#### (remove this section)

``` python
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

from modules.utils import save_performance
from src.modules.dl import test_model, train_model

data = pd.read_parquet('./data/3-processed/resume-features.parquet', columns=['Category', 'cleaned_resume'])

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print("Using {}.".format(device))


def tokenization(texts, tokenizer_type='basic_english', specials=['<unk>']):
    # Instantiate tokenizer
    tokenizer = get_tokenizer(tokenizer_type)
    # Tokenize text data
    tokens = [tokenizer(text) for text in texts]
    # Build vocabulary
    vocab = build_vocab_from_iterator(tokens, specials=specials)
    # Set default index for unknown tokens
    vocab.set_default_index(vocab['<unk>'])

    # Convert tokenized texts to a tensor
    tokenized_texts = [torch.tensor([vocab[token] for token in text], dtype=torch.int64) for text in tokens] 

    return tokenized_texts, vocab

def tokenization(texts, tokenizer_type='basic_english', specials=['<unk>'], device=device):
    tokenizer = get_tokenizer(tokenizer_type)
    tokens = [tokenizer(text) for text in texts]
    vocab = build_vocab_from_iterator(tokens, specials=specials)
    vocab.set_default_index(vocab['<unk>'])

    tokenized_texts = [torch.tensor([vocab[token] for token in text], dtype=torch.int64, device=device) for text in tokens]

    return tokenized_texts, vocab

class ResumeDataset(Dataset):
    def __init__(self, data, device=device):
        super().__init__()
        self.text = data.iloc[:,1]
        self.labels = torch.tensor(data.iloc[:,0], device=device)
        
        self.tokenized_texts, self.vocab = tokenization(self.text, device=device)

    def __len__(self):
        return len(self.labels)

    def vocab_size(self):
        return len(self.vocab)

    def num_class(self):
        return len(self.labels.unique())

    def __getitem__(self, idx):
        sequence = self.tokenized_texts[idx]
        label = self.labels[idx]
        return sequence, label

def collate_fn(batch, device=device):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return sequences_padded, labels

dataset = ResumeDataset(data)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
```

#### Model Architecture

The Transformer Encoder model consists of an embedding layer, a stack of encoder layers, and a feed-forward neural network for classification. The embedding layer converts the input sequences into dense representations, which are then passed through the encoder layers. Each encoder layer consists of a multi-head self-attention mechanism followed by a residual connection with layer normalization and a feed-forward neural network, followed by another residual connection with layer normalization. The output is finally passed through a feed-forward neural network for classification.

<figure>
<img src="&#39;./notebooks/classification/img/Encoder.jpeg&#39;" alt="Transformer Encoder Model" />
<figcaption aria-hidden="true">Transformer Encoder Model</figcaption>
</figure>

I decide to build the encoder model from scratch, as it allows me to better understand the architecture and the components of the model. I opt for a modular approach, where I construct each component of the model as a separate class and then combine them in the `TransformerEncoder` class.

> **Resources**
> The implementation of this model was in great part inspired by the following resources:
> - [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
> - [Coding a Transformer from Scratch on PyTorch (YouTube)](https://www.youtube.com/watch?v=ISNdQcPhsts)

``` python
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Dimensions of embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Embedding dimension
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Initialize positional embedding matrix (vocab_size, d_model)
        pe = torch.zeros(vocab_size, d_model)
        # Positional vector (vocab_size, 1)
        position = torch.arange(0, vocab_size).unsqueeze(1)
        # Frequency term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model))
        # Sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add batch dimension
        pe = pe.unsqueeze(0)
        # Save to class
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))
        # Numerical stability in case of 0 denominator
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Linear combination of layer norm with parameters gamma and beta
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Layer normalization for residual connection
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return self.dropout(self.norm(x1 + x2))

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # Linear layers and dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float =0.1, qkv_bias: bool = False, is_causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0,  "d_model is not divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.linear = nn.Linear(num_heads * self.head_dim, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape[:2]

        # Linear transformation and split into query, key, and value
        qkv = self.qkv(x)  # (batch_size, seq_length, 3 * embed_dim)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)  # (batch_size, seq_length, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)
        queries, keys, values = qkv  # 3 * (batch_size, num_heads, seq_length, head_dim)

        # Scaled Dot-Product Attention
        context_vec = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, dropout_p=self.dropout, is_causal=self.is_causal)

        # Combine heads, where self.d_model = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        context_vec = self.dropout_layer(self.linear(context_vec))

        return context_vec
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, heads)
        # First residual connection and layer normalization
        self.residual1 = ResidualConnection(d_model, dropout)
        # Feed-forward neural network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # Second residual connection and layer normalization
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, self.self_attn(x, x, x, attn_mask=mask)[0])
        return self.residual2(x, self.feed_forward(x))

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        # Embedding and positional embedding layers
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model, dropout)
        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_layers)])
        # Layer normalization
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, heads: int, d_ff: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        # Encoder model
        self.encoder = Encoder(vocab_size, d_model, n_layers, heads, d_ff, dropout)
        # Feed-forward neural network for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        x = torch.mean(x, dim=1)
        return self.fc(x)
```

#### Hyperparameters and Training

With the model constructed, I initialize the hyperparameters for training. As with the previous model, I obtain the vocabulary size and number of output features from the `ResumeDataset` class. The embedding size is set to 80, with the hidden dimension set to 180. The number of heads for the multi-head attention mechanism is set to 4, and the number of layers for the encoder stack is set to 4. The learning rate is set to 1e-3 and the model is trained for 20 epochs.

``` python
vocab_size = dataset.vocab_size()
d_model = 80
num_heads = 4
hidden_dim = 180
num_layers = 4
out_features = dataset.num_class()
lr = 0.001
epochs = 20
```

I instantiate the model with the hyperparameters and move it to the device. The criterion and optimizer are left unchanged from the previous model, with the optimizer set to the Adam optimizer and the criterion set to the CrossEntropyLoss, suitable for multi-class classification tasks. I also initialize the learning rate scheduler with a patience of 2, to prevent the model from overfitting. The model is then trained using the `train_model` function.

``` python
model = TransformerClassifier(vocab_size, d_model, num_heads, 
                                hidden_dim, num_layers, out_features, dropout=0).to(device)
criterion = CrossEntropyLoss()
loss = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(loss, patience=2)

train_model(model, train_loader, val_loader, epochs, criterion, loss, scheduler)
```

    Epoch 1/25
                                                                 
    Train Loss: 3.2978 | Train Acc: 0.0402
      Val Loss: 3.2317 |   Val Acc: 0.0370
    👉 New best model with val loss: 3.2317!
    ------------------------------
    Epoch 2/25
                                                                 
    Train Loss: 3.1908 | Train Acc: 0.0561
      Val Loss: 3.2332 |   Val Acc: 0.1088
    ------------------------------
    Epoch 3/25
                                                                  
    Train Loss: 2.9856 | Train Acc: 0.1344
      Val Loss: 2.8514 |   Val Acc: 0.1250
    👉 New best model with val loss: 2.8514!
    ------------------------------
    Epoch 4/25
                                                                 
    Train Loss: 2.7657 | Train Acc: 0.1939
      Val Loss: 2.7021 |   Val Acc: 0.2523
    👉 New best model with val loss: 2.7021!
    ------------------------------
    Epoch 5/25
                                                                 
    Train Loss: 2.4915 | Train Acc: 0.2639
      Val Loss: 2.3822 |   Val Acc: 0.3009
    👉 New best model with val loss: 2.3822!
    ------------------------------
    Epoch 6/25
                                                                 
    Train Loss: 2.2619 | Train Acc: 0.3433
      Val Loss: 2.3095 |   Val Acc: 0.3079
    👉 New best model with val loss: 2.3095!
    ------------------------------
    Epoch 7/25
                                                                 
    Train Loss: 2.0741 | Train Acc: 0.3849
      Val Loss: 2.3077 |   Val Acc: 0.3449
    👉 New best model with val loss: 2.3077!
    ------------------------------
    Epoch 8/25
                                                                  
    Train Loss: 1.8948 | Train Acc: 0.4335
      Val Loss: 1.9932 |   Val Acc: 0.3981
    👉 New best model with val loss: 1.9932!
    ------------------------------
    Epoch 9/25
                                                                 
    Train Loss: 1.8089 | Train Acc: 0.4559
      Val Loss: 1.9496 |   Val Acc: 0.4398
    👉 New best model with val loss: 1.9496!
    ------------------------------
    Epoch 10/25
                                                                 
    Train Loss: 1.7302 | Train Acc: 0.4921
      Val Loss: 1.9235 |   Val Acc: 0.4144
    👉 New best model with val loss: 1.9235!
    ------------------------------
    Epoch 11/25
                                                                 
    Train Loss: 1.5059 | Train Acc: 0.5635
      Val Loss: 1.7216 |   Val Acc: 0.5139
    👉 New best model with val loss: 1.7216!
    ------------------------------
    Epoch 12/25
                                                                 
    Train Loss: 1.3984 | Train Acc: 0.5903
      Val Loss: 1.6969 |   Val Acc: 0.5162
    👉 New best model with val loss: 1.6969!
    ------------------------------
    Epoch 13/25
                                                                 
    Train Loss: 1.3385 | Train Acc: 0.6066
      Val Loss: 1.6511 |   Val Acc: 0.5440
    👉 New best model with val loss: 1.6511!
    ------------------------------
    Epoch 14/25
                                                                 
    Train Loss: 1.2355 | Train Acc: 0.6339
      Val Loss: 1.6313 |   Val Acc: 0.5602
    👉 New best model with val loss: 1.6313!
    ------------------------------
    Epoch 15/25
                                                                  
    Train Loss: 1.1960 | Train Acc: 0.6463
      Val Loss: 1.5931 |   Val Acc: 0.5440
    👉 New best model with val loss: 1.5931!
    ------------------------------
    Epoch 16/25
                                                                 
    Train Loss: 1.0905 | Train Acc: 0.6716
      Val Loss: 1.4621 |   Val Acc: 0.5694
    👉 New best model with val loss: 1.4621!
    ------------------------------
    Epoch 17/25
                                                                 
    Train Loss: 1.0535 | Train Acc: 0.6736
      Val Loss: 1.4710 |   Val Acc: 0.6111
    ------------------------------
    Epoch 18/25
                                                                 
    Train Loss: 1.0474 | Train Acc: 0.6860
      Val Loss: 1.6559 |   Val Acc: 0.5532
    ------------------------------
    Epoch 19/25
                                                                 
    Train Loss: 1.0170 | Train Acc: 0.6791
      Val Loss: 1.3482 |   Val Acc: 0.6366
    👉 New best model with val loss: 1.3482!
    ------------------------------
    Epoch 20/25
                                                                 
    Train Loss: 0.9486 | Train Acc: 0.7143
      Val Loss: 1.4318 |   Val Acc: 0.5810
    ------------------------------
    Epoch 21/25
                                                                  
    Train Loss: 0.9223 | Train Acc: 0.7202
      Val Loss: 1.5865 |   Val Acc: 0.5694
    ------------------------------
    Epoch 22/25
                                                                  
    Train Loss: 0.9165 | Train Acc: 0.7267
      Val Loss: 1.4348 |   Val Acc: 0.6273
    ------------------------------
    Epoch 23/25
                                                                 
    Train Loss: 0.5495 | Train Acc: 0.8388
      Val Loss: 1.2119 |   Val Acc: 0.6898
    👉 New best model with val loss: 1.2119!
    ------------------------------
    Epoch 24/25
                                                                 
    Train Loss: 0.3607 | Train Acc: 0.9167
      Val Loss: 1.1326 |   Val Acc: 0.7269
    👉 New best model with val loss: 1.1326!
    ------------------------------
    Epoch 25/25
                                                                 
    Train Loss: 0.2782 | Train Acc: 0.9395
      Val Loss: 1.1033 |   Val Acc: 0.7315
    👉 New best model with val loss: 1.1033!
    ------------------------------
    Best model saved with val loss: 1.1033
    ✅ Training complete!

...

#### Evaluation

``` python
accuracy = test_model(model, test_loader, criterion)
```

    Test Loss: 1.2708 | Test Acc: 0.7361
    ✅ Testing complete!

...

As with the previous model, I save the performance metrics for later analysis.

``` python
save_performance(model_name='Transformer',
                 architecture='embed_layer->encoder->linear_layer',
                 embed_size='64',
                 learning_rate='1e-3',
                 epochs='20',
                 optimizer='Adam',
                 criterion='CrossEntropyLoss',
                 accuracy=80
                 )
```

