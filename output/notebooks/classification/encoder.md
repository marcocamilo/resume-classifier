---
title: Encoder Model
execute:
  eval: false
---


### Encoder Model

The following model utilizes the encoder component of the Transformer architecture for text classification. Unlike decoders, which convert dense representations into output sequences, encoders instead transform input sequences into dense representations. Their ability to extract sequence information and convert them to a dense representation makes them particularly useful for tasks such as sentiment analysis, named entity recognition, and text classification.

The encoder model follows the Transformer architecture described in *Attention is All You Need*[^enc-1] and is used as a baseline for transformer-based models. I construct the encoder architecture with an embedding layer, a stack of encoder layers, and a feed-forward neural network for classification. I then initialize the hyperparameters for training and train the model using the Adam optimizer and CrossEntropyLoss criterion. The model's performance is evaluated on the test set using accuracy as the evaluation metric.

The imported packages as well as the step in building the `DataLoader` are the same as those for the Feedforward Neural Network model (see [packages](https://marcocamilo.com/resume-analyzer#import-packages-and-data-1) and [data preparation](https://marcocamilo.com/resume-analyzer#dataloader). Therefore, I skip directly to the model architecture.

> #### Takeaways
>
> -   The Transformer Encoder model achieves an accuracy of 75% on the test set.
> -   This model serves as a robust baseline for transformer-based models in text classification tasks.

#### Model Architecture

The Encoder model consists of an embedding layer, a stack of encoder layers, and a fully connected neural network for classifying the outputs. The embedding layer converts the input sequences into dense representations, which are then passed through the encoder layers. Each encoder layer consists of a multi-head self-attention mechanism[^enc-2] followed by a residual connection with layer normalization[^enc-3] and a feed-forward neural network, followed by another residual connection with layer normalization. The output is finally passed through a feed-forward neural network for classification.

<figure>
<img src="&#39;./notebooks/classification/img/Encoder.jpeg&#39;" alt="Transformer Encoder Model" />
<figcaption aria-hidden="true">Transformer Encoder Model</figcaption>
</figure>

I decide to build the encoder model from scratch, as it allows me to better understand the architecture and the components of the model. I opt for a modular approach, where I construct each component of the model as a separate class and then combine them in the `TransformerEncoder` class.

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
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Multi-head self-attention mechanism
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # First residual connection and layer normalization
        self.residual1 = ResidualConnection(d_model, dropout)
        # Feed-forward neural network
        self.feed_forward = FeedForward(d_model, hidden_dim, dropout)
        # Second residual connection and layer normalization
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, self.multihead_attention(x, mask))
        x = self.residual2(x, self.feed_forward(x))
        return x

class EncoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, hidden_dim: int, num_layers: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(vocab_size, d_model, dropout)
        self.encoder = EncoderStack(d_model, num_heads, hidden_dim, num_layers, dropout)
        self.classifier = nn.Linear(d_model, out_features)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_embedding(x)
        x = self.encoder(x, mask)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
```

> **Further Reading:**
>
> The implementation of this model was in great part inspired by the following resources:
> - [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
> - [Coding a Transformer from Scratch on PyTorch (YouTube)](https://www.youtube.com/watch?v=ISNdQcPhsts)
> - [Text Classification with Transformer Encoders](https://towardsdatascience.com/text-classification-with-transformer-encoders-1dcaa50dabae)

#### Hyperparameters and Training

With the model constructed, I initialize the hyperparameters for training. As with the previous model, I obtain the vocabulary size and number of output features from the `ResumeDataset` class. The embedding size is fixed at 80, while the hidden dimension is set to 180. For the multi-head attention mechanism, I use 4 heads, with an the encoder stack comprising of 4 layers. I train the model for 20 epochs at a learning rate of 1e-3.

{: #open}

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
loss = Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(loss, patience=2)

train_model(model, train_loader, val_loader, epochs, criterion, loss, scheduler)
```

    accuracy
      training           (min:    0.043, max:    1.000, cur:    1.000)
      validation         (min:    0.035, max:    0.794, cur:    0.785)
    log loss
      training           (min:    0.027, max:    3.276, cur:    0.029)
      validation         (min:    1.023, max:    3.273, cur:    1.071)
    ------------------------------
    Best model saved:
    Val Loss: 1.0709 | Val Acc: 0.7847
    ✅ Training complete!

According to the performance plots, despite the encoder model achieving a better validation accuracy than the Feedforward Neural Network, the model exhibits a large gap between the training and validation performance, indicating that the model is overfitting. The model rapidly decreases the training and validation losses, and rapidly incrases their accuracies during the first 14 epochs. For the following 6 epochs, however, the training loss continues to rapidly decrease from 1.25 to near 0, while the validation loss stagnates at around 1.07. Similarly, the training accuracy rapidly increases to 100%, while the validation accuracy remains at 78%. The model converges 20 epochs in with a significant gap between training and validation models, indicating that the model is overfitting.

Nevertheless, the model achieves a validation accuracy of 78%, which is slightly higher than the Feedforward Neural Network model. I decide to evaluate the model on the test set to obtain the final accuracy.

#### Evaluation

I evaluate the model on the test set using the `test_model` function, which returns the accuracy of the model on the test set.

``` python
accuracy = test_model(model, test_loader, criterion)
```

    Test Loss: 1.2526 | Test Acc: 0.7454
    ✅ Testing complete!

Despite having implemented a more advanced model architecture with the addition of the multi-head self-attention mechanism, the encoder model achieves an accuracy of 74.5%, similar to the Feedforward Neural Network model. As discussed earlier, the model exhibits a large gap between the training and validation performance, which could be hindering the model's generalization capabilities.

Given a better training and validation performance, the model could potentially achieve a higher accuracy on the test set. The model's performance could likely be improved by means of data augmentation, modifying hyperparameters such as embedding size, hidden dimension, and number of layers, or by using a different optimizer or learning rate scheduler.

As with the previous models, I save the performance metrics for later analysis.

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

[^enc-1]: Vaswani, A. et al. (2017) ['Attention Is All You Need'](http://arxiv.org/abs/1706.03762), Advances in neural information processing systems, 30.

[^enc-2]: The multi-head self-attention mechanism is a component of the transformer model that weights the importance of different elements in a sequence by computing attention scores multiple times in parallel across different linear projections of the input.

[^enc-3]: Layer normalization is a normalization technique that uses the mean and variance statistics calculated across all features.
