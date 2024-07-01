---
title: Feedforward Neural Network
category: NLP
type: Text Classification
execute:
  eval: false
---


### Feedforward Neural Network

The next model is a Feedforward Neural Network (FNN){% sidenote 'Feedforward Neural Networks (FNN) are a type of neural networks where information moves in only one direction—forward—from the input nodes, through hidden layers, and to the output nodes' '' %} built using PyTorch. I create an iterator for the dataset using the `DataLoader` class, which tokenizes and numericalized the resumes, dynamically pads the sequences, and batches the data for training, saving memory and computation time. Then, I construct a simple neural network architecture with an embedding layer, followed by three fully connected layers with ReLU activation functions. The model is trained using the Adam optimizer and CrossEntropyLoss criterion, and its performance is evaluated on the test set using accuracy as the evaluation metric.

> #### Takeaways
>
> -   The Feedforward Neural Network achieved an accuracy of 73.15% with a loss of 1.1444 on the test set.
> -   Performance suggests minimal overfitting, given the small gap between training and validation.
> -   Model demonstrates robust generalization, with test accuracy aligning closely with validation accuracy.

#### Import Packages and Data

Aside from the standard PyTorch and pandas imports, I also import three custom functions:  


- `train_model`: trains the model based on the hyperparameters and data provided, prints the training and validation loss and accuracy in real-time, and saves the best model based on the iteration with the lowest validation loss. It also provides an option to visualize the training progress using the `PlotLosses` library or `matplotlib`. 
- `test_model`: evaluates the model on the test set using the best model saved during training and returns the testing accuracy.
- `save_performance`: saves the performance metrics of the model to a json file for future analysis.
{: #open}

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
from livelossplot import PlotLosses
from tqdm import tqdm

from src.modules.dl import train_model, test_model
from src.modules.utils import save_performance

data = pd.read_parquet('./data/3-processed/resume-features.parquet', columns=['Category', 'cleaned_resumes'])
```

When training deep learning models, I always code the option to use a GPU if available and set the `device` variable accordingly. This not only allows the model to leverage the parallel computing if available, but also makes the code reproducible across different device setups. The snippet below checks if a GPU is available and sets the device variable accordingly, which is later used by the `DataLoader` and model.
{: #open}

``` python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print("Using {}.".format(device))
```

    Using cuda.

#### Dataset and DataLoader

Before constructing the `Dataset` class, I define a `tokenization` function that instantiates the tokenizer, tokenizes the text data, and builds a vocabulary using PyTorch’s `get_tokenizer` and `build_vocab_from_iterator` functions. The function returns the tokenized texts to be indexed by the `DataLoader` during training, and the vocabulary, which will be used to determine the vocabulary size.
{: #open}

``` python
def tokenization(texts, tokenizer_type='basic_english', specials=['<unk>'], device=device):
    # Instantiate tokenizer
    tokenizer = get_tokenizer(tokenizer_type)
    # Tokenize text data
    tokens = [tokenizer(text) for text in texts]
    # Build vocabulary
    vocab = build_vocab_from_iterator(tokens, specials=specials)
    # Set default index for unknown tokens
    vocab.set_default_index(vocab['<unk>'])

    # Convert tokenized texts to a tensor
    tokenized_texts = [torch.tensor([vocab[token] for token in text], dtype=torch.int64, device=device) for text in tokens]

    return tokenized_texts, vocab
```

Next. I construct the `ResumeDataset` iterator, which preprocesses the text data using the `tokenization` function and indexes samples for the `DataLoader` during training. The `__len__` method returns the length of the dataset, the `vocab_size` method returns the size of the vocabulary, the `num_class` method returns the number of unique classes in the dataset, and the `__getitem__` method returns a sample of text and label from the dataset.
{: #open}

``` python
class ResumeDataset(Dataset):
    # Dataset initialization and preprocessing
    def __init__(self, data):
        # Initialize dataset attributes
        super().__init__()
        self.text = data.iloc[:,1]
        self.labels = data.iloc[:,0]
        
        self.tokenized_texts, self.vocab = tokenization(self.text)

    # Get length of dataset
    def __len__(self):
        return len(self.labels)

    # Get vocabulary size
    def vocab_size(self):
        return len(self.vocab)

    # Get number of classes
    def num_class(self):
        return len(self.labels.unique())

    # Get item from dataset
    def __getitem__(self, idx):
        sequence = self.tokenized_texts[idx]
        label = self.labels[idx]
        return sequence, label
```

I also define a `collate_fn` function to use dynamic padding when batching the data. Dynamic padding is a technique used to pad sequences to the length of the longest sequence in a batch, as opposed to the longest sequence in the entire dataset. Because the model expects uniform dimensions to perform operations, sequences need to be padded to ensure each one has the same length. Dynamic padding allows the model to process sequences of varying lengths more efficiently, saving memory and computation time.
{: #open}

``` python
def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Pad sequences to the longest sequence in the batch
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels
```

Finally, I instantiate the `ResumeDataset` class and split the dataset into 70% training, 15% validation, and 15% test sets using the `random_split` function. I create `DataLoader` iterators for each set, using the `collate_fn` function to apply dynamic padding to the sequences.
{: #open}

``` python
dataset = ResumeDataset(data)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
```

#### Model Architecture

The model is a simple Feedforward Neural Network with an embedding layer, followed by two fully connected layers with ReLU activation functions, and a final fully connected layer with the number of classes as the output size. The model architecture is defined in the `SimpleNN` class, which takes the vocabulary size, embedding size, number of classes as parameters. The `expansion_factor` is defined to determine the hidden dimension size, here set to 2.

The `EmbeddingBag` function efficiently computes the embeddings by performing a two-step operation: first, it creates embeddings for the input indices, adn then reduces the embedding output using the mean across the sequence dimension. This is useful for sequences of varying lengths, as it allows the model to process them more efficiently.
{: #open}

``` python
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=False)
        self.hidden_dim = embed_size * expansion_factor
        self.layer1 = nn.Linear(embed_size, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, embed_size)
        self.layer3 = nn.Linear(embed_size, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
```

#### Hyperparameters and Training

Before training, I set the hyperparameters for training the neural network. The vocabulary size and number of classes are obtained from the `ResumeDataset` class.The embedding size is set to 60 and the learning rate is set to 1e-3. The model is trained for 40 epochs. 
{: #open}

``` python
vocab_size = dataset.vocab_size()
num_class = dataset.num_class()
embed_size = 60
lr=0.001
epochs = 40
```

I then instantiate the model, sending it to the available device, and define the loss function and optimizer. The loss function is set to CrossEntropyLoss, which is suitable for multi-class classification tasks. The optimizer is Adam, an adaptive learning rate optimization algorithm well-suited for training deep neural networks. In addition, I define a learning rate scheduler that reduces the learning rate by a factor of 0.1 if the validation loss does not improve for `patience` number of epochs. This prevents the model from overfitting and improves generalization. The model and hyperparameters are then passed for training using the `train_model` function{% sidenote 'During fine-tunning, I found the model converged with better accuracy when using a dropout rate of 0.4.' '' %}.

To visualize the training progress, I set the `visualize` parameter to 'liveloss', which uses the [`PlotLosses` library](https://p.migdal.pl/livelossplot/) to create a dynamicallly updating plot that visulalized the training and validation loss and accuracy in real-time. This allows me to monitor the model's performance and make adjustments to the hyperparameters if necessary.

``` python
model = SimpleNN(vocab_size, embed_size, num_class, dropout=0.4).to(device)
criterion = CrossEntropyLoss()
loss = Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(loss, patience=2)

train_model(model, train_loader, val_loader, epochs, criterion, loss, scheduler, visualize='liveloss')
```

    accuracy
      training           (min:    0.036, max:    0.712, cur:    0.707)
      validation         (min:    0.037, max:    0.694, cur:    0.685)
    log loss
      training           (min:    0.873, max:    3.187, cur:    0.921)
      validation         (min:    1.274, max:    3.184, cur:    1.330)
    ------------------------------
    Best model saved:
    Val Loss: 1.3300 | Val Acc: 0.6852
    ✅ Training complete!

The performance plots reveal interesting patterns about the model’s performance. Starting with the loss, the training and validation losses decrease steadily until 30 epochs, showing that the model is effectively learning the patterns in the data. After this, the training loss continues to decrease, while the validation loss plateaus. However, the model finishes with only a small gap between the training and validation loss, with the training loss at 0.92 and the validation loss at 1.33. The small size of the gap indicates that the model is not overfitting the training data and generalizes well to unseen data.

Regarding accuracy, the training and validation accuracy increase steadily until 40 epochs, after which they converge. At convergence, the training accuracy is slightly higher than the validation accuracy, with the model achieving a training accuracy of 71% and a validation accuracy of 69%. This small difference between the training and validation accuracy indicates that the model does not overfit the training data and generalizes well to unseen data. The best saved model has a validation loss of 1.33 and a validation accuracy of 68.52%.

#### Evaluation

After training the model, I evaluate its performance on the test set using the `test_model` function. The function takes the trained model, test data loader, and criterion as input and returns the accuracy of the model on the test set.

``` python
accuracy = test_model(model, test_loader, criterion)
```

    Test Loss: 1.1444 | Test Acc: 0.7315
    ✅ Testing complete!

The Feedforward Neural Network model achieves an accuracy of **73.15%** and a loss of 1.1444 on the test set. As expected from the training and validation plots, the model performs reasonably well on unseen data, with the test accuracy aligning closely with the validation accuracy observed during training. The consistency between validation and test accuracies suggests that the model successfully generalizes to new data and demonstrates robustness in its predictions. 

Compared to the baseline model, the Feedforward Neural Network model achieved a lower accuracy on the test set. However, the model’s performance is still quite impressive considering the simplicity of the architecture. The model’s performance can likely be improved by introducing more advanced regularization techniques, improving the quality of embeddings{% sidenote 'Examples of pre-trained embeddings include Word2Vec, GloVe, and FastText' '' %}, or increasing the complexity of the model architecture. This last point is what I explore in the next section, where I implement a more advanced model architecture using a Transformer-based neural network that leverages self-attention mechanisms to capture long-range dependencies in the data.
To conclude this section, I save the performance metrics of the Feedforward Neural Network model for later analysis.
{: #open}

``` python
save_performance(model_name='Feedforward Neural Network',
                 architecture='embed_layer->dropout->120->dropout->60->dropout->num_classes',
                 embed_size='60',
                 learning_rate='1e-3',
                 epochs='50',
                 optimizer='Adam',
                 criterion='CrossEntropyLoss',
                 accuracy=73.15,
                 )
```
