---
title: BERT
category: NLP
type: Text Classification
execute:
  eval: false
---


### BERT

The last model is also an encoder-based model called the Bidirectional Encoder Representations from Transformers (BERT). BERT is a pre-trained transformer-based model that can be fine-tuned for a wide range of NLP tasks by adding task-specific output layers. The model is bidirectional, meaning that it can take into account the context of a word by looking at both the left and right context. This allows the model to capture a wider range of contextual information, which is particularly useful for tasks such as text classification. This allows the model to capture rich semantic relationships and dependencies within text sequences, which is particularly useful for tasks such as text classification.

As with previous PyTorch models, I create an iterable dataset using the `Dataset` and `DataLoader` classes, to tokenize the resumes using the BERT tokenizer, pad sequences to equal lengths, split the data and batch the data for training. I then construct the model architecture, consisting of the pre-trained BERT base model, an added dropout layer and a linear output layer for classification. I initialize the hyperparameters and train the model using Cross Entropy Loss along with the Adam optimizer. The model is evaluated as with other deep learning models using accuracy.

#### Import Packages

In addition to the standard deep learning packages used so far, I import three classes from the `transformers` package:

-   `BertModel`: loads the pre-trained BERT model.
-   `BertTokenizer`: constructs a BERT tokenizer.
-   `DataCollatorWithPadding`: builds a batch with dynamically padded sequences.

Because of BERT's output format, I also import a custom `train_BERT` and `test_BERT` function, which are specifically tailored to return the model's training and test performances using BERT's output, including input IDs and attention masks.

{: #open}

``` python
from transformers import BertModel, BertTokenizer, DataCollatorWithPadding

from src.modules.dl import train_BERT, test_BERT
```

#### Dataset and DataLoader

Before creating the dataset and dataloader, I initizalize the tokenizer and define the pre-trained BERT model. I then create the `ResumeBertDataset`, which tokenizes resumes and prepares them for model input.

In contrast to the previous model, I configure the tokenizer using the `.encode_plus` method, which returns a dictionary of the batch encodings, including tokenized input sequences and attention masks. I set the `padding` parameter to `False` to avoid padding, as this will be handled dynamically by the data collator. Additionally, I set `truncation` to `True` to truncate sequences that exceed the maximum length. The method also adds the special tokens `[CLS]` and `[SEP]` to the input sequences, required by BERT. I set the `return_tensors` parameter to `'pt'` to return PyTorch tensors. Finally, I return a dictionary with the processed input sequences, attention masks, and labels.

{: #open}

``` python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ResumeBertDataset(Dataset):
    def __init__(self, data, max_length, tokenizer=tokenizer, device=device):
        super().__init__()
        self.texts = data.iloc[:,1].values
        self.labels = torch.tensor(data.iloc[:,0])
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.labels)

    def num_class(self):
        return len(self.labels.unique())

    def __getitem__(self, idx):
        resumes = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            resumes,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
```

I initialize the dataset and set `max_length` to 512, which is the maximum number of tokens that BERT can process. I then split the dataset into 70% training, 15% validation, and 15% test sets using the `random_split` function. I use the `DataCollatorWithPadding` class to dynamically pad sequences to the maximum length in each batch. Finally, I create dataloaders for the training, validation, and test sets using the `DataLoader` class, setting the batch size to 16, shuffling the data, and assigning the data collator.

{: #open}

``` python
dataset = ResumeBertDataset(data, max_length=512)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
```

#### Model Architecture

The `BertResumeClassifier` model consists of the pre-trained BERT base model, a dropout layer, and a linear output layer to classify the resumes. The BERT model uses the `bert-base-uncased` pre-trained model to generate contextual embeddings from the input sequences. I extract the embeddings by indexing the `pooler_output` key from the output dictionary. These embeddings are then passed through a dropout layer to prevent overfitting, and subsequently fed into a fully connected linear layer which maps the embeddings to the desired number of output classes for classification.

{: #open}

``` python
class BertResumeClassifier(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.01):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )['pooler_output']
        output = self.dropout(pooled_output)
        output = self.out(output)
        return output
```

#### Hyperparameters and Training

Since BERT is a pre-trained model and it accepts a fixed input size of 512 tokens, the are less parameters to set from the model itself. The only parameter that needs to be set is the number of classes---which, as before, is obtained from the `Dataset` class.

I initialize the model, loss function, optimizer, and number of epochs. I use as before the Cross Entropy Loss function and the Adam optimizer, although this time with a learning rate of 2e-5, since it seemed to result in better performance. I train the model for only 10 epochs.

``` python
n_classes = dataset.num_class()

model = BertResumeClassifier(n_classes).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=2e-5)
epochs = 10

train_BERT(model, train_loader, val_loader, epochs, criterion, optimizer)
```

    accuracy
      training           (min:    0.050, max:    0.991, cur:    0.991)
      validation         (min:    0.120, max:    0.933, cur:    0.933)
    log loss
      training           (min:    0.081, max:    3.189, cur:    0.081)
      validation         (min:    0.414, max:    3.002, cur:    0.428)
    ------------------------------
    Best model saved:
    Val Loss: 0.4143 | Val Acc: 0.9190
    ✅ Training complete!

The BERT model shows a significant and steady decrease in both training and validation losses over the 10 epochs, indicating effective learning of data patterns. By the end of training, the model's train loss decreased from 3.1394 to 0.0589, while the validation loss decreased from 2.7347 to 0.5070. This consistent reduction highlights the model's ability to capture and generalize the data without overfitting, as demonstrated by the small gap between the training and validation losses by the end of the 10th epoch.

Regarding accuracy, the training and validation accuracies also show a steady increase over the 10 epochs. The training accuracy increased from 0.0774 to 0.9950, with validation accuracy improving from 0.3472 to 0.9028. This indicates that the model effectively learned the patterns in the data and generalizes well to unseen data. The small difference between the final training and validation accuracies demonstrates the model's robustness and ability to avoid overfitting, ensuring reliable performance on new data. The best saved model has a validation loss of 0.5070 and an accuracy of 0.9028.

#### Evaluation

As before, I evaluate the model using the `test_model` function using the best saved model.

``` python
accuracy = test_BERT(model, test_loader, criterion)
```

    Test Loss: 0.3984 | Test Acc: 0.9167
    ✅ Testing complete!

The BERT model achieved an impressive performance on the test set, reaching an accuracy of 91.67% with a test loss of 0.3984. This significantly outperforms all previous models tested so far. As expected from the training and validation performances, the model is robust and generalizes very well to unseen data. This is further confirmed by the very close alignement between test and validation accuracies, both of which represent datasets not previously seen by the model. The close accuracy between the validation data and the test data shows the model is capable to generalize to new resumes and effectively classify them into the correct categories.

As with previous models, I save its performance using the `save_performance` function.

{: #open}

``` python
save_performance(model_name='BERT',
                 architecture='bert-base-uncased>dropout->linear_layer',
                 embed_size='768',
                 learning_rate='2e-5',
                 epochs='10',
                 optimizer='Adam',
                 criterion='CrossEntropyLoss',
                 accuracy=91
                 )
```
