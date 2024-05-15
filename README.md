# Sequence Labeling

## Introduction
In this assignment, we focus on the MeasEval shared task from SemEval-2021, which involves extracting counts, measurements, and related context from scientific documents. Specifically, we concentrate on the Quantity recognition step, aiming to identify spans containing quantities within scientific texts.

## Approach
We develop a Recurrent Neural Network (RNN) using Keras, a high-level Deep Learning API, to tackle this task. The RNN architecture consists of an Embedding layer, a Bidirectional LSTM layer, and a TimeDistributed Dense layer.

### Libraries Used
```python
import pandas as pd
import numpy as np
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from tensorflow.keras.initializers import Constant
from sklearn.metrics import classification_report
```
### Downloading GloVe (Terminal command)
```bash
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri "http://nlp.stanford.edu/data/glove.6B.zip" -OutFile glove.6B.zip
Expand-Archive -LiteralPath .\glove.6B.zip -DestinationPath glove
```


## Reproducibility
To ensure reproducibility across different runs, we set a fixed random seed using the following code:
```python
seed = 42
set_random_seed(seed)
```

## Hyperparameters
We define the following hyperparameters for model configuration:

maxlen = 130: Maximum length of input sequence
epochs = 6: Number of training epochs
batch_size = 64: Batch size for gradient updates
embedding_dim = 300: Dimension of embeddings
rnn_units = 256: Number of units per RNN layer
Data Pre-processing
We perform several pre-processing steps to prepare the data for training the model.

## Loading Data
We load the training, development, and test datasets into Pandas DataFrames.
```python
train_data = load_data("data/train.tsv", shrink_dataset, seed)
dev_data = load_data("data/trial.tsv", shrink_dataset, seed)
test_data = load_data("data/eval.tsv", shrink_dataset, seed)
```

## Integrate Sentences
We aggregate the data corresponding to each sentence, resulting in a DataFrame with lemmas and labels for each sentence.
```python
train_examples = integrate_sentences(train_data)
dev_examples = integrate_sentences(dev_data)
test_examples = integrate_sentences(test_data)
```

## Data Formatting
We format the input and output sequences by translating lemmas and labels into indices and padding the sequences to a fixed length.
```python
x_train, y_train = format_examples(train_examples, word2idx, label2idx, maxlen)
x_dev, y_dev = format_examples(dev_examples, word2idx, label2idx, maxlen)
x_test, y_test = format_examples(test_examples, word2idx, label2idx, maxlen)
```

## Recurrent Neural Network
We construct a Sequential model consisting of an Embedding layer, a Bidirectional LSTM layer, and a TimeDistributed Dense layer.
```python
model = create_model(vocab_size, label_size, maxlen, embedding_dim, rnn_units)
```

### Training the Model
We train the model using the training data and evaluate its performance on the development data.
```python
train_model(model, x_train, y_train, x_dev, y_dev, batch_size, epochs)
```
### Making Predictions
We use the trained model to make predictions on the test data.
```python
predictions = make_predictions(model, x_test, batch_size)
```
### Evaluation
We evaluate the model's performance using classification metrics.
```python
evaluate(test_data)
```

## Pre-trained Word Embeddings
We experiment with initializing the Embedding layer with pre-trained GloVe embeddings to potentially improve model performance.

### Loading GloVe Embeddings
We load pre-trained GloVe embeddings into a dictionary.
```python
embedding_index = load_embeddings(glove_path)
```

### Creating Embedding Matrix
We create an embedding matrix to initialize the Embedding layer with GloVe embeddings.
```python
embedding_matrix = create_embedding_matrix(embedding_index, word2idx, vocab_size, embedding_dim)
```

### Creating Model with GloVe Embeddings
We create a new model using the embedding matrix initialized with GloVe embeddings.
```python
model_with_embeddings = create_model_with_embeddings(vocab_size, label_size, maxlen, embedding_dim, rnn_units, embedding_matrix)
```

### Training and Evaluation with GloVe Embeddings
We train and evaluate the model using GloVe embeddings.
```python
train_model(model_with_embeddings, x_train, y_train, x_dev, y_dev, batch_size, epochs)
predictions_with_embeddings = make_predictions(model_with_embeddings, x_test, batch_size)
test_data['prediction'] = predictions_to_labels(predictions_with_embeddings, x_test, labels)
evaluate(test_data)
```

## Conclusion
Through this assignment, we develop and evaluate a deep learning model for Quantity recognition in scientific texts. By experimenting with pre-trained word embeddings, we aim to enhance the model's performance and contribute to the advancement of natural language processing tasks in scientific domains.
