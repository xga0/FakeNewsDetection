#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fake News Detection using PyTorch

Implementation of an LSTM-based model for fake news classification using GloVe embeddings.
Features: two-layer LSTM, dropout regularization, tqdm progress bars, and early stopping.

Outputs in 'results' directory:
- training_history.png
- roc_curve.png
- classification_metrics.txt

Author: seangao
"""

import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
import string
import os
from datetime import datetime
from tqdm import tqdm

# Create output directory for plots and metrics
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATA LOAD
fakepath = 'Fake.csv'
truepath = 'True.csv'

fake = pd.read_csv(fakepath)
true = pd.read_csv(truepath)

# Fix pandas warnings by using .copy()
df_fake = fake[['title', 'text']].copy()
df_true = true[['title', 'text']].copy()

df_fake.loc[:, 'label'] = 0
df_true.loc[:, 'label'] = 1
df_true.loc[:, 'text'] = df_true['text'].apply(lambda x: x[x.find('-')+1:])  # REMOVE Reuters INFO

df = pd.concat([df_true, df_fake], ignore_index=True)

# PREPROCESSING
df['text'] = df['text'].apply(lambda x: x.strip())

# Fix regex patterns by using raw strings
df['title'] = df['title'].apply(lambda x: re.sub(r'\[[^]]*\]', '', x))
df['text'] = df['text'].apply(lambda x: re.sub(r'\[[^]]*\]', '', x))

df['title'] = df['title'].apply(lambda x: re.sub(r'[()]', '', x))
df['text'] = df['text'].apply(lambda x: re.sub(r'[()]', '', x))

stpw = stopwords.words('english')

def stpwfix(input_string):
    output_string = []
    for i in input_string.split():
        if i.strip().lower() not in stpw:
            output_string.append(i.strip())
    return ' '.join(output_string)

df['title'] = df['title'].apply(lambda x: stpwfix(x))
df['text'] = df['text'].apply(lambda x: stpwfix(x))

def addperiod(input_string):
    if not input_string.endswith('.'):
        input_string += '.'
        return input_string
    else:
        return input_string

df['title'] = df['title'].apply(lambda x: addperiod(x))

df['concat'] = df[['title', 'text']].agg(' '.join, axis=1)
df['concat'] = df['concat'].apply(lambda x: re.sub(r' +', ' ', x))

# PREPARE GLOVE EMBEDDING
embed_size = 50
maxlen = 1000

# Create vocabulary
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common())}
    return vocab

vocab = build_vocab(df['concat'].values)

# Convert texts to sequences
def text_to_seq(text, vocab, maxlen):
    seq = [vocab.get(word, 0) for word in text.split()]
    if len(seq) < maxlen:
        seq = seq + [0] * (maxlen - len(seq))
    else:
        seq = seq[:maxlen]
    return seq

X = np.array([text_to_seq(text, vocab, maxlen) for text in df['concat'].values])
y = df['label'].to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_path = 'glove.6B.50d.txt'

embeddings_index = {}
with open(embedding_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((len(vocab) + 1, embed_size))
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert data to PyTorch tensors
X_train = torch.LongTensor(X_train).to(device)
X_test = torch.LongTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Create custom dataset
class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loaders
train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class FakeNewsDetector(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, embed_size):
        super(FakeNewsDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Freeze embedding layer
        
        self.lstm = nn.LSTM(embed_size, 128, batch_first=True, dropout=0.1, num_layers=2)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = torch.max(lstm_out, dim=1)[0]  # Global max pooling
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize model
model = FakeNewsDetector(len(vocab) + 1, embedding_matrix, embed_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), acc=correct/total)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for validation
        val_loop = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, labels in val_loop:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), acc=correct/total)
        
        val_loss = val_loss / len(test_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if epoch > 0 and val_loss > val_losses[-2]:
            print('Early stopping triggered')
            break
    
    return train_losses, train_accs, val_losses, val_accs

# Train the model
train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, criterion, optimizer)

# Plot and save training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accs, 'go-', label='Training Accuracy')
plt.plot(val_accs, 'ro-', label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(train_losses, 'go-', label='Training Loss')
plt.plot(val_losses, 'ro-', label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(f'{output_dir}/training_history.png')
plt.close()

# Evaluate on test set
model.eval()
y_pred = []
y_pred_proba = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        y_pred.extend(predicted.cpu().numpy())
        y_pred_proba.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate and save ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'{output_dir}/roc_curve.png')
plt.close()

# Calculate and save classification metrics
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
classification_metrics = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])

# Save metrics to file
with open(f'{output_dir}/classification_metrics.txt', 'w') as f:
    f.write('Classification Metrics\n')
    f.write('====================\n\n')
    f.write(f'Confusion Matrix:\n{cm}\n\n')
    f.write(f'True Negatives: {tn}\n')
    f.write(f'False Positives: {fp}\n')
    f.write(f'False Negatives: {fn}\n')
    f.write(f'True Positives: {tp}\n\n')
    f.write(f'ROC AUC: {roc_auc:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(classification_metrics)

print(f'\nResults saved in {output_dir}/ directory:')
print('- training_history.png')
print('- roc_curve.png')
print('- classification_metrics.txt') 