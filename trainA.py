import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch import optim
import wandb
import csv 
import argparse
import time
import math

# Define global variables and constants
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 31  # Maximum length of a word

# Initialize device
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global tracking variables
filedata = []
attention_weights = []
xlabels = []
ylabels = []

# Configuration paths
class Config:
    def __init__(self, lang_code="te"):
        self.trainpath = f"/Users/ujjwalsingh/Development/dl/DA6401_A3/lexicons/{lang_code}.translit.sampled.train.tsv"
        self.validpath = f"/Users/ujjwalsingh/Development/dl/DA6401_A3/lexicons/{lang_code}.translit.sampled.dev.tsv"

# Language class for character mapping
class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addWord(self, word):
        for char in word:
            self.addchar(char)

    def addchar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

# Data processing functions
def readLangs(lang1, lang2, path):
    # Read the file and split into lines
    data = pd.read_csv(path, names=['eng', 'tel'], sep=' ', header=None)
    data = data.dropna()
    print(data)
    data = data.values.tolist()
    pairs = [[(s) for s in l] for l in data]

    # Create Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, path):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    return input_lang, output_lang, pairs

def indexesFromWord(lang, word):
    return [lang.char2index[l] for l in word]

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)

def wordFromTensor(lang, tensor):
    word = ''
    for i in tensor:
        if i.item() == EOS_token:
            break
        word += lang.index2char[i.item()]
    return word

# DataLoader creation
def get_dataloader(batch_size, path, input_lang=None, output_lang=None):
    if input_lang is None or output_lang is None:
        input_lang, output_lang, pairs = prepareData('eng', 'tel', path)
    else:
        _, _, pairs = prepareData('eng', 'tel', path)
    
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromWord(input_lang, inp)
        tgt_ids = indexesFromWord(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

# Model architecture
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers=1, dropout_p=0.1, bidirectional=False, cell_type='GRU'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_size)
        # Cell types
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size, num_layers=1, dropout_p=0.1, bidirectional=False, cell_type='GRU'):
        super(AttnDecoderRNN, self).__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = BahdanauAttention(hidden_size)
        
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
            
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        
        if self.cell_type == 'LSTM':
            h_n, c_n = encoder_hidden
            decoder_hidden = (h_n, c_n)
        else:
            decoder_hidden = encoder_hidden
            
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        
        if self.cell_type == 'LSTM':
            h_n, c_n = hidden
            query = h_n[-1].unsqueeze(0).permute(1, 0, 2)
        else:
            query = hidden[-1].unsqueeze(0).permute(1, 0, 2)

        context, attn_weights = self.attention(query, encoder_outputs)
        input_rnn = torch.cat((embedded, context), dim=2)

        if self.cell_type == 'LSTM':
            output, (h_n, c_n) = self.rnn(input_rnn, hidden)
            hidden = (h_n, c_n)
        else:
            output, hidden = self.rnn(input_rnn, hidden)
            
        output = self.out(output)
        return output, hidden, attn_weights

# Training functions
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    accuracy = 0
    
    encoder.train()
    decoder.train()
    
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), accuracy

def evaluate(dataloader, encoder, decoder, criterion, output_lang, input_lang, epoch, n_epochs, is_test=False):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            # Compute accuracy
            _, predicted_indices = decoder_outputs.max(dim=-1)
            correct = 0
            
            # Track attention for visualization if needed
            attention_check = True
            for i in range(predicted_indices.shape[0]):
                my_prediction = 'Wrong'
                # Compare predicted and target words
                if wordFromTensor(output_lang, predicted_indices[i]) == wordFromTensor(output_lang, target_tensor[i]):
                    correct += 1
                    my_prediction = 'Correct'
                
                # Save attention weights and predictions for final epoch and test set
                if epoch == n_epochs and is_test:
                    english = wordFromTensor(input_lang, input_tensor[i])
                    telugu_correct = wordFromTensor(output_lang, predicted_indices[i])
                    telugu_predicted = wordFromTensor(output_lang, target_tensor[i])
                    
                    # Store attention weights for visualization
                    attn_temp = []
                    if len(attention_weights) < 10:
                        temp1 = []
                        temp2 = []
                        for m in english:
                            temp1.append(m)
                        for m in telugu_predicted:
                            temp2.append(m)
                        ylabels.append(temp2)
                        xlabels.append(temp1)
                        
                        for j in range(0, len(telugu_predicted)):
                            temp = []
                            for k in range(0, len(english)):
                                temp.append(decoder_attn[i][j][k].cpu().item())
                            attn_temp.append(temp)
                        attention_weights.append(attn_temp)
                    
                    # Store prediction data
                    filedata.append({
                        'Data': wordFromTensor(input_lang, input_tensor[i]),
                        'Correct': wordFromTensor(output_lang, predicted_indices[i]),
                        'Predicted': wordFromTensor(output_lang, target_tensor[i]),
                        'Prediction': my_prediction
                    })
                
            total_correct += correct
            total_samples += input_tensor.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_samples) * 100

    return avg_loss, accuracy

def train(output_lang, train_dataloader, val_dataloader, encoder, decoder, n_epochs, input_lang, learning_rate=0.001, print_every=1):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    print("Training Started...")
    
    for epoch in range(1, n_epochs + 1):
        loss, acc = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        
        train_loss, train_accuracy = evaluate(train_dataloader, encoder, decoder, criterion, output_lang, input_lang, epoch, n_epochs, False)
        val_loss, val_accuracy = evaluate(val_dataloader, encoder, decoder, criterion, output_lang, input_lang, epoch, n_epochs, True)
        
        wandb.log({
            'epoch': epoch,
            'validation_accuracy': val_accuracy,
            'validation_loss': val_loss,
            'training_accuracy': train_accuracy,
            'training_loss': train_loss,
        })
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
            print((val_loss, val_accuracy, train_accuracy, train_loss))

# WandB training function
def train_wandb():
    wandb.login()
    wandb.init()
    
    # Get parameters from wandb config
    config = wandb.config
    batch_size = config.batchsize
    hidden_size = config.hiddenlayersize
    embedding_size = config.embeddingsize
    encoder_layers = config.encoderlayers
    decoder_layers = config.decoderlayers  # Same as encoder layers
    cell_type = config.celltype
    epochs = config.epochs
    bidirectional = False if config.bidirectional == 'no' else True
    dropout = config.dropout
    learning_rate = config.learningrate
    
    # Prepare data and model
    input_lang, output_lang, pairs = prepareData('eng', 'tel', Config().trainpath)
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, Config().trainpath, input_lang, output_lang)
    _, _, valid_dataloader = get_dataloader(batch_size, Config().validpath, input_lang, output_lang)
    
    # Initialize models
    encoder = EncoderRNN(input_lang.n_chars, hidden_size, embedding_size, encoder_layers, dropout, bidirectional, cell_type).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, embedding_size, decoder_layers, dropout, bidirectional, cell_type).to(device)
    
    # Train model
    train(output_lang, train_dataloader, valid_dataloader, encoder, decoder, epochs, input_lang, learning_rate=learning_rate, print_every=1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter sweep with wandb for Dakshina dataset.")
    parser.add_argument("-lang", "--language", type=str, default="te", help="Language to use from Dakshina dataset")
    args = parser.parse_args()
    
    # Update paths based on language
    config = Config(args.language)
    
    # Define sweep configuration
    sweep_config = {
        'method': 'grid',  # or 'bayes' for Bayesian optimization
        'name': 'dakshina-sweep',
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {'values': [10, 15]},
            'learningrate': {'values': [0.001, 0.0005, 0.0001]},
            'embeddingsize': {'values': [16, 32, 64, 256]},
            'encoderlayers': {'values': [1, 2, 3]},
            'decoderlayers': {'values': [1, 2, 3]},
            'hiddenlayersize': {'values': [32, 64, 128, 256]},
            'celltype': {'values': ['RNN', 'GRU', 'LSTM']},
            'bidirectional': {'values': ['yes', 'no']},
            'dropout': {'values': [0.2, 0.3]},
            'batchsize': {'values': [32, 64]},
            'beamsize': {'values': [1, 3, 5]},
            'language': {'values': [args.language]}
        }
    }

    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project='DA6401_A3', entity='cs23m071-indian-institute-of-technology-madras')
    wandb.agent(sweep_id, function=train_wandb, count=20)  # adjust count to your needs

if __name__ == "__main__":
    main()
