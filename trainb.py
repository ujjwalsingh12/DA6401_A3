# File paths
train_csv = "te.translit.sampled.train.csv"
test_csv = "te.translit.sampled.test.csv"
val_csv = "te.translit.sampled.dev.csv"

import pandas as pd

# Load and clean train data
train_data = pd.read_csv(train_csv, header=None).dropna()
train_input = train_data[0].to_numpy()
train_output = train_data[1].to_numpy()

# Load and clean validation data
val_data = pd.read_csv(val_csv, header=None).dropna()
val_input = val_data[0].to_numpy()
val_output = val_data[1].to_numpy()

# Load and clean test data
test_data = pd.read_csv(test_csv, header=None).dropna()

def pre_processing(train_input, train_output):
    data = {
        "all_characters": [],
        "char_num_map": {},
        "num_char_map": {},
        "source_charToNum": torch.zeros(len(train_input), 30, dtype=torch.int, device=device),
        "source_data": train_input,
        "all_characters_2": [],
        "char_num_map_2": {},
        "num_char_map_2": {},
        "val_charToNum": torch.zeros(len(train_output), 23, dtype=torch.int, device=device),
        "target_data": train_output,
        "source_len": 0,
        "target_len": 0
    }
    k = 0
    for i in range(len(train_input)):
        train_input[i] = "{" + train_input[i] + "}" * (29 - len(train_input[i]))
        charToNum = []
        for char in train_input[i]:
            if char not in data["all_characters"]:
                data["all_characters"].append(char)
                index = data["all_characters"].index(char)
                data["char_num_map"][char] = index
                data["num_char_map"][index] = char
            else:
                index = data["all_characters"].index(char)
            charToNum.append(index)
        my_tensor = torch.tensor(charToNum, device=device)
        data["source_charToNum"][k] = my_tensor

        charToNum1 = []
        train_output[i] = "{" + train_output[i] + "}" * (22 - len(train_output[i]))
        for char in train_output[i]:
            if char not in data["all_characters_2"]:
                data["all_characters_2"].append(char)
                index = data["all_characters_2"].index(char)
                data["char_num_map_2"][char] = index
                data["num_char_map_2"][index] = char
            else:
                index = data["all_characters_2"].index(char)
            charToNum1.append(index)
        my_tensor1 = torch.tensor(charToNum1, device=device)
        data["val_charToNum"][k] = my_tensor1
        k += 1

    data["source_len"] = len(data["all_characters"])
    data["target_len"] = len(data["all_characters_2"])
    return data

data = pre_processing(copy.copy(train_input), copy.copy(train_output))

def pre_processing_validation(val_input, val_output):
    data2 = {
        "all_characters": [],
        "char_num_map": {},
        "num_char_map": {},
        "source_charToNum": torch.zeros(len(val_input), 30, dtype=torch.int, device=device),
        "source_data": val_input,
        "all_characters_2": [],
        "char_num_map_2": {},
        "num_char_map_2": {},
        "val_charToNum": torch.zeros(len(val_output), 23, dtype=torch.int, device=device),
        "target_data": val_output,
        "source_len": 0,
        "target_len": 0
    }
    k = 0
    m1 = data["char_num_map"]
    m2 = data["char_num_map_2"]
    for i in range(len(val_input)):
        val_input[i] = "{" + val_input[i] + "}" * (29 - len(val_input[i]))
        charToNum = []
        for char in val_input[i]:
            if char not in data2["all_characters"]:
                data2["all_characters"].append(char)
                index = m1[char]
                data2["char_num_map"][char] = index
                data2["num_char_map"][index] = char
            else:
                index = m1[char]
            charToNum.append(index)
        my_tensor = torch.tensor(charToNum, device=device)
        data2["source_charToNum"][k] = my_tensor

        charToNum1 = []
        val_output[i] = "{" + val_output[i] + "}" * (22 - len(val_output[i]))
        for char in val_output[i]:
            if char not in data2["all_characters_2"]:
                data2["all_characters_2"].append(char)
                index = m2[char]
                data2["char_num_map_2"][char] = index
                data2["num_char_map_2"][index] = char
            else:
                index = m2[char]
            charToNum1.append(index)
        my_tensor1 = torch.tensor(charToNum1, device=device)
        data2["val_charToNum"][k] = my_tensor1
        k += 1

    data2["source_len"] = len(data2["all_characters"])
    data2["target_len"] = len(data2["all_characters_2"])
    return data2

data2 = pre_processing_validation(copy.copy(val_input), copy.copy(val_output))

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.source = x
        self.target = y

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        source_data = self.source[idx]
        target_data = self.target[idx]
        return source_data, target_data

def validationAccuracy(encoder, decoder, batchsize, tf_ratio, cellType, bidirection):
    dataLoader = dataLoaderFun("validation", batchsize)
    encoder.eval()
    decoder.eval()
    total_sequences = 0
    total_correct_sequences = 0
    total_char_matches = 0
    total_characters = 0
    total_loss = 0

    lossFunction = nn.NLLLoss()

    for source_batch, target_batch in dataLoader:
        actual_batch_size = source_batch.shape[0]
        total_sequences += actual_batch_size
        total_characters += target_batch.numel()

        encoder_initial_state = encoder.getInitialState(actual_batch_size)
        if bidirection == "Yes":
            reversed_batch = torch.flip(source_batch, dims=[1])
            source_batch = (source_batch + reversed_batch) // 2
        if cellType == 'LSTM':
            encoder_initial_state = (encoder_initial_state, encoder.getInitialState(actual_batch_size))

        encoder_states, _ = encoder(source_batch, encoder_initial_state)
        decoder_current_state = encoder_states[-1, :, :, :]
        encoder_final_layer_states = encoder_states[:, -1, :, :]
        output_seq_len = target_batch.shape[1]

        loss = 0
        decoder_actual_output = []
        randNumber = random.random()

        for i in range(output_seq_len):
            if i == 0:
                decoder_current_input = torch.full((actual_batch_size, 1), 0, device=device)
            else:
                if randNumber < tf_ratio:
                    decoder_current_input = target_batch[:, i].reshape(actual_batch_size, 1)
                else:
                    decoder_current_input = decoder_current_input.reshape(actual_batch_size, 1)
            decoder_output, decoder_current_state, _ = decoder(decoder_current_input, decoder_current_state, encoder_final_layer_states)
            topv, topi = decoder_output.topk(1)
            decoder_current_input = topi.squeeze().detach()
            decoder_actual_output.append(decoder_current_input)

            decoder_output = decoder_output[:, -1, :]
            curr_target_chars = target_batch[:, i].long()
            loss += lossFunction(decoder_output, curr_target_chars)

        total_loss += loss.item() / output_seq_len
        decoder_actual_output = torch.cat(decoder_actual_output, dim=0).reshape(output_seq_len, actual_batch_size).transpose(0, 1)
        total_correct_sequences += (decoder_actual_output == target_batch).all(dim=1).sum().item()
        total_char_matches += (decoder_actual_output == target_batch).sum().item()

    encoder.train()
    decoder.train()

    wandb.log({
        'validation_loss': total_loss / len(dataLoader),
        'validation_accuracy': total_correct_sequences / total_sequences,
        'validation_char_accuracy': total_char_matches / total_characters
    })

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze().unsqueeze(1)
        weights = F.softmax(scores, dim=0)
        weights = weights.permute(2, 1, 0)
        keys = keys.permute(1, 0, 2)
        context = torch.bmm(weights, keys)
        return context, weights

class Encoder(nn.Module):
    def __init__(self, inputDim, embSize, encoderLayers, hiddenLayerNuerons, cellType, batch_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(inputDim, embSize)
        self.encoderLayers = encoderLayers
        self.hiddenLayerNuerons = hiddenLayerNuerons
        self.cellType = cellType
        if cellType == 'GRU':
            self.rnn = nn.GRU(embSize, hiddenLayerNuerons, num_layers=encoderLayers, batch_first=True)
        elif cellType == 'RNN':
            self.rnn = nn.RNN(embSize, hiddenLayerNuerons, num_layers=encoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize, hiddenLayerNuerons, num_layers=encoderLayers, batch_first=True)

    def forward(self, sourceBatch, encoderCurrState):
        sequenceLength = sourceBatch.shape[1]
        batch_size = sourceBatch.shape[0]
        encoderStates = torch.zeros(sequenceLength, self.encoderLayers, batch_size, self.hiddenLayerNuerons, device=device)
        for i in range(sequenceLength):
            currInput = sourceBatch[:, i].reshape(batch_size, 1)
            _, encoderCurrState = self.statesCalculation(currInput, encoderCurrState)
            if self.cellType == 'LSTM':
                encoderStates[i] = encoderCurrState[1]
            else:
                encoderStates[i] = encoderCurrState
        return encoderStates, encoderCurrState

    def statesCalculation(self, currentInput, prevState):
        embdInput = self.embedding(currentInput)
        output, prev_state = self.rnn(embdInput, prevState)
        return output, prev_state

    def getInitialState(self, batch_size):
        return torch.zeros(self.encoderLayers, batch_size, self.hiddenLayerNuerons, device=device)

class Decoder(nn.Module):
    def __init__(self, outputDim, embSize, hiddenLayerNuerons, decoderLayers, cellType, dropout_p):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(outputDim, embSize)
        self.cellType = cellType
        if cellType == 'GRU':
            self.rnn = nn.GRU(embSize + hiddenLayerNuerons, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)
        elif cellType == 'RNN':
            self.rnn = nn.RNN(embSize + hiddenLayerNuerons, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)
        else:
            self.rnn = nn.LSTM(embSize + hiddenLayerNuerons, hiddenLayerNuerons, num_layers=decoderLayers, batch_first=True)
        self.fc = nn.Linear(hiddenLayerNuerons, outputDim)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention(hiddenLayerNuerons).to(device)

    def forward(self, current_input, prev_state, encoder_final_layers):
        if self.cellType == 'LSTM':
            context, attn_weights = self.attention(prev_state[1][-1, :, :], encoder_final_layers)
        else:
            context, attn_weights = self.attention(prev_state[-1, :, :], encoder_final_layers)
        embd_input = self.embedding(current_input)
        curr_embd = F.relu(embd_input)
        input_gru = torch.cat((curr_embd, context), dim=2)
        output, prev_state = self.rnn(input_gru, prev_state)
        output = self.dropout(output)
        output = self.softmax(self.fc(output))
        return output, prev_state, attn_weights


def dataLoaderFun(dataName, batch_size):
    if dataName == 'train':
        dataset = MyDataset(data["source_charToNum"], data['val_charToNum'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = MyDataset(data2["source_charToNum"], data2['val_charToNum'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train(embSize, encoderLayers, decoderLayers, hiddenLayerNuerons, cellType, bidirection, dropout, epochs, batchsize, learningRate, optimizer, tf_ratio):
    dataLoader = dataLoaderFun("train", batchsize)
    encoder = Encoder(data["source_len"], embSize, encoderLayers, hiddenLayerNuerons, cellType, batchsize).to(device)
    decoder = Decoder(data["target_len"], embSize, hiddenLayerNuerons, encoderLayers, cellType, dropout).to(device)
    if optimizer == 'Adam':
        encoderOptimizer = optim.Adam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.Adam(decoder.parameters(), lr=learningRate)
    else:
        encoderOptimizer = optim.NAdam(encoder.parameters(), lr=learningRate)
        decoderOptimizer = optim.NAdam(decoder.parameters(), lr=learningRate)
    lossFunction = nn.NLLLoss()
    for epoch in range(epochs):
        train_accuracy = 0
        train_loss = 0
        for batch_num, (source_batch, target_batch) in enumerate(dataLoader):
            actual_batch_size = source_batch.shape[0]
            encoder_initial_state = encoder.getInitialState(actual_batch_size)
            if bidirection == "Yes":
                reversed_batch = torch.flip(source_batch, dims=[1])
                source_batch = (source_batch + reversed_batch) // 2
            if cellType == 'LSTM':
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState(actual_batch_size))
            encoder_states, dummy = encoder(source_batch, encoder_initial_state)
            decoder_current_state = dummy
            encoder_final_layer_states = encoder_states[:, -1, :, :]
            loss = 0
            output_seq_len = target_batch.shape[1]
            decoder_actual_output = []
            randNumber = random.random()
            for i in range(output_seq_len):
                if i == 0:
                    decoder_current_input = torch.full((actual_batch_size, 1), 0, device=device)
                else:
                    if randNumber < tf_ratio:
                        decoder_current_input = target_batch[:, i].reshape(actual_batch_size, 1)
                    else:
                        decoder_current_input = decoder_current_input.reshape(actual_batch_size, 1)
                decoder_output, decoder_current_state, _ = decoder(decoder_current_input, decoder_current_state, encoder_final_layer_states)
                topv, topi = decoder_output.topk(1)
                decoder_current_input = topi.squeeze().detach()
                decoder_actual_output.append(decoder_current_input)
                decoder_output = decoder_output[:, -1, :]
                curr_target_chars = target_batch[:, i].type(dtype=torch.long)
                loss += (lossFunction(decoder_output, curr_target_chars))
            decoder_actual_output = torch.cat(decoder_actual_output, dim=0).reshape(output_seq_len, actual_batch_size).transpose(0, 1)
            train_accuracy += (decoder_actual_output == target_batch).all(dim=1).sum().item()
            train_loss += (loss.item() / output_seq_len)
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()
            encoderOptimizer.step()
            decoderOptimizer.step()

        #Logging train metrics here
        wandb.log({'train_accuracy': train_accuracy / len(data["source_charToNum"])})
        wandb.log({'train_loss': train_loss / len(dataLoader)})

        validationAccuracy(encoder, decoder, batchsize, tf_ratio, cellType, bidirection)


def numToCharConverter(inputArray, outputArray, data):
    mp = data['num_char_map_2']
    for row1, row2 in zip(inputArray, outputArray):
        t1 = ''.join([mp[e1.item()] for e1 in row1])
        t2 = ''.join([mp[e2.item()] for e2 in row2])


import wandb

def train_model():
    # Initialize wandb run first
    with wandb.init(project='DA6401_A3', entity='cs23m071-indian-institute-of-technology-madras') as run:
        config = wandb.config

        # Dynamically name the run after initialization
        run.name = f"embedding{config.embSize}_cellType{config.cellType}_batchSize{config.batchsize}"

        # Call your training logic
        train(
            embSize=config.embSize,
            encoderLayers=config.encoderLayers,
            decoderLayers=config.decoderLayers,
            hiddenLayerNuerons=config.hiddenLayerNuerons,
            cellType=config.cellType,
            bidirection=config.bidirection,
            dropout=config.dropout,
            epochs=config.epochs,
            batchsize=config.batchsize,
            learningRate=config.learningRate,
            optimizer=config.optimizer,
            tf_ratio=config.tf_ratio
        )

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'name': 'Assignment_3_with_Attention_new',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy',
    },
    'parameters': {
        'embSize': {'values': [32, 64, 128]},
        'encoderLayers': {'values': [1, 5, 10]},
        'decoderLayers': {'values': [1, 5, 10]},
        'hiddenLayerNuerons': {'values': [64, 256, 512]},
        'cellType': {'values': ['GRU', 'RNN']},
        'bidirection': {'values': ['no', 'Yes']},
        'dropout': {'values': [0, 0.1, 0.2, 0.5]},
        'epochs': {'values': [15, 20, 25]},
        'batchsize': {'values': [32, 64, 128]},
        'learningRate': {'values': [1e-2, 1e-3, 1e-4]},
        'optimizer': {'values': ['Adam', 'Nadam']},
        'tf_ratio': {'values': [0.2, 0.5, 0.7]}
    }
}

# Create the sweep
sweep_id = wandb.sweep(
    sweep=sweep_config,
    project='DA6401_A3',
    entity='cs23m071-indian-institute-of-technology-madras'
)

# Launch the sweep agent
wandb.agent(
    sweep_id=sweep_id,
    function=train_model,
    count=25,
    entity='cs23m071-indian-institute-of-technology-madras',
    project='DA6401_A3'
)

