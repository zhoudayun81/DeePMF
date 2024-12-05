import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch import optim
#from sklearn.model_selection import ParameterGrid
import optuna
# import my own modules
import config_reader
import approach_functions as af

torch.manual_seed(af.RANDOM_SEED)
random.seed(af.RANDOM_SEED)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class One2OneDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data) - 1
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx + 1]
        return torch.tensor(x, dtype=torch.float).reshape(-1), torch.tensor(y, dtype=torch.float).reshape(-1)

class CustomTimeSeriesSplit:
    def __init__(self, n_splits=10, test_size=1, val_size=1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.val_size = val_size

    def split(self, indices):
        dataset_size = len(indices)
        chunk_size = dataset_size // self.n_splits
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0. Consider adjusting n_splits or dataset size.")
        chunk_indices = [indices[i:i + chunk_size] for i in range(0, dataset_size, chunk_size)]
        # ignore any remainders
        if len(chunk_indices) > self.n_splits:
            chunk_indices = chunk_indices[:self.n_splits]
        for fold in range(self.n_splits):
            # Use the chunk before the last one for validation
            test_indices = chunk_indices[fold][-self.test_size:]
            val_indices = chunk_indices[fold][-self.test_size - self.val_size:-self.test_size] if self.val_size > 0 else []
            # Use remaining chunks as training
            train_indices = chunk_indices[fold][:-self.test_size - self.val_size]
            yield train_indices, val_indices, test_indices

def get_data_loaders(data, n_splits, batch_size):
    dataset_size = len(data)
    indices = list(range(dataset_size))
    tscv = CustomTimeSeriesSplit(n_splits=n_splits)
    train_loaders = []
    val_loaders = []
    test_loaders = []
    for fold, (train_index, val_index, test_indices) in enumerate(tscv.split(indices)):
        fold_train_dataset = Subset(data, train_index)
        fold_val_dataset = Subset(data, val_index)
        fold_test_dataset = Subset(data, test_indices)
        # Create data loaders
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=False)
        fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)
        fold_test_loader = DataLoader(fold_test_dataset, batch_size=batch_size, shuffle=False)
        train_loaders.append(fold_train_loader)
        val_loaders.append(fold_val_loader)
        test_loaders.append(fold_test_loader)
    return train_loaders, val_loaders, test_loaders

class ModelParam:
    def __init__(self, matrix_size, hidden_size, num_layers, batch_size, kernel_size=None, padding=0, dropout_probability=None, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256):
        self.matrix_size = matrix_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.dropout_probability = dropout_probability
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.padding = padding
    
#%%
class VanillaNN(nn.Module):
    def __init__(self, modelparam):
        super(VanillaNN, self).__init__()
        self.vanilla = nn.Sequential(
            nn.Linear(modelparam.matrix_size, modelparam.hidden_size),
            nn.ReLU(),                                  # Activation function
            nn.Linear(modelparam.hidden_size, modelparam.hidden_size),
            nn.ReLU(),
            nn.Linear(modelparam.hidden_size, modelparam.matrix_size),
        )
    def forward(self, x):
        x = self.vanilla(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, modelparam):
        super(CustomCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, modelparam.hidden_size, kernel_size=modelparam.kernel_size),  # Convolutional layer
            nn.ReLU(),                                  # Activation function
            nn.Conv1d(modelparam.hidden_size, 1, kernel_size=modelparam.kernel_size),  # Convolutional layer
            nn.ReLU(),                                  # Activation function
            nn.Linear(modelparam.matrix_size, modelparam.matrix_size),
        )
    def forward(self, x):
        x = self.cnn(x)  # Pass input through the cnn
        return x
    
class CustomRNN(nn.Module):
    def __init__(self, modelparam):
        super(CustomRNN, self).__init__()
        self.rnn = nn.RNN(modelparam.matrix_size, modelparam.hidden_size, num_layers=modelparam.num_layers, dropout=modelparam.dropout_probability, batch_first=True)
        self.fc = nn.Linear(modelparam.hidden_size, modelparam.matrix_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
    
class CustomGRU(nn.Module):
    def __init__(self, modelparam):
        super(CustomGRU, self).__init__()
        self.gru = nn.GRU(modelparam.matrix_size, modelparam.hidden_size, num_layers=modelparam.num_layers, dropout=modelparam.dropout_probability, batch_first=True)
        self.fc = nn.Linear(modelparam.hidden_size, modelparam.matrix_size)
    def forward(self, x):
        out, _ = self.gru(x) 
        out = self.fc(out)
        return out

class CustomLSTM(nn.Module):
    def __init__(self, modelparam):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(modelparam.matrix_size, modelparam.hidden_size, num_layers=modelparam.num_layers, dropout=modelparam.dropout_probability, batch_first=True)
        self.fc = nn.Linear(modelparam.hidden_size, modelparam.matrix_size)
    def forward(self, x):
        out, _ = self.lstm(x)  # Pass input through the lstm
        out = self.fc(out)
        return out
    
class CustomTransformer(nn.Module):
    def __init__(self, modelparam):
        super(CustomTransformer, self).__init__()
        self.transformer = nn.Transformer(modelparam.matrix_size, nhead=int(modelparam.matrix_size**0.5), num_encoder_layers=modelparam.num_encoder_layers, num_decoder_layers=modelparam.num_decoder_layers, dim_feedforward=modelparam.dim_feedforward, dropout=modelparam.dropout_probability, batch_first=True)
    def forward(self, x, y):
        x = self.transformer(x, y)
        return x

# Function to create model based on model_name parameter
def create_model(model_name, matrix_size):
    # Dictionary mapping model names to their classes
    model_mapping = {
        'cnn': CustomCNN,
        'rnn': CustomRNN,
        'lstm': CustomLSTM,
        'gru': CustomGRU,
        'vanilla': VanillaNN,
        'transformer': CustomTransformer,
    }
    # Check if the model_name exists in model_mapping
    if model_name in model_mapping:
        # Instantiate the corresponding model class
        model = model_mapping[model_name](matrix_size)
    else:
        raise ValueError(f"Model '{model_name}' not found in model mapping.")
    return model.to(DEVICE)

def train_best(best_model_params, model_name, activity_info, train_loader):
    num_elements = (len(activity_info)+1)**2
    optimizer_name = best_model_params['optimizer']
    loss_function_name = best_model_params['loss_function']
    learning_rate = best_model_params['learning_rate']
    hidden_size = best_model_params['hidden_size']
    #num_layers = best_model_params['num_layers']
    epochs = best_model_params['epochs']
    criterion = loss_criterion(loss_function_name)
    if model_name == 'vanilla':
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
    elif model_name == 'cnn':
        kernel_size = config_reader.KERNEL_SIZE # fix for 1 now
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
        #modelparam.padding = padding
        modelparam.kernel_size = kernel_size
    elif model_name == 'rnn':
        dropout_probability = best_model_params['dropout_probability']
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'gru':
        dropout_probability = best_model_params['dropout_probability']
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'lstm':
        dropout_probability = best_model_params['dropout_probability']
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'transformer':
        dropout_probability = best_model_params['dropout_probability']
        modelparam = ModelParam(num_elements, hidden_size, config_reader.NUM_LAYERS, config_reader.BATCH_SIZE)
        modelparam.dropout_probability = dropout_probability
    model = create_model(model_name, modelparam)
    optimizer = optimizer_function(optimizer_name, model, learning_rate)
    # Train and evaluate model
    for epoch in range(epochs):
        train_model(model, model_name, train_loader, criterion, optimizer)
    return model, criterion

def prepare(trial, model_name, activity_info, loss_function_name, optimizer_name, learning_rate, hidden_size, num_layers, batch_size):
    num_elements = (len(activity_info)+1)**2
    criterion = loss_criterion(loss_function_name)
    if model_name == 'vanilla':
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
    elif model_name == 'cnn':
        #kernel_size = trial.suggest_categorical('kernel_size', [1])
        kernel_size = 1 # fix for 1 now
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
        #modelparam.padding = padding
        modelparam.kernel_size = kernel_size
    elif model_name == 'rnn':
        dropout_probability = trial.suggest_float('dropout_probability', config_reader.DROPOUT_PROBABILITY[0], config_reader.DROPOUT_PROBABILITY[-1])
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'gru':
        dropout_probability = trial.suggest_float('dropout_probability', config_reader.DROPOUT_PROBABILITY[0], config_reader.DROPOUT_PROBABILITY[-1])
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'lstm':
        dropout_probability = trial.suggest_float('dropout_probability', config_reader.DROPOUT_PROBABILITY[0], config_reader.DROPOUT_PROBABILITY[-1])
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
        modelparam.dropout_probability = dropout_probability
    elif model_name == 'transformer':
        dropout_probability = trial.suggest_float('dropout_probability', config_reader.DROPOUT_PROBABILITY[0], config_reader.DROPOUT_PROBABILITY[-1])
        modelparam = ModelParam(num_elements, hidden_size, num_layers, batch_size)
        modelparam.dropout_probability = dropout_probability
    model = create_model(model_name, modelparam)
    optimizer = optimizer_function(optimizer_name, model, learning_rate)
    return criterion, model, optimizer

def objective(trial, model_name, activity_info, train_loader, val_loader):
    # Hyperparameters to optimize
    optimizer_name = trial.suggest_categorical('optimizer', config_reader.OPTIMIZER)
    loss_function_name = trial.suggest_categorical('loss_function', config_reader.LOSS_FUNCTION)
    learning_rate = trial.suggest_float('learning_rate', config_reader.LEARNING_RATE[0], config_reader.LEARNING_RATE[-1])
    epochs = trial.suggest_int('epochs', config_reader.EPOCHS[0], config_reader.EPOCHS[-1])
    #batch_size = trial.suggest_categorical('batch_size', [1])
    batch_size = config_reader.BATCH_SIZE #fix for 1 now
    hidden_size = trial.suggest_categorical('hidden_size', config_reader.HIDDEN_SIZE)
    #num_layers = trial.suggest_int('num_layers', config_reader.NUM_LAYERS[0], config_reader.NUM_LAYERS[-1])
    num_layers = config_reader.NUM_LAYERS
    criterion, model, optimizer = prepare(trial, model_name, activity_info, loss_function_name, optimizer_name, learning_rate, hidden_size, num_layers, batch_size)
    # Train and evaluate model
    for epoch in range(epochs):
        train_model(model, model_name, train_loader, criterion, optimizer)
        val_loss, _, _ = evaluate_model(model, model_name, val_loader, criterion)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_loss

#%%
def train_model(model, model_name, train_loader, criterion, optimizer):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        if model_name == 'transformer':
            outputs = model(inputs, targets)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate_model(model, model_name, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    outputs = []
    ground_truth = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if model_name == 'transformer':
                output = model(inputs, targets)
            else:
                output = model(inputs)
            loss = criterion(output, targets)
            val_loss += loss.item() * inputs.size(0)
            outputs.append(output.reshape(int(output.size(1)**0.5), int(output.size(1)**0.5)).cpu().numpy().round().astype(int))
            ground_truth.append(targets.reshape(int(targets.size(1)**0.5), int(targets.size(1)**0.5)).cpu().numpy())
    return val_loss/len(val_loader.dataset), outputs, ground_truth

def save_state_dict_only(state_dict, dictpath):
    torch.save(state_dict, dictpath)

def save_state_dict(state_dict, dictpath, content, textpath):
    torch.save(state_dict, dictpath)
    with open(textpath, 'w') as output:
        output.write(af.tostring(content))

# define loss function
def loss_criterion(loss_function_name):
    #by default, use mean absolute error (MAE)
    criterion = nn.L1Loss()
    if loss_function_name=='L1Loss':
        criterion = nn.L1Loss()
    elif loss_function_name=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name=='MSELoss':
        criterion = nn.MSELoss()
    elif loss_function_name=='CTCLoss':
        criterion = nn.CTCLoss() # used differently
    elif loss_function_name=='NLLLoss':
        criterion = nn.NLLLoss()
    elif loss_function_name=='PoissonNLLLoss':
        criterion = nn.PoissonNLLLoss()
    elif loss_function_name=='GaussianNLLLoss':
        criterion = nn.GaussianNLLLoss() # used differently
    elif loss_function_name=='KLDivLoss':
        criterion = nn.KLDivLoss() 
    elif loss_function_name=='BCELoss':
        criterion = nn.BCELoss()
    elif loss_function_name=='BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function_name=='MarginRankingLoss':
        criterion = nn.MarginRankingLoss() # used differently
    elif loss_function_name=='HingeEmbeddingLoss':
        criterion = nn.HingeEmbeddingLoss()
    elif loss_function_name=='MultiLabelMarginLoss':
        criterion = nn.MultiLabelMarginLoss()
    elif loss_function_name=='HuberLoss':
        criterion = nn.HuberLoss()
    elif loss_function_name=='SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    elif loss_function_name=='SoftMarginLoss':
        criterion = nn.SoftMarginLoss()
    elif loss_function_name=='MultiLabelSoftMarginLoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif loss_function_name=='CosineEmbeddingLoss':
        criterion = nn.CosineEmbeddingLoss()
    elif loss_function_name=='MultiMarginLoss':
        criterion = nn.MultiMarginLoss()
    elif loss_function_name=='TripletMarginLoss':
        criterion = nn.TripletMarginLoss() # used differently
    elif loss_function_name=='TripletMarginWithDistanceLoss':
        criterion = nn.TripletMarginWithDistanceLoss() # used differently
    return criterion

# define optimizer
def optimizer_function(optimizer_name, model, learning_rate):
    # by default, use SGD
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name=='SGD':
        optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name=='Adadelta':
        optimiser = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Adadelta':
        optimiser = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Adam':
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='AdamW':
        optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name=='SparseAdam':
        optimiser = optim.SparseAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='ASGD':
        optimiser = optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_name=='LBFGS':
        optimiser = optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_name=='NAdam':
        optimiser = optim.NAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='RAdam':
        optimiser = optim.RAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name=='RMSprop':
        optimiser = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name=='Rprop':
        optimiser = optim.Rprop(model.parameters(), lr=learning_rate)
    return optimiser