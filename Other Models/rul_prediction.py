import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
train_data = pd.read_csv('Processed_Train_FD001.csv')
test_data = pd.read_csv('Processed_test_FD001.csv')

# Load original RUL values
y_true = pd.read_csv("data/RUL_FD001.txt", delim_whitespace=True, names=["RUL"])
y_true["unit_number"] = y_true.index

# Define the feature columns for scaling (excluding 'RUL' and 'unit_number')
train_columns = [column for column in train_data.columns if column not in ['RUL', 'unit_number']]
# Define the feature columns for scaling (excluding 'RUL' and 'unit_number')
test_columns = [column for column in test_data.columns if column not in ['RUL', 'unit_number']]

# Apply MinMax scaling to the training data
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
train_data[train_columns] = min_max_scaler.fit_transform(train_data[train_columns])
test_data[test_columns] = min_max_scaler.fit_transform(test_data[test_columns])
test_data = test_data.drop('RUL', axis=1)

# Define a custom dataset class for your data
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y

# Functions
def gen_train(data, seq_length, seq_cols):
    """
        function to prepare train data into (samples, time steps, features)
        id_df = train dataframe
        seq_length = look back period
        seq_cols = feature columns
    """

    data_array = data[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)

def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]


def gen_test(id_df, seq_length, seq_cols, mask_value):
    """
        function to prepare test data into (samples, time steps, features)
        function only returns last sequence of data for every unit
        id_df = test dataframe
        seq_length = look back period
        seq_cols = feature columns
    """
    df_mask = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    df_mask[:] = mask_value

    id_df = df_mask.append(id_df, ignore_index=True)

    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []

    start = num_elements - seq_length
    stop = num_elements

    lstm_array.append(data_array[start:stop, :])

    return np.array(lstm_array)

# Define your LSTM-based model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output from the sequence
        return out

sequence_length = 50
mask_value = 0
seq_cols = [column for column in train_data.columns if column !="RUL"]


# Initialize your data and model
x_train = np.concatenate([gen_train(train_data[train_data['unit_number'] == unit], sequence_length, seq_cols) for unit in train_data['unit_number'].unique()])
y_train = np.concatenate([gen_target(train_data[train_data['unit_number'] == unit], sequence_length, "RUL") for unit in train_data['unit_number'].unique()])

nb_features = x_train.shape[2]
nb_out = 1

train_dataset = CustomDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = LSTMModel(input_size=nb_features, hidden_size=100)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

# Training loop
for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    x_test = np.concatenate(
        [gen_test(test_data[test_data['unit_number'] == unit], sequence_length, seq_cols, mask_value) for unit in
         test_data['unit_number'].unique()])
    y_test = y_true.RUL.values
    test_inputs = torch.tensor(x_test, dtype=torch.float32)
    test_outputs = model(test_inputs)

    # Ensure that the target size matches the output size
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Ensure that the target size matches the output size
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    mse = criterion(test_outputs, y_test)

print("MSE on test data:", mse.item())


