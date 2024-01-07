import pandas as pd
from add_remaining_useful_life import *

def create_lag_features(df, window_size):
    # Create lag features for each sensor that exists in the dataframe
    sensor_cols = [col for col in df.columns if col.startswith('s_')]
    for sensor in sensor_cols:
        for i in range(1, window_size + 1):
            df[f'{sensor}_lag_{i}'] = df[sensor].shift(i)
    return df

def feature_engineering(data, window_size=5):
    # Apply create_lag_features function to each group within the grouped DataFrame
    lagged_data = [create_lag_features(group, window_size) for _, group in data]
    # Concatenate the results to get the modified DataFrame
    modified_data = pd.concat(lagged_data, ignore_index=True)

    # Regroup the modified data based on "unit_nr"
    modified_data_grouped = modified_data.groupby("unit_nr")

    return modified_data_grouped


def loading_FD001():

    # define filepath to read data
    dir_path = 'data/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features in training set
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    # separate title information and sensor data
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]

    # min-max normalization of the sensor data
    data_norm = (data - data.min()) / (data.max() - data.min())
    train_norm = pd.concat([title, data_norm], axis=1)

    # add piece-wise target remaining useful life
    train_norm = add_remaining_useful_life(train_norm)
    train_norm['RUL'].clip(upper=125, inplace=True) # in the paper the MAX RUL is mentioned as 125

    # Group the training set with unit
    train_group = train_norm.groupby(by="unit_nr")

    # Drop non-informative features in the testing set
    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    test_norm = pd.concat([title, data_norm], axis=1)

    # Group the testing set with unit
    test_group = test_norm.groupby(by="unit_nr")

    # # Fill NaN values with zeros
    # train_group = train_group.fillna(0)
    # test_group = test_group.fillna(0)

    # Apply feature engineering to the training and testing data
    train_group = feature_engineering(train_group)
    test_group = feature_engineering(test_group)

    # print(train_group)
    # print(test_group)

    return train_group, y_test, test_group


if __name__ == "__main__":
    train_group, y_test, test_group = loading_FD001()

    # Concatenate the train_group and test_group into single DataFrames
    train_df = pd.concat([group for _, group in train_group])
    test_df = pd.concat([group for _, group in test_group])

    # Save concatenated DataFrames to CSV files
    train_df.to_csv('processed_data/train_group.csv', index=False)
    test_df.to_csv('processed_data/test_group.csv', index=False)

    # Save y_test to a CSV file
    y_test.to_csv('processed_data/y_test.csv', index=False)


