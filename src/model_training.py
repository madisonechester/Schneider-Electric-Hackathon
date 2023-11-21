import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1
from tensorflow.compat.v1 import keras
from tensorflow.keras.models import save_model
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy


def load_data(file_path):
    # TODO: Load processed data from CSV file

    df = pd.read_csv(file_path)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    return df

def save_df(df, file_path):
    df.to_csv(file_path)
    pass

def train_val_test_df_split(perc, df):

    # split df
    # test
    split = int(df.shape[0]*perc)
    df_aux = df[:split]
    df_test = df[split:]

    # train and validation
    split = int(df_aux.shape[0]*perc)
    df_train = df_aux[:split]
    df_val = df_aux[split:]

    print("Shapes:")
    print("df_train:", df_train.shape)
    print('==================')
    print("df_val:", df_val.shape)
    print('==================')
    print("df_test:", df_test.shape)

    return df_train, df_val, df_test


def scale_data(features_to_scale, df_train, df_val, df_test):
    
    scaler = MinMaxScaler()
    
    # Fit the scaler on training data and transform the features
    df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])
    
    # Transform validation and test data using the scaler fitted on training data
    df_val[features_to_scale] = scaler.transform(df_val[features_to_scale])
    df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

    save_df(df_train, os.path.join('..\data\scaled_data', 'df_train.csv'))
    save_df(df_val, os.path.join('..\data\scaled_data', 'df_val.csv'))
    save_df(df_test, os.path.join('..\data\scaled_data', 'df_test.csv'))
    
    return df_train, df_val, df_test

def create_sequences(X, y, window):
    rows = X.shape[0]
    features = X.shape[1]

    sequences = np.zeros((rows - window - 1 , window, features))
    target_values = np.zeros((rows - window - 1 ))

    for i in range(rows - window -1):

        sequences[i, : , : ] = X[ i : (i + window), : ] # cogemos las filas de i a i+window y todas las columnas (features)
        target_values[i] = y[i + window]

        # print(i)

    return sequences, target_values


def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 5
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr


def train_model(sequences_train, target_values_train, sequences_val, target_values_val, num_countries):
    # TODO: Initialize your model and train it
    # Define the LSTM model for classification

    lr_scheduler = LearningRateScheduler(step_decay)
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50, input_shape=(sequences_train.shape[1], sequences_train.shape[2]), kernel_regularizer=keras.regularizers.l1(0.01)))
    model.add(keras.layers.Dense(units=num_countries, activation='softmax', kernel_regularizer=keras.regularizers.l1(0.01)))

    # Compile the model
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.compile(
        optimizer=optimizer, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.summary()


    # Train the model
    history = model.fit(
        sequences_train, 
        target_values_train, 
        epochs=10, 
        batch_size=4, 
        validation_data=(sequences_val, target_values_val), 
        callbacks=[lr_scheduler, early_stopping]
    )


    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    return model



def save_model(model, model_path):
    # TODO: Save your trained model

    model.save(model_path)
    print('=========================================')
    print(f'Saved Model in {model_path}')
    print('=========================================')




def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--file_path',  
        type=str, 
        default='..\data\final_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='..\models\model_adri.h5', 
        help='Path to save the trained model'
    )
    return parser.parse_args()



def main(file_path, model_path):
    file_path = '../data/final_data.csv'
    df = load_data(file_path)
    print('=========================================')
    print('File correctly loaded')
    print('=========================================')
    countries = ['HU', 'IT', 'PO', 'SP', 'DE', 'DK', 'SE', 'NE']
    labels_countries = {
        'SP': 0, # Spain
        'UK': 1, # United Kingdom
        'DE': 2, # Germany
        'DK': 3, # Denmark
        'HU': 5, # Hungary
        'SE': 4, # Sweden
        'IT': 6, # Italy
        'PO': 7, # Poland
        'NL': 8 # Netherlands
    }

    num_countries = len(countries)

    features = ['Hour', 'spring', 'summer', 'winter', 'day_of_week', 'is_weekend',
        'DEgen', 'DKgen',  'HUgen', 'ITgen', 'NEgen',  'POgen', 'SEgen', 'SPgen',
        'DEload', 'DKload', 'HUload', 'ITload', 'NEload', 'POload', 'SEload', 'SPload', 
        'DE_surplus',  'DK_surplus', 'HU_surplus', 'IT_surplus', 'NE_surplus', 'PO_surplus', 'SE_surplus', 'SP_surplus'  
    ]

    num_features = len(features)
    label_column = 'label'
    print('num_features: ', num_features)
    print('num_countries: ', num_countries)

    print('=========================================')
    print('Splitting data')
    print('=========================================')

    perc = 0.8
    df_train, df_val, df_test = train_val_test_df_split(perc, df)

    
    features_to_scale = [
    'Hour', 'DEgen', 'DEload', 'DKgen', 'DKload', 'HUgen', 'HUload',
    'ITgen', 'ITload', 'NEgen', 'NEload', 'POgen', 'POload', 'SEgen',
    'SEload', 'SPgen', 'SPload', 'HU_surplus', 'IT_surplus', 'PO_surplus',
    'SP_surplus', 'DE_surplus', 'DK_surplus', 'SE_surplus', 'NE_surplus',
    'day_of_week']

    scaled_df_train, scaled_df_val, scaled_df_test = scale_data(features_to_scale, df_train, df_val, df_test)

    print('=========================================')
    print('Creating arrays')
    print('=========================================')

    # create arrays
    X_train, X_val, X_test = scaled_df_train[features].values, scaled_df_val[features].values, scaled_df_test[features].values
    y_train, y_val, y_test = df_train[label_column].values, df_val[label_column].values, df_test[label_column].values

    window = 48 # The amount of hours we want to use as input data

    # Usage example
    sequences_train, target_values_train = create_sequences(X_train, y_train, window)
    sequences_val, target_values_val = create_sequences(X_val, y_val, window)
    sequences_test, target_values_test = create_sequences(X_test, y_test, window)

    print('=========================================')
    print('Training Model')
    print('=========================================')
    
    model = train_model(sequences_train, target_values_train, sequences_val, target_values_val, num_countries)

    print('=========================================')
    print('Saving Model')
    print('=========================================')
    model_path = os.path.join('..', 'models', 'model_adri.h5')
    save_model(model, model_path)
    print('=========================================')
    print(f'Model saved to {model_path}')
    print('=========================================')

if __name__ == "__main__":
    args = parse_arguments()
    main(args.file_path, args.model_path)