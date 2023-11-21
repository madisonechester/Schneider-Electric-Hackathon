import pandas as pd
import argparse
import numpy as np
import json
from model_training import create_sequences
from tensorflow.keras.models import load_model
import os

def load_data(file_path):
    # TODO: Load processed data from CSV file

    scaled_df_test = pd.read_csv(file_path)
    scaled_df_test.interpolate(method='linear', limit_direction='both', inplace=True)
    
    return scaled_df_test


def make_predictions(scaled_df_test, model):
    # TODO: Use the model to make predictions on the test data

    features = ['Hour', 'spring', 'summer', 'winter', 'day_of_week', 'is_weekend',
        'DEgen', 'DKgen',  'HUgen', 'ITgen', 'NEgen',  'POgen', 'SEgen', 'SPgen',
        'DEload', 'DKload', 'HUload', 'ITload', 'NEload', 'POload', 'SEload', 'SPload', 
        'DE_surplus',  'DK_surplus', 'HU_surplus', 'IT_surplus', 'NE_surplus', 'PO_surplus', 'SE_surplus', 'SP_surplus'  
    ]

    num_features = len(features)
    label_column = 'label'
    window = 48

    X_test, y_test = scaled_df_test[features].values, scaled_df_test[label_column].values
    sequences_test, _ = create_sequences(X_test, y_test, window)
    predictions = model.predict(sequences_test)
    predicted_labels = np.argmax(predictions, axis=1)
    result_dict = {"target": {}}
    for i, label in enumerate(predicted_labels):
        result_dict["target"][str(i + 1)] = int(label)

    return result_dict


def save_predictions(result_dict, json_file_path):
    # TODO: Save predictions to a JSON file

    # Save the dictionary as JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(result_dict, json_file)

    print('=========================================')
    print(f"Result dictionary saved to {json_file_path}")
    print('=========================================')

    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='..\data\scaled_data\df_test.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default=os.path.join('..', 'models', 'model_adri.h5'),
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='..\predictions\predictions_adri.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(file_path, model_path, output_file):
    file_path = '..\data\scaled_data\df_test.csv'
    scaled_df_test = load_data(file_path)
    print('==============================')
    print('Loaded scaled df_test')
    print('==============================')
    model = load_model(model_path)
    print('==============================')
    print('Loaded model')
    print('==============================')
    predictions = make_predictions(scaled_df_test, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
