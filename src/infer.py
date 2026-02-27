# infer.py
import argparse
import joblib
from features import extract_features

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run inference on a WAV file using a trained model')
    parser.add_argument('--model', required=True, help='Path to the saved model (.joblib)')
    parser.add_argument('--file', required=True, help='Path to the WAV file for prediction')
    args = parser.parse_args()

    # Load the trained model
    model = joblib.load(args.model)

    # Extract features from the input audio
    features = extract_features(args.file)

    # Predict emotion
    pred = model.predict([features])
    print('Predicted Emotion:', pred[0])
