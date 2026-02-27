# evaluate.py
import os
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from features import extract_features

# Function to extract label from RAVDESS filename
def get_label_from_filename(filename):
    base = os.path.basename(filename)
    parts = base.split('-')
    if len(parts) >= 3:
        emo_code = parts[2]
        mapping = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        return mapping.get(emo_code, 'unknown')
    return 'unknown'

# Load dataset and extract features
def load_dataset(path_glob):
    X, y = [], []
    for f in glob.glob(path_glob, recursive=True):
        try:
            feat = extract_features(f)
            label = get_label_from_filename(f)
            if label != 'unknown':
                X.append(feat)
                y.append(label)
        except Exception as e:
            print('Error processing', f, e)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Path to saved model and dataset
    MODEL_PATH = '../models/rf_model.joblib'  # change to SVM model if needed
    DATA_PATH = '../data/**/*.wav'

    print('Loading trained model...')
    model = joblib.load(MODEL_PATH)

    print('Loading dataset and extracting features...')
    X, y = load_dataset(DATA_PATH)
    print(f'Total samples: {len(X)}')

    print('Evaluating model...')
    y_pred = model.predict(X)

    print('\n=== Classification Report ===')
    print(classification_report(y, y_pred))

    print('=== Accuracy Score ===')
    print(f'Accuracy: {accuracy_score(y, y_pred):.4f}')

    print('=== Confusion Matrix ===')
    print(confusion_matrix(y, y_pred))

    # Optional: save report to a text file
    with open('evaluation_report.txt', 'w') as f:
        f.write('Classification Report\n')
        f.write(classification_report(y, y_pred))
        f.write('\nAccuracy: {:.4f}\n'.format(accuracy_score(y, y_pred)))
        f.write('\nConfusion Matrix\n')
        f.write(np.array2string(confusion_matrix(y, y_pred)))
    print('\nEvaluation report saved as evaluation_report.txt')
