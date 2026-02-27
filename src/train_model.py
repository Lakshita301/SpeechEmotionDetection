import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# ======================
# 1️⃣ Load and preprocess data
# ======================

DATA_DIR = "data"
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

X, y = [], []

for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            label = emotions.get(emotion_code)
            if label:
                feature = extract_features(os.path.join(root, file))
                if feature is not None:
                    X.append(feature)
                    y.append(label)

X, y = np.array(X), np.array(y)
print(f"✅ Extracted {len(X)} samples")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 2️⃣ Evaluation function
# ======================

def evaluate_model(model, X_test, y_test, name, save_model=False, is_keras=False):
    y_pred = model.predict(X_test)

    # If keras model, convert softmax to labels
    if is_keras:
        y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    if save_model:
        if is_keras:
            model.save(f"{name.replace(' ', '_')}.h5")
        else:
            joblib.dump(model, f"{name.replace(' ', '_')}.joblib")
        print(f"✅ {name} saved!")

    return acc

# ======================
# 3️⃣ Model Training
# ======================

results = {}

# Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, min_samples_split=3, random_state=42)
rf.fit(X_train, y_train)
results['Random Forest'] = evaluate_model(rf, X_test, y_test, "Random Forest", save_model=True)

# SVM
print("\n⚙️ Training SVM...")
svm = SVC(kernel='rbf', C=5, gamma='scale')
svm.fit(X_train, y_train)
results['SVM'] = evaluate_model(svm, X_test, y_test, "SVM", save_model=True)

# Gradient Boosting
print("\n🚀 Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
results['Gradient Boosting'] = evaluate_model(gb, X_test, y_test, "Gradient Boosting", save_model=True)

# Artificial Neural Network
print("\n🧠 Training Artificial Neural Network...")
ann = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = ann.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

test_loss, test_acc = ann.evaluate(X_test, y_test, verbose=0)
print(f"\n🧩 ANN Accuracy: {test_acc:.4f}")

# ANN Evaluation
results['ANN'] = evaluate_model(ann, X_test, y_test, "ANN", save_model=True, is_keras=True)

# ======================
# 4️⃣ Model Comparison
# ======================

print("\n=== 🏆 Model Comparison ===")
for model_name, acc in results.items():
    print(f"{model_name:20}: {acc:.3f}")
