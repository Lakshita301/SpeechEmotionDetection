# Speech Emotion Detection (Mini Project)

**Team:** Mahashweta Panigrahi(PES2UG23CS315)
          , Lakshita Negi(PES2UG23CS301)

**Course:** UE23CS352A Machine Learning — Mini Project  

---

## 1. Project Overview

Detect emotions from short speech clips using classical ML models (SVM / Random Forest) trained on RAVDESS dataset features (MFCCs, chroma, mel, tonnetz).

---

## 2. Objectives

- Build a reproducible ML pipeline for Speech Emotion Recognition (SER) using **RAVDESS dataset**.
- Extract robust audio features using `librosa`.
- Train and evaluate SVM and Random Forest models with cross-validation.
- Save the best model and provide an inference script.
- Prepare deliverables: GitHub repo, one-page PDF summary, presentation slides, and demo-ready code.

---

## 3. Dataset: RAVDESS

- **Full name:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Classes:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **File format:** WAV files named as `03-01-01-01-01-01-01.wav`
- **Emotion code mapping (third field):**
  - 01 = Neutral  
  - 02 = Calm  
  - 03 = Happy  
  - 04 = Sad  
  - 05 = Angry  
  - 06 = Fearful  
  - 07 = Disgust  
  - 08 = Surprised

> **Note:** Only audio files are needed. Video files are not used.

---

## 4. File Structure
Speech-Emotion-Detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/ # RAVDESS dataset (do not commit)
├── src/
│ ├── features.py
│ ├── train_model.py
│ ├── evaluate.py
│ └── infer.py
├── models/ # Trained models (do not commit)
└── one_page_writeup.pdf
