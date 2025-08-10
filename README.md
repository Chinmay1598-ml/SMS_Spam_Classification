# SMS Spam Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project classifies SMS messages as spam or ham using machine learning. It includes data preprocessing, feature extraction (TF-IDF), model training (with tuning), and evaluation. The best model (Random Forest) achieves ~98% accuracy.

**Dataset:** 5572 SMS messages (source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) or similar; included as `data/Message_Classification.csv`).

**Key Insights:**
- Class imbalance: 86.6% ham, 13.4% spam → F1-score is crucial.
- Top model: Random Forest (Accuracy: 0.9803, F1: 0.9203).
- Common spam words: "free", "win", "call" (from word clouds).

## Requirements
- Python 3.11+
- Install dependencies: `pip install -r requirements.txt`

`requirements.txt` content:
pandas
numpy
matplotlib
seaborn
nltk
wordcloud
scikit-learn
xgboost
pretty-table
joblib

## How to Run
1. Clone the repo: `git clone https://github.com/yourusername/sms-spam-classification.git`
2. Navigate: `cd sms-spam-classification`
3. Install deps: `pip install -r requirements.txt`
4. Run the notebook: `jupyter notebook SMS_Spam_Classification.ipynb`
5. Interact: Enter messages in the prediction section.

## Project Structure
- `SMS_Spam_Classification.ipynb`: Main notebook.
- `data/`: Dataset.
- `model_evaluation_results_with_hyperparameters.csv`: Evaluation output.
- `Random_Forest_best_model.pkl`: Saved best model.

## Results
| Model                  | Accuracy | Precision | Recall | F1 Score | Best Hyperparameters                          |
|------------------------|----------|-----------|--------|----------|-----------------------------------------------|
| Logistic Regression   | 0.9794  | 0.9922   | 0.8523 | 0.9170  | {'C': 10, 'solver': 'liblinear'}             |
| Support Vector Machine| 0.9785  | 0.9496   | 0.8859 | 0.9167  | {'C': 10, 'kernel': 'linear'}                |
| Random Forest         | 0.9803  | 1.0000   | 0.8523 | 0.9203  | {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 50} |
| K-Nearest Neighbors   | 0.9417  | 1.0000   | 0.5638 | 0.7210  | {'n_neighbors': 3, 'weights': 'distance'}    |
| XGBoost               | 0.9731  | 0.9685   | 0.8255 | 0.8913  | {'learning_rate': 0.2, 'n_estimators': 200}  |

## Future Work
- Integrate BERT for better NLP.
- Deploy as a web app (e.g., Flask/Streamlit).
- Handle multilingual SMS.

## License
MIT License. See LICENSE file.
