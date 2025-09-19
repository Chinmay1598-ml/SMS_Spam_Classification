# SMS Spam Classification

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This project explores the task of classifying SMS messages as **spam** or **ham (legitimate)**.
I started with exploratory analysis to understand the dataset better, then tried out multiple models with both imbalanced and balanced data, and finally wrapped everything up with a **FastAPI backend + PyQt5 desktop GUI** for live predictions.

The focus was not just on accuracy, but on building an **end-to-end flow**:

* exploring the dataset →
* identifying class imbalance →
* training + tuning models →
* and finally deploying them in an app.

**Dataset**

* 5572 SMS messages (source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) or similar).
* Included locally as: `data/Message_Classification.csv`.

---

## 🔎 Exploratory Data Analysis (EDA)

Some key insights from EDA:

* Around **86.6% messages are ham**, only **13.4% are spam** → imbalance is significant.
* Spam texts are slightly longer on average.
* Word clouds showed recurring spam tokens: *free*, *win*, *call*.

📊 **Example Outputs (from `outputs/` folder):**

<img width="823" height="586" alt="label_distribution" src="https://github.com/user-attachments/assets/d0bd2d82-4859-42e8-89de-4049423611ed" />
<img width="1753" height="586" alt="message_length_distribution" src="https://github.com/user-attachments/assets/151ca134-bb9e-41f4-9120-059f8cca5d7d" />
<img width="1185" height="639" alt="wordcloud_spam" src="https://github.com/user-attachments/assets/2f12935d-aaeb-4e6a-a6b7-32baf1b5fa1a" />
<img width="1185" height="639" alt="wordcloud_ham" src="https://github.com/user-attachments/assets/7ac9409f-2532-4d34-9cbc-1fa87b266615" />

EDA confirmed that **F1-score** (not just accuracy) is the right metric to optimize.

---

## 🤖 Modeling

I evaluated multiple classifiers on **TF–IDF features**:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* K-Nearest Neighbors (KNN)
* XGBoost

To handle imbalance, experiments were run on:

* **Imbalanced dataset** (original distribution)
* **SMOTE-balanced dataset** (oversampled spam class)

📈 **Model Comparison**

<img width="1469" height="426" alt="Screenshot 2025-09-18 215600" src="https://github.com/user-attachments/assets/a13b20b7-322f-4856-8cb0-a79a696748e5" />

---

## 🌐 API (FastAPI)

The **FastAPI backend** exposes endpoints to use the models:

* `/predict-imbalanced` → trained on original data.
* `/predict-smote` → trained with SMOTE.

Each returns:

```json
{
  "prediction": "spam",
  "confidence": 0.94
}
```

▶️ Run with:

```bash
uvicorn app:app --reload
```

---

## 🖥️ GUI (PyQt5 Desktop App)

For a **desktop demo**, I built a PyQt5 GUI:

* Simple text box to type/paste an SMS
* Dropdown to pick the model (imbalanced vs SMOTE)
* Output shows prediction (ham/spam) + confidence score

📷 <img width="1919" height="1018" alt="Screenshot 2025-09-19 131144" src="https://github.com/user-attachments/assets/c16b5b1f-c566-42d2-9c3a-7e50a946e6ff" />

This makes the project usable offline as a small desktop tool.

---

## 📂 Project Structure

```
sms-spam-classification/
├── data/                    # Datasets
├── outputs/                 # EDA figures + processed CSV
├── models/                  # Saved models
├── src/                     # Source codes
├── app.py                   # FastAPI backend
├── gui.py                   # PyQt5 GUI
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

* Python 3.11+

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
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
fastapi
uvicorn
pyqt5
```

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/sms-spam-classification.git
   cd sms-spam-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run EDA (optional, regenerates figures):

   ```bash
   python Exploratory_Data_Analysis_polished.py --input data/Message_Classification.csv --output outputs
   ```
4. Try the notebook:

   ```bash
   jupyter notebook SMS_Spam_Classification.ipynb
   ```
5. Run API:

   ```bash
   uvicorn app:app --reload
   ```
6. Launch GUI:

   ```bash
   python gui.py
   ```

---

## 🔮 Future Work

* Try transformer-based models (BERT, DistilBERT).
* Add multilingual support.
* Package GUI as `.exe` for easier use.
* Dockerize API + GUI.

---

## 📜 License

MIT License. See LICENSE file.
