# 🧠 Multi-Class Thyroid Function Classification Using Lab Test Data with Machine Learning and Deep Learning

## 🔍 Project Overview

Thyroid dysfunctions, especially **hypothyroidism**, can significantly affect metabolism, cognition, mood, and heart health. However, due to overlapping symptoms like fatigue, weight changes, and cold sensitivity, thyroid disorders are often **misdiagnosed**.

This project presents a **machine learning and deep learning-based solution** that classifies thyroid function using **clinical lab test data**, aiming to assist in **early and accurate diagnosis**.

---

## 📊 Dataset

- **Source**: [UCI Thyroid Disease Repository](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)
- **Files Used**: `hypothyroid.data`, `allhyper.data`
- **Final Records**: 3,100+ patient records
- **Features**: 18 lab and clinical attributes, including:
  - TSH, T3, TT4, T4U, FTI
  - Patient age, sex, medication status, pregnancy, etc.
- **Target Classes**:
  - `0`: Negative (Normal)
  - `1`: Hypothyroid

(Note: Hyperthyroid class was dropped due to extreme imbalance.)

---

## 🧼 Data Preprocessing

- Replaced `?` with NaN
- Converted numerical columns to float
- Handled missing values using **mean imputation**
- Dropped irrelevant or mostly empty features (e.g., `TBG`)
- Encoded categorical features (`f`/`t`, `M`/`F`) to `0`/`1`
- Applied `StandardScaler` to normalize numerical values

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized **class imbalance** and correlations
- Plotted:
  - Correlation heatmap
  - Boxplots by class for key features (TSH, TT4, FTI)
  - KDE plots for feature distribution
- Observed strong correlations between TT4, T3, and FTI

---

## 🤖 Machine Learning Models

Models implemented:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Custom Neural Network (DNN)
- GRU & CNN

All models were trained using the same preprocessed data.

---

## 🧠 Deep Learning Architectures

### 🔹 Custom DNN:
- 3 Dense layers
- ReLU + Sigmoid activations
- Dropout layers to prevent overfitting
- Optimized using `Adam` and `binary_crossentropy`

### 🔹 Tuned CNN:
- 2 Conv1D + MaxPooling + Dropout
- Flatten → Dense → Output
- Used EarlyStopping to avoid overfitting
- Best performance among all models

---

## 📊 Evaluation Metrics

All models evaluated using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC-AUC Curve**

📌 **CNN Model Accuracy:** ~98%  
📌 **Random Forest & SVM:** ~96%  
📌 **Logistic Regression:** ~94%

---

## 📈 Results Snapshot

## 📊 Results Snapshot
| Model                | Accuracy | F1-score (Hypothyroid) |
|----------------------|----------|-------------------------|
| Logistic Regression  | 94%      | 0.72                   |
| SVM                  | 96%      | 0.71                   |
| Random Forest        | 96%      | 0.92                   |
| Neural Network (DNN) | 98%      | 0.81                   |
| GRU                  | 98%      | 0.78                   |
| CNN                  | 96%      | 0.53                   |
| **XGBoost (SMOTE)**  | **99%**  | **0.94**               |

---

## 🏥 Real-World Use Case

> A hospital doctor could input a patient’s blood test data into this system, and the model will instantly predict thyroid condition (normal or hypothyroid), saving time and reducing human error.

---

## 🔧 Tools & Libraries Used

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- TensorFlow & Keras
- Imbalanced-learn (optional SMOTE)
- XGBoost
- Google Colab

---

## 📌 Folder Structure

