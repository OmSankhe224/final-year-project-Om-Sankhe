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

# 📈 Results Snapshot

## 📊 Comprehensive Model Comparison

| Model                    | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------------------------|----------|-----------|--------|----------|---------|
| **XGBoost (SMOTE)**       | **0.9953** | **0.9355** | **0.9667** | **0.9508** | **0.9967** |
| Random Forest             | 0.9937   | 0.9333    | 0.9333 | 0.9333   | 0.9977  |
| Neural Network (RMSPROP)  | 0.9889   | 0.8966    | 0.8667 | 0.8814   | 0.9768  |
| CNN                       | 0.9842   | 0.8333    | 0.8333 | 0.8333   | 0.9956  |
| Decision Tree (SMOTE)     | 0.9842   | 0.8125    | 0.8667 | 0.8387   | 0.9284  |
| GRU                       | 0.9842   | 0.7778    | 0.9333 | 0.8485   | 0.9794  |
| Logistic Regression       | 0.9826   | 0.8519    | 0.7667 | 0.8070   | 0.9945  |
| SVM (GridSearch)          | 0.9763   | 0.7586    | 0.7333 | 0.7458   | 0.9861  |
| Neural Network (ADAM)     | 0.9747   | 0.7188    | 0.7667 | 0.7419   | 0.9718  |

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

