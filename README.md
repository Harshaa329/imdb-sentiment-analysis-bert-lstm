# 🎬 IMDb Sentiment Analysis with ML, LSTM & BERT

This project performs sentiment classification of IMDb movie reviews using multiple approaches — including traditional Machine Learning models, a Deep Learning LSTM model, and a modern Transformer-based BERT model. The purpose is to compare performance across different techniques and evaluate trade-offs between accuracy, complexity, and training cost.

---

## 🚀 Project Highlights
- 📊 Dataset: 50,000 labeled IMDb movie reviews (binary sentiment)
- 🧠 Models implemented:
  - Logistic Regression, SVM, RandomForest
  - LSTM (Deep Learning)
  - BERT (Transformer model)
- 🏆 Best Model Performance: **BERT – Accuracy 92%**
- 🔍 Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve
- 🎯 Use Case: Customer sentiment analysis for reviews on platforms like Amazon, IMDb, TripAdvisor

---

## 📂 Repository Structure

---

## 📈 Model Performance Comparison

| Model | Accuracy | F1 Score |
|--------|---------|---------|
| Logistic Regression | 0.85 | 0.84 |
| LSTM | 0.88 | 0.87 |
| **BERT** | **0.92** | **0.92** |

---

## 🧪 Tech Stack
- Python, NumPy, Pandas, Scikit-learn
- TensorFlow / Keras
- Transformers (BERT)
- Matplotlib / Seaborn / Plotly

---

## 🛠️ Future Enhancements
- Add Streamlit inference web application
- Hyperparameter tuning & dropout optimization
- MLflow experiment tracking

---

## 🙋‍♀️ Author
**Harshaa Hariharan**  
Machine Learning & Data Science  
LinkedIn: *www.linkedin.com/in/harshaa-harshini-64522530hbc329*  
Portfolio Website: *(coming soon)*

---

## 📊 Model Evaluation

### Sentiment Class Distribution
![Sentiment Distribution](Visuals/class_distribution.png)

### Performance Comparison
![Model Performance Comparison](visuals/performance_comparison_baseline_lstm_bert.png)

### BERT Loss Curve
![BERT Loss Curve](visuals/bert_loss_curve.png)

### BERT ROC Curve
![BERT ROC Curve](visuals/bert_roc_curve.png)

### BERT Confusion Matrix
![BERT Confusion Matrix](visuals/bert_confusion_matrix.png)

