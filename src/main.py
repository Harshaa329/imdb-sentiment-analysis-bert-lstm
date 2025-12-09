"""
main.py
Central pipeline for IMDb Sentiment Analysis:
- Loads dataset
- Preprocesses text
- Train-test split
- Runs ML, LSTM, BERT training
- Evaluates all models
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from preprocessing import preprocess_text, prepare_lstm_sequences, prepare_bert_inputs
from data_loader import load_imdb_dataset
from train_ml import train_ml_models
from train_lstm import train_lstm_model
from train_bert import train_bert_model
from evaluate import print_metrics, plot_confusion_matrix, plot_roc_curve


# -----------------------
# 1. Load Dataset
# -----------------------
df = load_imdb_dataset("IMDB Dataset.csv")
print(df.head())

# -----------------------
# 2. Preprocess Text
# -----------------------
df = preprocess_text(df)

# -----------------------
# 3. Train-Test Split
# -----------------------
X = df["clean_review"]
y = df["sentiment_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Split completed:", len(X_train), len(X_test))


# ---------------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODELS (TF-IDF + NB + SVM)
# ---------------------------------------------------------
tfidf_vectorizer, nb_model, svm_model = train_ml_models(X_train, y_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

y_pred_svm = svm_model.predict(X_test_tfidf)
y_prob_svm = svm_model.decision_function(X_test_tfidf)

print_metrics(y_test, y_pred_svm, "Linear SVM")
plot_confusion_matrix(y_test, y_pred_svm, "Linear SVM")
plot_roc_curve(y_test, y_prob_svm, "Linear SVM")


# ---------------------------------------------------------
# 5. TRAIN LSTM MODEL
# ---------------------------------------------------------
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = prepare_lstm_sequences(df, X_train, X_test, y_train, y_test)

lstm_model, history = train_lstm_model(
    X_train_lstm, y_train_lstm,
    X_test_lstm, y_test_lstm
)


# ---------------------------------------------------------
# 6. TRAIN BERT MODEL
# ---------------------------------------------------------
(X_train_ids, X_test_ids,
 X_train_masks, X_test_masks,
 y_train_bert, y_test_bert) = prepare_bert_inputs(df, X_train, X_test, y_train, y_test)

bert_results = train_bert_model(
    X_train_ids, X_train_masks, y_train_bert,
    X_test_ids, X_test_masks, y_test_bert
)

y_pred_bert, y_prob_bert = bert_results


print_metrics(y_test_bert, y_pred_bert, "BERT Model")
plot_confusion_matrix(y_test_bert, y_pred_bert, "BERT Model")
plot_roc_curve(y_test_bert, y_prob_bert, "BERT Model")


print("\n\n🎉 All models trained & evaluated successfully!")
