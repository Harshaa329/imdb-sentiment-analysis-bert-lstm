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

# Local src imports
from src.preprocessing import (
    clean_text,
    remove_stopwords,
    lemmatize_text,
    get_lstm_sequences,
    get_bert_tokens
)

from src.data_loader import load_imdb_dataset
from src.train_ml import train_ml_models
from src.train_lstm import train_lstm_model
from src.train_bert import train_bert_model
from src.evaluate import print_metrics, plot_confusion_matrix, plot_roc_curve


# --------------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------------
df = load_imdb_dataset("IMDB Dataset.csv")
print("\nDataset loaded:")
print(df.head())


# --------------------------------------------------------
# 2. PREPROCESS TEXT
# --------------------------------------------------------
df["clean_review"] = df["review"].apply(clean_text)
df["clean_no_stopwords"] = df["clean_review"].apply(remove_stopwords)
df["lemmatized_review"] = df["clean_no_stopwords"].apply(lemmatize_text)

# Binary labels
df["sentiment_binary"] = df["sentiment"].map({"negative": 0, "positive": 1})

print("\nPreprocessing complete!")


# --------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------------
X = df["lemmatized_review"]
y = df["sentiment_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nSplit completed:", len(X_train), "train,", len(X_test), "test")


# --------------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODELS (TF-IDF + NB + SVM)
# --------------------------------------------------------
tfidf_vectorizer, nb_model, svm_model = train_ml_models(X_train, y_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

y_pred_svm = svm_model.predict(X_test_tfidf)
y_prob_svm = svm_model.decision_function(X_test_tfidf)

print_metrics(y_test, y_pred_svm, "Linear SVM")
plot_confusion_matrix(y_test, y_pred_svm, "Linear SVM")
plot_roc_curve(y_test, y_prob_svm, "Linear SVM")


# --------------------------------------------------------
# 5. TRAIN LSTM MODEL
# --------------------------------------------------------
X_train_lstm, tokenizer = get_lstm_sequences(X_train)
X_test_lstm, _ = get_lstm_sequences(X_test)

lstm_model, history = train_lstm_model(
    X_train_lstm, y_train.values,
    X_test_lstm, y_test.values
)


# --------------------------------------------------------
# 6. TRAIN BERT MODEL
# --------------------------------------------------------
bert_inputs = get_bert_tokens(df["clean_review"])
input_ids = bert_inputs["input_ids"]
attention_masks = bert_inputs["attention_mask"]

# split using same indices
train_idx = X_train.index
test_idx = X_test.index

X_train_ids = input_ids[train_idx]
X_test_ids = input_ids[test_idx]
X_train_masks = attention_masks[train_idx]
X_test_masks = attention_masks[test_idx]

y_train_bert = y_train.values
y_test_bert = y_test.values

y_pred_bert, y_prob_bert = train_bert_model(
    X_train_ids, X_train_masks, y_train_bert,
    X_test_ids, X_test_masks, y_test_bert
)

print_metrics(y_test_bert, y_pred_bert, "BERT Model")
plot_confusion_matrix(y_test_bert, y_pred_bert, "BERT Model")
plot_roc_curve(y_test_bert, y_prob_bert, "BERT Model")


print("\n\n🎉 All models trained & evaluated successfully!")
