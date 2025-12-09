"""
Train classical Machine Learning models:
- TF-IDF Vectorizer
- Naïve Bayes (MultinomialNB)
- Linear Support Vector Machine (LinearSVC)

Returns:
    tfidf_vectorizer, nb_model, svm_model
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_ml_models(X_train, y_train, max_features=10000):
    """
    Train TF-IDF, Naïve Bayes & Linear SVM on classical text features.

    Args:
        X_train (list or pd.Series): Clean text training data
        y_train (list or pd.Series): Binary sentiment labels
        max_features (int): TF-IDF vocabulary size

    Returns:
        tfidf_vectorizer, nb_model, svm_model
    """

    # ----------------------------------
    # TF–IDF Vectorization
    # ----------------------------------
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words='english'
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    print("TF-IDF training matrix shape:", X_train_tfidf.shape)

    # ----------------------------------
    # Train Naïve Bayes
    # ----------------------------------
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # ----------------------------------
    # Train Linear SVM
    # ----------------------------------
    svm_model = LinearSVC()
    svm_model.fit(X_train_tfidf, y_train)

    print("Training done for both ML models.")

    return tfidf_vectorizer, nb_model, svm_model
