import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('wordnet')


# --------------------------------------------------
# 1. Text Cleaning
# --------------------------------------------------
def clean_text(text):
    """
    Clean raw text by removing HTML tags, symbols, numbers, and extra spaces.
    Converts text to lowercase.
    """
    text = re.sub(r'<.*?>', '', text)                   # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)            # remove non-alphabetic chars
    text = text.lower()                                 # lowercase
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra whitespace
    return text


# --------------------------------------------------
# 2. Stopword Removal (for classical ML models)
# --------------------------------------------------
def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)


# --------------------------------------------------
# 3. Lemmatization
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


# --------------------------------------------------
# 4. TF-IDF Vectorizer (for SVM / Naïve Bayes)
# --------------------------------------------------
def get_tfidf_vectors(text_data):
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_tfidf = tfidf_vectorizer.fit_transform(text_data)
    return X_tfidf, tfidf_vectorizer


# --------------------------------------------------
# 5. Tokenizer for LSTM (Deep Learning)
# --------------------------------------------------
def get_lstm_sequences(text_data, num_words=5000, max_len=50):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<oov>')
    tokenizer.fit_on_texts(text_data)

    sequences = tokenizer.texts_to_sequences(text_data)
    padded = pad_sequences(sequences, padding="post", maxlen=max_len)
    return padded, tokenizer


# --------------------------------------------------
# 6. Tokenizer for BERT
# --------------------------------------------------
def get_bert_tokens(text_data):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_inputs = bert_tokenizer(
        text_data.tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return bert_inputs, bert_tokenizer
