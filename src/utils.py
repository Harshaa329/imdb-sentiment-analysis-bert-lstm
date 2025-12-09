"""
utils.py
General-purpose helper functions used across preprocessing,
model training (ML, LSTM, BERT), evaluation, and utilities.
"""

import numpy as np
import datetime
import os


# -----------------------------------------------------------
# 1. Time Formatting Utility
# -----------------------------------------------------------
def format_time(seconds):
    """
    Converts seconds into hh:mm:ss formatting.
    Useful for long training loops (LSTM / BERT).
    """
    return str(datetime.timedelta(seconds=int(seconds)))


# -----------------------------------------------------------
# 2. Directory Creation Helper
# -----------------------------------------------------------
def safe_mkdir(path):
    """
    Creates a folder if it does not already exist.
    Prevents errors when saving models, logs, plots, etc.
    """
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------------------------------------
# 3. LSTM Helper Functions
# -----------------------------------------------------------
def stack_lstm_sequences(df, index_list):
    """
    Convert stored token sequences (df['lstm_tokens']) into
    a consistent NumPy 2D array while preserving train/test split alignment.
    """
    return np.vstack(df.loc[index_list, "lstm_tokens"].values)


# -----------------------------------------------------------
# 4. BERT Helper Functions
# -----------------------------------------------------------
def pad_or_truncate(token_ids, max_len=256):
    """
    Pads or truncates token ID lists to a fixed length.

    BERT requires uniform sequence length for batching.
    Padding token = 0 (BERT standard).
    """
    if len(token_ids) >= max_len:
        return token_ids[:max_len]
    else:
        return token_ids + [0] * (max_len - len(token_ids))


def create_attention_mask(input_ids):
    """
    Creates attention masks for BERT:
      1 → real token
      0 → padding token
    """
    return [1 if token != 0 else 0 for token in input_ids]


# -----------------------------------------------------------
# 5. Pretty Print Helpers
# -----------------------------------------------------------
def print_header(title):
    """
    Prints clean title sections in console logs.
    """
    print("\n" + "="*60)
    print(title)
    print("="*60)
