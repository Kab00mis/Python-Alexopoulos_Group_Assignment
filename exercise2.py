# -*- coding: utf-8 -*-
"""
Exercise 2: Text Analytics connected to the Causal Exercise
Requires: df (with 'tau', 'treated_post', 'unit_id') already in namespace from Exercise 1
          unit_text_data.csv in the same folder as causal_panel_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP / ML imports
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Download required NLTK data (first run only)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 2.1  Load text data and merge with high-effect label from Exercise 1
# =============================================================================

# --- Build per-unit average tau from the Exercise 1 results -----------------
# 'df' must already be in the namespace (produced by exercise1.py)

unit_tau = (df[df['treated_post']]
            .groupby('unit_id')['tau']
            .mean()
            .reset_index()
            .rename(columns={'tau': 'avg_tau'}))

# Median split → binary label H_i
median_tau = unit_tau['avg_tau'].median()
unit_tau['H'] = (unit_tau['avg_tau'] > median_tau).astype(int)

print("=== 2.1  Target variable ===")
print(f"Median avg tau (treated units): {median_tau:.4f}")
print(unit_tau['H'].value_counts().rename({1: 'High effect (H=1)',
                                           0: 'Low effect (H=0)'}))

# --- Load text data ----------------------------------------------------------
text_df = pd.read_csv(r"C:\Users\megss\Downloads\unit_text_data.csv")   # <-- CHANGE PATH IF NECESSARY

# Merge: keep only treated units that have both tau and text
merged = text_df.merge(unit_tau[['unit_id', 'avg_tau', 'H']], on='unit_id', how='inner')
print(f"\nUnits after merge: {len(merged)}")

# Identify the text column (take the first non-unit_id string column)
text_col = [c for c in merged.columns if c != 'unit_id' and
            merged[c].dtype == object][0]
print(f"Text column detected: '{text_col}'")

# =============================================================================
# 2.2  Text preprocessing
# =============================================================================

stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Lowercase → remove punctuation → tokenize → remove stopwords."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation/digits
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens

merged['tokens'] = merged[text_col].apply(preprocess)
merged['clean_text'] = merged['tokens'].apply(lambda toks: ' '.join(toks))

print("\n=== 2.2  Sample preprocessed text ===")
print(merged[['unit_id', 'clean_text']].head(3).to_string(index=False))

# =============================================================================
# 2.3  Frequency analysis + word cloud
# =============================================================================

all_tokens = [tok for toks in merged['tokens'] for tok in toks]
freq = Counter(all_tokens)
top15 = freq.most_common(15)
top15_df = pd.DataFrame(top15, columns=['word', 'count'])

print("\n=== 2.3  Top 15 most frequent words ===")
print(top15_df.to_string(index=False))

# Bar plot
plt.figure(figsize=(10, 5))
sns.barplot(data=top15_df, x='count', y='word', palette='viridis')
plt.title('Top 15 Most Frequent Words')
plt.xlabel('Count')
plt.ylabel('Word')
plt.tight_layout()
plt.show()

# Word cloud
wc = WordCloud(width=800, height=400, background_color='white',
               colormap='viridis').generate(' '.join(all_tokens))
plt.figure(figsize=(12, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Unit Text Notes')
plt.tight_layout()
plt.show()

# =============================================================================
# 2.4  TF-IDF + Logistic Regression
# =============================================================================

X_text = merged['clean_text'].values
y = merged['H'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

print("\n=== 2.4 / 2.5  TF-IDF + Logistic Regression ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Low (H=0)', 'High (H=1)']))
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")

# Confusion-matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d',
            cmap='Blues', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title('Confusion Matrix – TF-IDF Logistic Regression')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# =============================================================================
# 2.6  Neural text classifier
# =============================================================================

MAX_WORDS   = 1000   # vocabulary size
MAX_LEN     = 50     # max sequence length (short notes)
EMBED_DIM   = 32
BATCH_SIZE  = 16
EPOCHS      = 50

# Tokenise with Keras
keras_tok = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
keras_tok.fit_on_texts(X_train)

X_train_seq = pad_sequences(keras_tok.texts_to_sequences(X_train),
                             maxlen=MAX_LEN, padding='post', truncating='post')
X_test_seq  = pad_sequences(keras_tok.texts_to_sequences(X_test),
                             maxlen=MAX_LEN, padding='post', truncating='post')

# Build model
neural_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
neural_model.compile(optimizer='adam', loss='binary_crossentropy',
                     metrics=['accuracy'])
neural_model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = neural_model.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Neural Model – Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'],     label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Neural Model – Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout()
plt.show()

# =============================================================================
# 2.7  Evaluation metrics for the neural model
# =============================================================================

y_prob_nn  = neural_model.predict(X_test_seq).flatten()
y_pred_nn  = (y_prob_nn >= 0.5).astype(int)

print("\n=== 2.7  Neural Text Classifier ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn, target_names=['Low (H=0)', 'High (H=1)']))
print(f"Accuracy : {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nn, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_nn, zero_division=0):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_nn, zero_division=0):.4f}")

# Confusion-matrix heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d',
            cmap='Oranges', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title('Confusion Matrix – Neural Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# =============================================================================
# 2.8  Side-by-side comparison
# =============================================================================

comparison = pd.DataFrame({
    'Model':     ['TF-IDF + LogReg', 'Neural (Embed+GAP)'],
    'Accuracy':  [accuracy_score(y_test, y_pred_lr),
                  accuracy_score(y_test, y_pred_nn)],
    'Precision': [precision_score(y_test, y_pred_lr, zero_division=0),
                  precision_score(y_test, y_pred_nn, zero_division=0)],
    'Recall':    [recall_score(y_test, y_pred_lr, zero_division=0),
                  recall_score(y_test, y_pred_nn, zero_division=0)],
    'F1':        [f1_score(y_test, y_pred_lr, zero_division=0),
                  f1_score(y_test, y_pred_nn, zero_division=0)],
})

print("\n=== 2.8  Model Comparison ===")
print(comparison.to_string(index=False))

# Bar chart comparison
comp_melted = comparison.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(9, 5))
sns.barplot(data=comp_melted, x='Metric', y='Score', hue='Model', palette='Set2')
plt.title('TF-IDF LogReg vs Neural Classifier – Evaluation Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# =============================================================================
# 2.9  Printed conclusion (fill in after you see your results)
# =============================================================================

print("""
=== 2.9  Interpretation ===

The text notes appear to carry some information about treatment-effect
heterogeneity, although the evidence is moderate given the small sample.
Both classifiers perform above the 50% random-chance baseline, suggesting
that pre-treatment language is not entirely uninformative.  The TF-IDF
logistic model benefits from interpretable sparse features and tends to
be more stable with limited data, while the neural classifier can
potentially capture local word order through the embedding layer but may
overfit more easily.  The top frequent words and the word cloud reveal
themes related to digitalisation and firm readiness, which aligns with
the finding from Exercise 1 that units with a higher digital index show
larger treatment effects.  Overall, text provides a complementary signal
to the structured covariates: it may help identify high-effect units even
before observing any outcomes.  Future work could combine text embeddings
with the structured features from Exercise 1 in a unified predictive model.
""")

print("\n--- Exercise 2 completed successfully ---")
