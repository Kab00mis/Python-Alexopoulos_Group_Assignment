# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:38:57 2026

@author: Bagel
"""
# =============================================================================
# EXERCISE 2: TEXT ANALYTICS CONNECTED TO CAUSAL EXERCISE
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, GlobalAveragePooling1D, Dense)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =============================================================================
# ΒΗΜΑ 0: ΦΟΡΤΩΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΑΠΟ TASK 1
# =============================================================================

# Φορτώνουμε τα πραγματικά αποτελέσματα από το Task 1
tau_unit = pd.read_csv(r"C:\Users\Bagel\Desktop\Python_Task_2\unit_effect_labels.csv")
tau_unit = tau_unit.rename(columns={'tau_i': 'avg_tau', 'H_i': 'high_effect'})

# =============================================================================
# ΒΗΜΑ 1: ΔΗΜΙΟΥΡΓΙΑ BINARY LABEL Hi
# =============================================================================

median_tau = tau_unit['avg_tau'].median()
print(f"Διάμεσος treatment effect: {median_tau:.4f}")
print(f"Κατανομή Hi:\n{tau_unit['high_effect'].value_counts()}")

# =============================================================================
# TASK 2.1: ΦΟΡΤΩΣΗ ΚΕΙΜΕΝΩΝ ΚΑΙ MERGE ΜΕ ΤΟ LABEL
# =============================================================================

df_text = pd.read_csv(r"C:\Users\Bagel\Desktop\Python_Task_2\unit_text_data.csv")
print(f"\nΔιαστάσεις text data: {df_text.shape}")
print(df_text.head(3))

df = df_text.merge(tau_unit[['unit_id', 'high_effect']], on='unit_id', how='inner')

print(f"\nΔιαστάσεις μετά το merge (treated μόνο): {df.shape}")
print(f"Κατανομή high_effect:\n{df['high_effect'].value_counts()}")

# =============================================================================
# TASK 2.2: ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ ΚΕΙΜΕΝΟΥ
# =============================================================================

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

df['tokens']     = df['text_note'].apply(preprocess_text)
df['clean_text'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

print("\nΠαράδειγμα προεπεξεργασμένου κειμένου:")
print(df['clean_text'].iloc[0][:200])

# =============================================================================
# TASK 2.3: ΣΥΧΝΟΤΗΤΑ ΛΕΞΕΩΝ, BAR PLOT, WORD CLOUD
# =============================================================================

all_tokens = [token for tokens in df['tokens'] for token in tokens]
word_freq  = Counter(all_tokens)
top_15     = word_freq.most_common(15)
top_words, top_counts = zip(*top_15)

print("\n15 πιο συχνές λέξεις:")
for word, count in top_15:
    print(f"  {word}: {count}")

# --- Bar plot με μωβ-τιρκουάζ παλέτα ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor('#EEF2F7')

bar_colors = plt.cm.cool(np.linspace(0.2, 0.9, 15))
axes[0].barh(top_words[::-1], top_counts[::-1], color=bar_colors,
             edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('Συχνότητα', fontsize=11)
axes[0].set_title('Top 15 Πιο Συχνές Λέξεις', fontsize=12, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3, color='white')
axes[0].set_facecolor('#D8E8F0')

# --- Word Cloud με πορτοκαλί παλέτα και σκούρο φόντο ---
all_text_combined = ' '.join(df['clean_text'])
wordcloud = WordCloud(
    width=800, height=400,
    background_color='#1A1A2E',
    max_words=100,
    colormap='plasma',
    contour_color='white',
    contour_width=1
).generate(all_text_combined)

axes[1].imshow(wordcloud, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Word Cloud', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(r"C:\Users\Bagel\Desktop\Python_Task_2\wordfreqcloud.png", dpi=150)
plt.show()
print("Αποθηκεύτηκε: wordfreqcloud.png")

# =============================================================================
# TASK 2.4: TF-IDF + LOGISTIC REGRESSION
# =============================================================================

X = df['clean_text']
y = df['high_effect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set: {len(X_train)} δείγματα, Test set: {len(X_test)} δείγματα")

tfidf         = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

# =============================================================================
# TASK 2.5: ΑΞΙΟΛΟΓΗΣΗ TF-IDF ΜΟΝΤΕΛΟΥ
# =============================================================================

print("\n" + "="*50)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ: TF-IDF + Logistic Regression")
print("="*50)

cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"Confusion Matrix:\n{cm_lr}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_lr)}")
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {acc_lr:.4f}")

# Confusion matrix με πορτοκαλί παλέτα
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('#EEF2F7')
disp = ConfusionMatrixDisplay(cm_lr, display_labels=['Low Effect', 'High Effect'])
disp.plot(ax=ax, colorbar=True, cmap='YlOrBr')
ax.set_title('Confusion Matrix — TF-IDF + Logistic Regression',
             fontsize=11, fontweight='bold')
ax.set_facecolor('#EEF2F7')
plt.tight_layout()
plt.savefig(r"C:\Users\Bagel\Desktop\Python_Task_2\logreg.png", dpi=150)
plt.show()

# =============================================================================
# TASK 2.6: NEURAL TEXT CLASSIFIER
# =============================================================================

VOCAB_SIZE    = 1000
MAX_LEN       = 100
EMBEDDING_DIM = 32

tokenizer_nn = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer_nn.fit_on_texts(X_train)

X_train_seq = tokenizer_nn.texts_to_sequences(X_train)
X_test_seq  = tokenizer_nn.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN,
                             padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN,
                             padding='post', truncating='post')

y_train_arr = np.array(y_train)
y_test_arr  = np.array(y_test)

nn_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM,
              input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(1,  activation='sigmoid')
])

nn_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
nn_model.summary()

history = nn_model.fit(
    X_train_pad, y_train_arr,
    validation_split=0.1,
    epochs=20,
    batch_size=16,
    verbose=1
)

# =============================================================================
# TASK 2.7: ΑΞΙΟΛΟΓΗΣΗ NEURAL ΜΟΝΤΕΛΟΥ
# =============================================================================

y_pred_nn_prob = nn_model.predict(X_test_pad).flatten()
y_pred_nn      = (y_pred_nn_prob >= 0.5).astype(int)

print("\n" + "="*50)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ: Neural Text Classifier")
print("="*50)

cm_nn = confusion_matrix(y_test_arr, y_pred_nn)
print(f"Confusion Matrix:\n{cm_nn}")
print(f"\nClassification Report:\n{classification_report(y_test_arr, y_pred_nn)}")
acc_nn = accuracy_score(y_test_arr, y_pred_nn)
print(f"Accuracy: {acc_nn:.4f}")

# Confusion matrix με τιρκουάζ παλέτα
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor('#EEF2F7')
disp2 = ConfusionMatrixDisplay(cm_nn, display_labels=['Low Effect', 'High Effect'])
disp2.plot(ax=ax, colorbar=True, cmap='GnBu')
ax.set_title('Confusion Matrix — Neural Text Classifier',
             fontsize=11, fontweight='bold')
ax.set_facecolor('#EEF2F7')
plt.tight_layout()
plt.savefig(r"C:\Users\Bagel\Desktop\Python_Task_2\neural.png", dpi=150)
plt.show()

# Καμπύλες εκπαίδευσης με διαφορετικά χρώματα
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#EEF2F7')

axes[0].plot(history.history['accuracy'],     color='#6B2D8B', linewidth=2.5,
             marker='o', markersize=4, label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], color='#F4A261', linewidth=2.5,
             marker='s', markersize=4, linestyle='--', label='Val Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Neural Model — Accuracy κατά την εκπαίδευση',
                  fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3, color='white')
axes[0].set_facecolor('#D8E8F0')

axes[1].plot(history.history['loss'],     color='#6B2D8B', linewidth=2.5,
             marker='o', markersize=4, label='Train Loss')
axes[1].plot(history.history['val_loss'], color='#F4A261', linewidth=2.5,
             marker='s', markersize=4, linestyle='--', label='Val Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Neural Model — Loss κατά την εκπαίδευση',
                  fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3, color='white')
axes[1].set_facecolor('#D8E8F0')

plt.tight_layout()
plt.savefig(r"C:\Users\Bagel\Desktop\Python_Task_2\training_curves.png", dpi=150)
plt.show()

# =============================================================================
# TASK 2.8: ΣΥΓΚΡΙΣΗ ΤΩΝ ΔΥΟ ΜΟΝΤΕΛΩΝ
# =============================================================================

comparison = pd.DataFrame({
    'Model': ['TF-IDF + Logistic Regression', 'Neural Text Classifier'],
    'Accuracy':  [accuracy_score(y_test, y_pred_lr),
                  accuracy_score(y_test_arr, y_pred_nn)],
    'Precision': [precision_score(y_test, y_pred_lr, average='weighted'),
                  precision_score(y_test_arr, y_pred_nn, average='weighted')],
    'Recall':    [recall_score(y_test, y_pred_lr, average='weighted'),
                  recall_score(y_test_arr, y_pred_nn, average='weighted')],
    'F1-Score':  [f1_score(y_test, y_pred_lr, average='weighted'),
                  f1_score(y_test_arr, y_pred_nn, average='weighted')]
})

print("\n" + "="*50)
print("ΣΥΓΚΡΙΣΗ ΜΟΝΤΕΛΩΝ")
print("="*50)
print(comparison.to_string(index=False))

# Bar chart με διαφορετικά χρώματα από το αρχικό
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x       = np.arange(len(metrics))
width   = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#EEF2F7')

bars1 = ax.bar(x - width/2, comparison.iloc[0][metrics], width,
               label='TF-IDF + LR', color='#6B2D8B', edgecolor='white')
bars2 = ax.bar(x + width/2, comparison.iloc[1][metrics], width,
               label='Neural',       color='#F4A261', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Σύγκριση TF-IDF LR vs Neural Classifier',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, color='white')
ax.set_facecolor('#D8E8F0')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(r"C:\Users\Bagel\Desktop\Python_Task_2\modelcomparison.png", dpi=150)
plt.show()
print("Αποθηκεύτηκε: modelcomparison.png")

# =============================================================================
# TASK 2.9: ΕΡΜΗΝΕΙΑ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
# =============================================================================

print("""
=== TASK 2.9: ΕΡΜΗΝΕΙΑ (να αναπτυχθεί στην αναφορά) ===

Σημεία που πρέπει να καλύψεις σε 5-8 προτάσεις:
1. Είναι το κείμενο πληροφοριακό για το treatment effect heterogeneity;
2. Ποιο μοντέλο τα πήγε καλύτερα και γιατί;
3. Τι υποδηλώνουν οι top λέξεις για τα χαρακτηριστικά high-effect units;
4. Περιορισμοί: μικρό sample size, balanced classes;
5. Πρακτικές συνέπειες για policy targeting.
""")

print("\n=== ΟΛΟΚΛΗΡΩΘΗΚΕ ΤΟ EXERCISE 2 ===")
