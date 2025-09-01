# ---
# Título: Clasificador de Interés de Búsquedas con NLP y Deep Learning
# Autor: [Tu Nombre]
# ---

# --- 1. SETUP: IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# --- SOLUCIÓN PARA EL CUELGUE DE TENSORFLOW ---
# Esta línea deshabilita una optimización que puede causar problemas en algunos sistemas
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

# Descargar recursos y cargar modelos
nltk.download('stopwords', quiet=True)
try:
    nlp = spacy.load('es_core_news_sm')
except OSError:
    print('Modelo de spaCy no encontrado. Descargando...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
    nlp = spacy.load('es_core_news_sm')


# --- 2. CARGA DE DATOS Y CREACIÓN DE LA VARIABLE OBJETIVO ---
FILE_PATH = 'bq-results.csv' # <-- Archivo actualizado
print(f"Cargando dataset desde: {FILE_PATH}")
df = pd.read_csv(FILE_PATH)

# Ingeniería de Característica: Crear la clase de interés a partir del CTR
df['ctr'] = df['clicks'] / (df['impressions'] + 1)
median_ctr = df['ctr'].median()
df['clase_interes'] = (df['ctr'] > median_ctr).astype(int)
print(f"\nUmbral de CTR (mediana) para 'Alto Interés': {median_ctr:.4f}")

# Definir columnas
TEXT_COLUMN = 'query'
LABEL_COLUMN = 'clase_interes'
df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)


# --- 3. PREPROCESAMIENTO NLP CON SPACY Y NLTK ---
print("\n--- Iniciando Preprocesamiento NLP ---")
stop_words = set(stopwords.words('spanish'))

def preprocess_text_spacy(text):
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    doc = nlp(text)
    # Lematización y eliminación de stopwords
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return ' '.join(lemmas)

df['query_limpia'] = df[TEXT_COLUMN].apply(preprocess_text_spacy)

print("\nEjemplo de preprocesamiento:")
print("Original:", df[TEXT_COLUMN].iloc[10])
print("Procesada:", df['query_limpia'].iloc[10])


# --- 4. DIVISIÓN DE DATOS ---
X = df['query_limpia']
y = df[LABEL_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 5. MODELADO CON MACHINE LEARNING (BASELINE) ---
print("\n--- Entrenando Modelos Baseline con Scikit-Learn ---")

# --- Modelo 1: Bag of Words (BoW) ---
print("\n--- Modelo 1: Regresión Logística con Bag of Words ---")
bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)
print(f"Accuracy (BoW): {accuracy_score(y_test, y_pred_bow):.4f}")

# --- Modelo 2: TF-IDF ---
print("\n--- Modelo 2: Regresión Logística con TF-IDF ---")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
print(f"Accuracy (TF-IDF): {accuracy_score(y_test, y_pred_tfidf):.4f}")
print("\nReporte de Clasificación (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf, target_names=['Bajo Interés', 'Alto Interés']))


# --- 6. MODELADO CON DEEP LEARNING ---
print("\n\n--- Iniciando Proceso de Deep Learning ---")
# Hiperparámetros
VOCAB_SIZE = 10000
MAX_LENGTH = 30 # Las queries suelen ser cortas
EMBEDDING_DIM = 64
EPOCHS = 10
BATCH_SIZE = 32

# Tokenización y Padding
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Construcción de la Red Neuronal
model_dl = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dl.summary()

# Checkpoint para guardar el mejor modelo
checkpoint_path = 'best_dl_model.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

print("\nEntrenando la red neuronal...")
history = model_dl.fit(train_padded, y_train,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_data=(test_padded, y_test),
                       callbacks=[checkpoint_callback],
                       verbose=2)

# Evaluación del mejor modelo
print("\nCargando y evaluando el mejor modelo de Deep Learning...")
model_dl.load_weights(checkpoint_path)
loss, accuracy = model_dl.evaluate(test_padded, y_test)
print(f"\n--- Resultados del Modelo de Deep Learning (Mejor Checkpoint) ---")
print(f'Accuracy en prueba: {accuracy:.4f}')
print(f'Pérdida en prueba: {loss:.4f}')

y_pred_dl = (model_dl.predict(test_padded) > 0.5).astype("int32")
print("\nReporte de Clasificación (Deep Learning):")
print(classification_report(y_test, y_pred_dl, target_names=['Bajo Interés', 'Alto Interés']))

print("\n--- FIN DEL PROYECTO ---")

