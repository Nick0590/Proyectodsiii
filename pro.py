#SETUP: IMPORTACIÓN DE LIBRERÍAS Y CONFIG

# Importo pandas para manejar los datos en formato de tabla (DataFrame).
import pandas as pd
# Importo numpy para operaciones numéricas
import numpy as np
# Importo 're' para usar expresiones regulares y limpiar el texto.
import re
# Importo nltk,para manejar las 'stopwords'.
import nltk
# Específicamente, importo la lista de 'stopwords' (palabras comunes como 'el', 'la', 'y').
from nltk.corpus import stopwords
# Importo spacy para la lematización.
import spacy
# De scikit-learn, importo la función para dividir los datos en entrenamiento y prueba.
from sklearn.model_selection import train_test_split
# importo los métodos para convertir texto en vectores numéricos: Bag of Words y TF-IDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# importo el modelo de Regresión Logística, que será mi modelo base.
from sklearn.linear_model import LogisticRegression
# importo las métricas para evaluar qué tan buenos son mis modelos.
from sklearn.metrics import classification_report, accuracy_score
# Importo TensorFlow, la librería principal para construir la red neuronal.
import tensorflow as tf
# De Keras (dentro de TensorFlow), importo las herramientas para procesar texto para la red.
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# De Keras, importo las piezas para construir mi modelo de Deep Learning.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
# para guardar automáticamente el mejor modelo durante el entrenamiento.
from tensorflow.keras.callbacks import ModelCheckpoint

# caso especifico Mac
# Aplico una configuración específica para TensorFlow en mi Mac para evitar que se congele.
tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})

# Descargo la lista de 'stopwords' de NLTK si es necesario, de forma silenciosa.
nltk.download('stopwords', quiet=True)
# Intento cargar el modelo de español de spaCy.
try:
    nlp = spacy.load('es_core_news_sm')
# Si no está instalado, lo descargo automáticamente y luego lo cargo.
except OSError:
    print('Modelo de spaCy no encontrado. Descargando...')
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'es_core_news_sm'])
    nlp = spacy.load('es_core_news_sm')

# Defino el nombre de mi archivo de datos para poder cambiarlo fácilmente si es necesario.
FILE_PATH = 'bq-results.csv'
print(f"Cargando dataset desde: {FILE_PATH}")
# Cargo el archivo CSV en un DataFrame de pandas.
df = pd.read_csv(FILE_PATH)

# Creo una nueva columna 'ctr' (Click-Through Rate) dividiendo clics por impresiones.
df['ctr'] = df['clicks'] / (df['impressions'] + 1)
# Calculo la mediana del CTR, que usaré como umbral para separar las clases.
median_ctr = df['ctr'].median()
# Creo mi columna objetivo 'clase_interes': 1 si el CTR es mayor que la mediana, 0 si no.
df['clase_interes'] = (df['ctr'] > median_ctr).astype(int)
print(f"\nUmbral de CTR (mediana) para 'Alto Interés': {median_ctr:.4f}")

# Defino los nombres de las columnas de texto y de la etiqueta para usarlos más adelante.
TEXT_COLUMN = 'query'
LABEL_COLUMN = 'clase_interes'
# Elimino cualquier fila que no tenga texto o etiqueta para evitar errores.
df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)


print("\n--- Iniciando Preprocesamiento NLP ---")
# Cargo la lista de palabras vacías en español desde NLTK.
stop_words = set(stopwords.words('spanish'))

# Defino una función para limpiar y procesar cada texto.
def preprocess_text_spacy(text):
    # Convierto el texto a minúsculas y me aseguro de que sea un string
    text = str(text).lower()
    # Elimino cualquier caracter que no sea una letra del alfabeto español o un espacio
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)
    # Proceso el texto con el modelo de spacy
    doc = nlp(text)
    # Lematizo cada palabra y elimino las stopwords.
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    # Uno las palabras lematizadas de nuevo en un solo string.
    return ' '.join(lemmas)

# Aplico esta función a toda la columna de 'query' para crear una nueva columna con el texto limpio.
df['query_limpia'] = df[TEXT_COLUMN].apply(preprocess_text_spacy)

# Muestro un ejemplo para verificar que el preprocesamiento funcionó.
print("\nEjemplo de preprocesamiento:")
print("Original:", df[TEXT_COLUMN].iloc[10])
print("Procesada:", df['query_limpia'].iloc[10])


# --- 4. DIVISIÓN DE DATOS ---

# Separo mis datos: X es el texto limpio (features) y 'y' es la etiqueta (target).
X = df['query_limpia']
y = df[LABEL_COLUMN]
# Divido los datos en conjuntos de entrenamiento (80%) y prueba (20%), asegurando que la proporción de clases se mantenga.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#MODELADO CON MACHINE LEARNING (BASELINE) 

print("\n Entrenando Modelos Baseline con Scikit-Learn ")

# Modelo 1: Bag of Words (BoW) 
print("\n Modelo 1: Regresión Logística con Bag of Words ")
# Inicializo el vectorizador Bag of Words, limitando el vocabulario a las 5000 palabras más frecuentes.
bow_vectorizer = CountVectorizer(max_features=5000)
# Aprendo el vocabulario y transformo los datos de entrenamiento a vectores BoW
X_train_bow = bow_vectorizer.fit_transform(X_train)
# Transformo los datos de prueba usando el mismo vocabulario aprendido
X_test_bow = bow_vectorizer.transform(X_test)
# Inicializo el modelo de Regresión Logística.
model_bow = LogisticRegression(max_iter=1000)
# Entreno el modelo con los datos de entrenamiento vectorizados.
model_bow.fit(X_train_bow, y_train)
# Hago predicciones sobre los datos de prueba
y_pred_bow = model_bow.predict(X_test_bow)
# Imprimo la precisión del modelo
print(f"Accuracy (BoW): {accuracy_score(y_test, y_pred_bow):.4f}")

# --- Modelo 2: TF-IDF ---
print("\n Modelo 2: Regresión Logística con TF-IDF ")
# Inicializo el vectorizador TF-IDF, similar al BoW.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# Aprendo y transformo los datos de entrenamiento a vectores TF-IDF.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Transformo los datos de prueba.
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Inicializo y entreno el modelo de Regresión Logística.
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)
# Hago predicciones.
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
# Imprimo la precisión y un reporte más detallado para este modelo.
print(f"Accuracy (TF-IDF): {accuracy_score(y_test, y_pred_tfidf):.4f}")
print("\nReporte de Clasificación (TF-IDF):")
print(classification_report(y_test, y_pred_tfidf, target_names=['Bajo Interés', 'Alto Interés']))


#   MODELADO CON DEEP LEARNING 

print("\n\n Iniciando Proceso de Deep Learning ")
# Defino los hiperparámetros para mi red neuronal
VOCAB_SIZE = 10000  # Tamaño del vocabulario.
MAX_LENGTH = 30     # Longitud máxima de las secuencias de texto.
EMBEDDING_DIM = 64  # Dimensión de los vectores de palabras (embeddings).
EPOCHS = 10         # Número de veces que el modelo verá todos los datos de entrenamiento.
BATCH_SIZE = 32     # Número de muestras de datos por cada actualización del modelo.

# Preparo el texto para la red neuronal.
# Inicializo el Tokenizer de Keras.
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
# Creo el vocabulario basado en los textos de entrenamiento.
tokenizer.fit_on_texts(X_train)
# Convierto los textos de entrenamiento a secuencias de números.
train_sequences = tokenizer.texts_to_sequences(X_train)
# Hago que todas las secuencias tengan la misma longitud (MAX_LENGTH) añadiendo ceros al final (padding).
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
# Repito el proceso para los datos de prueba.
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# Construyo la arquitectura de la Red Neuronal capa por capa.
model_dl = Sequential([
    # Capa de Embedding: Convierte los números de las palabras en vectores densos de EMBEDDING_DIM.
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    # Capa de Pooling: Promedia los vectores de palabras para obtener un solo vector por texto.
    GlobalAveragePooling1D(),
    # Capa Densa: Una capa neuronal estándar con 24 neuronas y activación ReLU.
    Dense(24, activation='relu'),
    # Capa de Dropout: Desactiva aleatoriamente el 30% de las neuronas para prevenir el sobreajuste.
    Dropout(0.3),
    # Capa de Salida: Una sola neurona con activación sigmoide para dar una probabilidad entre 0 y 1.
    Dense(1, activation='sigmoid')
])
# Compilo el modelo, definiendo el optimizador, la función de pérdida y la métrica a monitorear.
model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Imprimo un resumen de la arquitectura del modelo.
model_dl.summary()

# Configuro el Checkpoint para guardar solo los pesos del mejor modelo encontrado durante el entrenamiento.
checkpoint_path = 'best_dl_model.weights.h5'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

print("\nEntrenando la red neuronal...")
# Entreno el modelo con los datos de entrenamiento, usando los datos de prueba para validación en cada época.
history = model_dl.fit(train_padded, y_train,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_data=(test_padded, y_test),
                       callbacks=[checkpoint_callback],
                       verbose=2)

# Evalúo el rendimiento del mejor modelo guardado.
print("\nCargando y evaluando el mejor modelo de Deep Learning...")
# Cargo los pesos del mejor modelo que se guardó.
model_dl.load_weights(checkpoint_path)
# Evalúo este modelo con los datos de prueba para obtener la pérdida y precisión finales.
loss, accuracy = model_dl.evaluate(test_padded, y_test)
print(f"\n--- Resultados del Modelo de Deep Learning (Mejor Checkpoint) ---")
print(f'Accuracy en prueba: {accuracy:.4f}')
print(f'Pérdida en prueba: {loss:.4f}')

# Hago predicciones con el mejor modelo para generar el reporte de clasificación.
y_pred_dl = (model_dl.predict(test_padded) > 0.5).astype("int32")
print("\nReporte de Clasificación (Deep Learning):")
print(classification_report(y_test, y_pred_dl, target_names=['Bajo Interés', 'Alto Interés']))

print("\n FIN ")


