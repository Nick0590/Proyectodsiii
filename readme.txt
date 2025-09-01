Proyecto Final: Clasificador de Interés de Búsquedas con NLP y Deep Learning
	1	Descripción del Proyecto
El objetivo de este proyecto es desarrollar y evaluar un modelo capaz de analizar el texto de una consulta de búsqueda en Google y predecir si generará un "Alto Interés" (alta probabilidad de clic) o "Bajo Interés" para una página web.
Para lograr esto, se utiliza un dataset que contiene búsquedas (query) que se realizaron para una web de noticias en especifico, sus impresiones y los clics que generaron. La métrica clave para definir el "interés" es el CTR (Click-Through Rate), que se calcula como clics / impresiones.
	2	Objetivos del Proyecto
Objetivo General
	•	Evaluar y comparar la efectividad de un modelo de Machine Learning clásico (Regresión Logística con TF-IDF) frente a un modelo de Deep Learning (Red Neuronal con Embeddings) para clasificar el interés de las consultas de búsqueda del dataset proporcionado.
Objetivos Específicos
	1	Implementar un pipeline de preprocesamiento NLP: Preparar los datos de texto crudo aplicando limpieza, eliminación de stopwords (NLTK) y lematización (spaCy) para que puedan ser interpretados por los modelos.
	2	Desarrollar un modelo baseline: Entrenar y evaluar un modelo de Machine Learning simple pero robusto que sirva como punto de referencia para medir el rendimiento del modelo más complejo.
	3	Construir un modelo de Deep Learning: Entrenar y evaluar una red neuronal sencilla, guardando su mejor versión mediante checkpoints, para compararla de forma justa contra el modelo baseline.
	4	Dataset
El dataset utilizado es bq-results.csv y contiene las siguientes columnas relevantes:
	•	query: El texto de la búsqueda realizada por el usuario.
	•	clicks: El número de veces que se hizo clic en el resultado.
	•	impressions: El número de veces que el resultado se mostró.
Se realizó una etapa de ingeniería de características para crear la variable objetivo clase_interes. Se calculó el CTR para cada query y se utilizó la mediana de todos los CTRs como umbral. Las búsquedas con un CTR por encima de la mediana se clasificaron como "Alto Interés" (1) y las demás como "Bajo Interés" (0).
	4	Metodología
El proyecto sigue un flujo de trabajo estándar de ciencia de datos, comparando modelos clásicos contra una aproximación de Deep Learning.
4.1. Preprocesamiento NLP
Para preparar el texto de las búsquedas para los modelos, se aplicó un pipeline de preprocesamiento robusto utilizando las librerías NLTK y spaCy:
	1	Limpieza y Normalización: Se convirtió todo el texto a minúsculas y se eliminaron caracteres especiales y números.
	2	Eliminación de Stopwords: Se utilizó la lista de palabras vacías en español de NLTK para eliminar términos comunes que no aportan significado (ej: 'el', 'la', 'de').
	3	Lematización: Se utilizó el modelo es_core_news_sm de spaCy para reducir cada palabra a su forma raíz o lema (ej: "corriendo" -> "correr"). Esto ayuda a agrupar palabras con el mismo significado.
4.2. Modelos Baseline con Scikit-Learn
Se entrenaron dos modelos de Regresión Logística como punto de referencia para evaluar el rendimiento. Se utilizaron dos técnicas de vectorización diferentes:
	1	Bag of Words (BoW): Representa el texto contando la frecuencia de cada palabra.
	2	TF-IDF: Representa el texto asignando un peso a cada palabra según su importancia en una query específica en relación con todo el dataset.
4.3. Modelo de Deep Learning con TensorFlow/Keras
Se construyó una red neuronal sencilla con la siguiente arquitectura:
	•	Capa de Embedding: Convierte las palabras en vectores numéricos densos, permitiendo que el modelo capture relaciones semánticas.
	•	Capa GlobalAveragePooling1D: Reduce la dimensionalidad promediando los vectores de palabras de cada query.
	•	Capas Densas: Capas neuronales conectadas que aprenden los patrones para la clasificación final.
	•	Checkpoint: Se utilizó un ModelCheckpoint para guardar automáticamente la mejor versión del modelo durante el entrenamiento, basándose en la precisión de validación.
	5	Resultados y Conclusión
Tras entrenar y evaluar todos los modelos con los datos de prueba, se obtuvieron los siguientes resultados de precisión (Accuracy):
Modelo
Accuracy
Regresión Logística (Bag of Words)
76.43%
Regresión Logística (TF-IDF)
76.43%
Red Neuronal (Deep Learning)
70.71%
Conclusión: Para este dataset, el modelo clásico de Regresión Logística con TF-IDF superó al modelo de Deep Learning. A pesar de su simplicidad, demostró ser más efectivo y, notablemente, superior en identificar correctamente las búsquedas de "Alto Interés" (con un recall de 0.87 frente al 0.57 del modelo DL).
Este resultado subraya la importancia de establecer siempre un baseline sólido, ya que una mayor complejidad del modelo no siempre garantiza un mejor rendimiento.

Clonar el Repositorio

git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA_DEL_PROYECTO>

Crear y Activar un Entorno Virtual
Esto crea un entorno aislado para instalar las dependencias y evitar conflictos.

# Crear el entorno
python3 -m venv venv

# Activar 
source venv/bin/activate

# Activar en Windows
# venv\Scripts\activate

Instalar las Dependencias
El archivo requirements.txt contiene todas las librerías necesarias.

pip install -r requirements.txt

Ejecutar el Script Principal
Asegúrate de que el archivo bq-results.csv y tu script .py estén en la misma carpeta.

python3 pro.py

NOTA: Versionar de tensorflow exclusiva para Mac y linea posterior a la carga de librerias exclusiva para Mac.
