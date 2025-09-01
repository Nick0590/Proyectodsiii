import nltk
import ssl

# Este truco crea un contexto de seguridad temporal no verificado.
# Permite a NLTK saltarse la verificación de certificados que está fallando en tu Mac.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Intentando descargar el paquete 'stopwords' de NLTK...")

# Ahora descargamos el recurso que falta.
nltk.download('stopwords')

print("\n¡Descarga completada! Ahora puedes ejecutar tu script principal 'pro.py'.")
