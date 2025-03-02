import numpy as np
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # type: ignore
from PIL import Image
import logging
import io

# 🔹 Initialisation des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 Chargement du modèle ResNet50 pré-entraîné sur ImageNet
logger.info("Chargement du modèle ResNet50...")
model = ResNet50(weights="imagenet")
logger.info("Modèle chargé avec succès.")

def preprocess_image(image_bytes):
    """
    Convertit une image reçue en entrée en un format compatible avec ResNet50.
    
    Args:
        image_bytes (bytes): Image sous forme de bytes.

    Returns:
        np.ndarray: Image prétraitée prête pour la prédiction.
    """
    try:
        # 🔹 Ouverture de l'image avec PIL
        img = Image.open(io.BytesIO(image_bytes))

        # 🔹 Vérification du format d'image
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 🔹 Redimensionnement à la taille requise (224x224)
        img = img.resize((224, 224))

        # 🔹 Conversion en tableau NumPy
        img_array = image.img_to_array(img)

        # 🔹 Ajout de la dimension batch (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 🔹 Prétraitement selon les exigences de ResNet50
        img_array = preprocess_input(img_array)

        return img_array
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement de l'image : {e}")
        return None

def predict_image(image_bytes):
    """
    Effectue la prédiction sur une image.

    Args:
        image_bytes (bytes): Image en bytes.

    Returns:
        list: Liste des 3 meilleures prédictions avec leur probabilité.
    """
    # 🔹 Prétraitement de l'image
    img_array = preprocess_image(image_bytes)
    if img_array is None:
        return None

    # 🔹 Prédiction avec le modèle
    preds = model.predict(img_array)

    # 🔹 Décodage des résultats (top 10 classes les plus probables)
    result = decode_predictions(preds, top=10)[0]

    # 🔹 Formatage des prédictions
    predictions = [
        {"label": label, "probability": float(prob)}
        for (_, label, prob) in result
    ]

    return predictions
