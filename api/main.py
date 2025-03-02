from fastapi import FastAPI, UploadFile, File, HTTPException
import logging
from models.model import predict_image

# 🔹 Initialisation des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 Initialisation de l'application FastAPI
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint permettant de recevoir une image et de retourner la prédiction du modèle.

    Args:
        file (UploadFile): Une image envoyée par l'utilisateur.

    Returns:
        dict: Les trois prédictions les plus probables avec leur score de confiance.
    """
    try:
        # 🔹 Lecture du contenu de l'image
        image_bytes = await file.read()

        # 🔹 Prédiction de l'image
        predictions = predict_image(image_bytes)
        if predictions is None:
            raise HTTPException(status_code=400, detail="Erreur lors du traitement de l'image.")

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")
