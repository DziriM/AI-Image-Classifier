# 🖼️ AI Image Classifier

🚀 **Projet de classification d’images avec IA - TensorFlow, FastAPI, Streamlit**

Ce projet permet de classer des images en utilisant un **modèle de Deep Learning pré-entraîné (ResNet50)** via une **API REST avec FastAPI** et une **interface utilisateur avec Streamlit**.

---

## 📌 Fonctionnalités

✅ **Classification d’images avec un modèle IA pré-entraîné (ResNet50)**  
✅ **API REST en FastAPI pour envoyer une image et obtenir une prédiction**  
✅ **Interface Web en Streamlit pour tester l’IA avec un simple upload**  
✅ **Optimisation du prétraitement des images (redimensionnement, normalisation)**  
✅ **Exécution locale avec Python et Docker (facultatif)**

---

## 🔧 Installation & Lancement

### **1️⃣ Cloner le projet**

```powershell
git clone https://github.com/DziriM/AI-Image-Classifier.git
cd AI-Image-Classifier
```

### **2️⃣ Installer les dépendances**

- Assure-toi d’avoir Python installé (vérifie avec python --version). Puis exécute :

```powershell
pip install -r requirements.txt
```

### **3️⃣ Lancer l’API FastAPI**

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

➡ Accéder à Swagger pour tester l’API : http://127.0.0.1:8000/docs

### **4️⃣ Lancer l’interface Streamlit**

Dans un autre terminal PowerShell, exécute :

```powershell
streamlit run streamlit_app/app.py
```

➡ Ouvrir l’interface Web : http://localhost:8501

🎯 Comment fonctionne l’API ?
L’API utilise FastAPI pour gérer les requêtes.

🔹 Endpoint principal : /predict/
📤 Input : Une image envoyée en multipart/form-data
📥 Output : Une prédiction IA avec 3 résultats possibles
📌 Exemple d’appel API avec curl :

```powershell
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@chemin/vers/mon_image.jpg'
```

📌 Réponse JSON attendue :

```json
{
  "predictions": [
    { "label": "golden retriever", "probability": 98.76 },
    { "label": "labrador", "probability": 85.32 },
    { "label": "beagle", "probability": 67.45 }
  ]
}
```

## 📦 Optionnel : Exécuter avec Docker

### **📌 1️⃣ Construire l’image Docker**

```powershell
docker build -t ai-image-classifier .
```

### **📌 2️⃣ Lancer le conteneur**

```powershell
docker run -p 8000:8000 ai-image-classifier
```

🔮 Améliorations futures
🛠 Roadmap des prochaines améliorations :
🔹 Ajouter un mode d’entraînement personnalisé avec TensorFlow
🔹 Déployer l’API sur un service Cloud (Azure, Render, Railway)
🔹 Supporter d’autres modèles IA pour la classification (VGG16, MobileNet, etc.)

## 🤝 Contribuer au projet

### **Les contributions sont les bienvenues ! 🚀**

- 📌 Fork le repo
- 📌 Crée une branche (git checkout -b feature-xxx)
- 📌 Fais tes modifications et commit (git commit -m "Ajout de xxx")
- 📌 Push ta branche et ouvre une Pull Request !

## 📚 Pour aller plus loin

Si vous souhaitez en apprendre davantage sur la base de données ImageNet utilisée pour entraîner ResNet50, voici la publication originale de 2009 :

📄 **ImageNet: A Large-Scale Hierarchical Image Database**  
🔗 [Lire la publication](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)
