# 🚨 Application de Détection d'Accidents - IA YOLO

Cette application Streamlit permet de détecter des accidents à partir d'images ou de vidéos en utilisant un modèle YOLO entraîné, elle permet aussi de faire de la detection d'objet en temps réel via la webcam.

## 🎯 Fonctionnalités

- **Upload de fichiers** : Support des images (JPG, PNG, BMP) et vidéos (MP4, AVI, MOV, MKV)
- **Détection d'accidents** : Utilisation du modèle YOLO `detection_accident.pt`
- **Augmentation d'images** : Filtres et améliorations avant la détection
- **Interface intuitive** : Contrôles paramétrables et affichage des résultats
- **Temps réel** : Mesure des performances (FPS)
- **Vidéos annotées** : Génération de vidéos avec boîtes de détection
- **Mode rapide** : Optimisations de performance pour analyse accélérée
- **Optimisation de résolution** : Redimensionnement automatique des vidéos pour 4x plus de vitesse
- **Mode de détection d'objet en temps réel** : Utilisation du modèle YOLOv12


## 🚀 Installation

1. **Cloner le projet** :
```bash
git clone <votre-repo>
cd FINAL
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Lancer l'application** :
```bash
streamlit run app.py
```

## 📁 Structure du Projet

```
FINAL/
├── app.py                    # Application Streamlit principale
├── requirements.txt          # Dépendances Python
├── README.md                # Ce fichier
└── models/
    └── detection_accident.pt # Modèle YOLOV12 déjà entraîné
    └── yolo12s.pt
```

## 🎛️ Utilisation

### 1. Upload de Fichiers
- Uploadez une image ou une vidéo dans la zone dédiée
- L'application détecte automatiquement le type de fichier



### 3. Paramètres du Modèle
- **Seuil de confiance** : Ajustez la sensibilité de détection (0.1 à 1.0)
- **Nombre max de détections** : Limitez le nombre d'objets détectés
- **Affichage** : Choisissez ce qui doit être affiché (étiquettes, scores, FPS)

### 4. Augmentation d'Images (Images uniquement)
- **Luminosité** : Ajustez la luminosité (0.5x à 2.0x)
- **Contraste** : Modifiez le contraste (0.5x à 2.0x)
- **Saturation** : Ajustez la saturation (0.0x à 2.0x)
- **Netteté** : Améliorez la netteté (0.0x à 2.0x)
- **Flou** : Ajoutez du flou (0 à 10)
- **Bruit** : Ajoutez du bruit (0 à 50)

### 5. Détection
- Cliquez sur "🚀 Lancer la Détection"
- L'application applique les augmentations (si configurées)
- Le modèle YOLO analyse le fichier
- Les résultats s'affichent avec les boîtes de détection

## 🔧 Configuration Avancée

### Modèle YOLO
- Placez votre modèle dans le dossier `model/`
- Modifiez le chemin dans `app.py` si nécessaire
- Support des formats `.pt`, `.onnx`, `.engine`

### Personnalisation CSS
- Modifiez les styles dans la section CSS de `app.py`
- Ajustez les couleurs, espacements et animations

## 📊 Formats Supportés

### Images
- JPG, JPEG
- PNG
- BMP

### Vidéos
- MP4
- AVI
- MOV
- MKV

## 🚨 Résolution de Problèmes

### Erreur de chargement du modèle
- Vérifiez que le fichier `detection_accident.pt` existe
- Assurez-vous que PyTorch est installé correctement

### Problèmes de mémoire
- Réduisez la taille des images/vidéos
- Ajustez le nombre max de détections

### Performance lente
- Utilisez un GPU si disponible
- Ajustez le seuil de confiance

## 🤝 Contribution

Pour contribuer à ce projet :
1. Fork le repository
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Créez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 📞 Support

Pour toute question ou problème :
- Ouvrez une issue sur GitHub
- Contactez l'équipe de développement

---

**🚨 Application de Détection d'Accidents **  
*Développé avec Streamlit et Ultralytics*
