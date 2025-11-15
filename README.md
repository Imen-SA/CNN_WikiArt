🖼️ WikiArt Style Classification — CNN Deep Learning Project

Projet de classification des styles artistiques du dataset WikiArt à l'aide d'un CNN (Convolutional Neural Network).

📌 Objectif du projet

Le but est de développer un modèle CNN capable de reconnaître et de classer différentes catégories de styles artistiques (impressionnisme, cubisme, surréalisme, etc.) à partir des images d'œuvres du dataset WikiArt.

Le pipeline inclut :

Le prétraitement des données brutes.

L'application de l'augmentation de données pour la robustesse du modèle.

L'entraînement d'un modèle pour des performances de classification optimales.

🗂️ Structure du projet

WikiArt-CNN/
│
├── README.md           # Documentation principale
├── CNN_wikiArt.ipynb   # Notebook principal contenant le code d'entraînement
├── dataset/            # Dossier contenant les images, organisées par style (classe)
├── models/             # Dossier pour les modèles sauvegardés (.h5)
├── results/            # Dossier pour les courbes de performance, matrices de confusion et exemples
└── requirements.txt    # Liste des dépendances Python


🧠 Modèle & Architecture

Le modèle est un réseau de neurones convolutif séquentiel conçu pour extraire des caractéristiques visuelles complexes spécifiques aux styles artistiques.

Résumé de l'Architecture

Couche

Description

Sortie (Exemple)

Input

Image 128x128x3

(None, 128, 128, 3)

Conv2D

32 filtres, ReLU

(None, 128, 128, 32)

MaxPooling2D

Réduction de taille

(None, 64, 64, 32)

Dropout

Taux de 25% (régularisation)

-

Conv2D

64 filtres, ReLU

(None, 64, 64, 64)

MaxPooling2D

Réduction de taille

(None, 32, 32, 64)

Flatten

Aplatissement des données

(None, 65536)

Dense

128 neurones, ReLU

(None, 128)

Dense

output_classes neurones, Softmax

(None, N_CLASSES)

Configuration de l'Entraînement

Optimizer : Adam (avec un taux d'apprentissage recommandé de lr=0.0001).

Loss Function : Categorical Crossentropy (adaptée pour la classification multi-classes).

🖼️ Dataset WikiArt

Le dataset est composé de plusieurs milliers d’œuvres d'art classées selon leur style.

Prétraitement

Les images sont prétraitées de la manière suivante :

Redimensionnement : (128 × 128 pixels).

Normalisation : Mise à l'échelle des valeurs de pixels (division par 255.0).

Augmentation de Données : Pour améliorer la robustesse et généraliser le modèle.

Transformation

Paramètres

Rescale

1./255

Rotation

30 degrés max

Zoom

0.15 max

Flip

Horizontal

Exemples de Classes

Les exemples ci-dessous montrent la diversité des styles gérés par le modèle.
| Style | Image |
| :--- | :--- |
| Impressionnisme |  |
| Cubisme |  |
| Surréalisme |  |

⚙️ Préparation des données (Code)

Le code suivant utilise ImageDataGenerator pour charger les données, appliquer l'augmentation et créer les générateurs d'entraînement et de validation.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2 # 20% des données pour la validation
)

train_data = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


🚀 Entraînement du modèle (Code)

Le modèle est entraîné sur 30 époques avec un mécanisme de ModelCheckpoint pour sauvegarder automatiquement le meilleur modèle basé sur la précision de la validation (val_accuracy).

from tensorflow.keras.callbacks import ModelCheckpoint

# Sauvegarde uniquement le modèle le plus performant
checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[checkpoint]
)


Exemple de Progression

Epoch 8/30
accuracy: 0.6395 - loss: 0.9641
val_accuracy: 0.6198 - val_loss: 1.0365


📊 Résultats & Performances

Accuracy Entraînement : ~65–70%

Accuracy Validation : ~60%

Les courbes de perte (Loss) et de précision (Accuracy), ainsi que la matrice de confusion, sont disponibles dans le dossier results/.

🧪 Exemple d'inférence (Prédiction)

Comment utiliser le modèle sauvegardé (best_model.h5) pour prédire le style d'une nouvelle œuvre :

from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Charger et prétraiter l'image
img = image.load_img("test.jpg", target_size=(128,128))
# 2. Convertir en tableau numpy, normaliser et ajouter la dimension du batch
img_array = np.expand_dims(np.array(img)/255.0, axis=0) 

# 3. Prédiction
pred = model.predict(img_array)

# 4. Afficher le résultat
# (Assurez-vous que 'class_names' est défini et correspond aux index du générateur)
# Exemple: class_names = list(train_data.class_indices.keys())
print("Predicted style:", class_names[np.argmax(pred)])


📦 Installation & Exécution

Suivez ces étapes pour cloner le projet et lancer le notebook.

1️⃣ Cloner le projet :

git clone [https://github.com/Imen-SA/CNN_WikiArt.git](https://github.com/Imen-SA/CNN_WikiArt.git)
cd CNN_WikiArt


2️⃣ Installer les dépendances :

pip install -r requirements.txt


3️⃣ Lancer le notebook :

jupyter notebook CNN_wikiArt.ipynb


📝 Licence

Ce projet est distribué sous la Licence MIT.

🔗 Liens

GitHub : https://github.com/Imen-SA/CNN_WikiArt