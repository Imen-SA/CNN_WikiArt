# 🖼️ WikiArt Style Classification — CNN Deep Learning Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Keras](https://img.shields.io/badge/Keras-2.14-red)
![License](https://img.shields.io/badge/License-MIT-green)

Ce projet implémente un **réseau de neurones convolutif (CNN)** pour classifier des œuvres d’art du dataset WikiArt en différents **styles artistiques**.

---

## 📚 Table des matières

- ✨ Objectif du projet  
- 📂 Structure du projet  
- 🧠 Modèle & Architecture  
- 🖼️ Dataset WikiArt  
- ⚙️ Préparation des données  
- 🚀 Entraînement du modèle  
- 📊 Résultats & Performances  
- 🧪 Exemple d'inférence  
- 📦 Installation & Exécution  
- 📝 Licence  

---

## ✨ Objectif du projet

Construire un CNN performant capable de reconnaître différents **styles artistiques** (impressionnisme, cubisme, surréalisme…).  
Le modèle apprend automatiquement à partir d’images extraites du dataset WikiArt.

---

## 📂 Structure du projet

WikiArt-CNN/
│── 📝 README.md → documentation complète
│── 📓 CNN_wikiArt.ipynb → notebook principal
│── 📁 dataset/ → images organisées par style
│── 📁 models/ → sauvegarde des modèles (.h5)
│── 📁 results/ → courbes, matrices, prédictions
│── requirements.txt → packages Python requis

yaml
Copier le code

---

## 🧠 Modèle & Architecture

- 2 blocs **Conv2D + MaxPooling2D**  
- **Dropout** pour réduire l’overfitting  
- Flatten → Dense → Softmax  
- Optimizer : **Adam (lr=0.0001)**  
- Loss : **Categorical Crossentropy**  

**Résumé architecture :**  
Conv2D (32 filtres)
MaxPooling
Dropout (0.25)
Conv2D (64 filtres)
MaxPooling
Flatten
Dense(128)
Dense(output_classes, activation='softmax')

yaml
Copier le code

---

## 🖼️ Dataset WikiArt

- Contient des milliers d’œuvres  
- Chaque classe représente un style artistique  
- Prétraitement :  
  - Redimensionnement en 128×128  
  - Normalisation (/255)  
  - Augmentation (rotation, zoom, flip, shift)  

**Exemple de classes :**  
![Exemple Impressionnisme](results/impressionism.jpg)  
![Exemple Cubisme](results/cubism.jpg)  
![Exemple Surréalisme](results/surrealism.jpg)  

---

## ⚙️ Préparation des données

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2
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
🚀 Entraînement du modèle
python
Copier le code
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[checkpoint]
)
Exemple de progression :

makefile
Copier le code
Epoch 8/30
accuracy: 0.6395 - loss: 0.9641
val_accuracy: 0.6198 - val_loss: 1.0365
📊 Résultats & Performances
Accuracy entraînement : ~65–70%

Accuracy validation : ~60%

Matrice de confusion :

Courbes d’accuracy et loss :

GIF animé du modèle en action :

🧪 Exemple d'inférence
python
Copier le code
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("test.jpg", target_size=(128,128))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)

pred = model.predict(img_array)
print("Predicted style:", class_names[np.argmax(pred)])
📦 Installation & Exécution
1️⃣ Cloner le projet

bash
Copier le code
git clone https://github.com/<ton_user>/WikiArt-CNN.git
cd WikiArt-CNN
2️⃣ Installer les dépendances

bash
Copier le code
pip install -r requirements.txt
3️⃣ Lancer le notebook

bash
Copier le code
jupyter notebook CNN_wikiArt.ipynb