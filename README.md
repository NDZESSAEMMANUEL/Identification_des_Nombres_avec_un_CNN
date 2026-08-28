# Reconnaissance de chiffres manuscrits — MNIST (CNN avec Keras)

Ce projet entraîne un réseau de neurones convolutif (CNN) simple pour reconnaître des chiffres manuscrits (0 à 9) à partir du jeu de données **MNIST**, en utilisant **TensorFlow / Keras**.

## Aperçu

Le modèle est composé de deux couches de convolution suivies d'une couche dense de sortie (softmax). Après entraînement, le script affiche un échantillon d'images de test avec le label attendu, le label prédit et le taux de confiance de la prédiction.

Exemple de résultat :

```
Attendu 7 - Prédit 7 (100%)
Attendu 5 - Prédit 5 (91%)
...
```

## Prérequis

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

Installation des dépendances :

```bash
pip install tensorflow numpy matplotlib
```

## Structure du modèle

| Couche          | Détails                              |
|-----------------|---------------------------------------|
| Input           | (28, 28, 1)                          |
| Conv2D          | 32 filtres, noyau 3x3, activation ReLU |
| Conv2D          | 16 filtres, noyau 3x3, activation ReLU |
| Flatten         | —                                     |
| Dense (sortie)  | 10 neurones, activation Softmax       |

- **Fonction de perte** : `categorical_crossentropy`
- **Optimiseur** : `adam`
- **Métrique** : `accuracy`
- **Épochs** : 5
- **Batch size** : 32

## Données

Le jeu de données MNIST est chargé directement via `tensorflow.keras.datasets.mnist` :
- 60 000 images d'entraînement
- 10 000 images de test
- Images en niveaux de gris de 28x28 pixels

Les pixels sont normalisés (division par 255) et les labels sont encodés en *one-hot* via `to_categorical`.

## Utilisation

Exécuter le script principal :

```bash
python mnist_cnn.py
```

Le script va :
1. Charger et préparer les données MNIST.
2. Construire et compiler le modèle CNN.
3. Entraîner le modèle sur 5 épochs.
4. Évaluer la précision sur le jeu de test.
5. Afficher une grille de 10 images de test avec les prédictions et le taux de confiance.

## Résultats attendus

Avec cette architecture simple, on obtient généralement une précision de test autour de **98 %** après 5 épochs.

## Améliorations possibles

- Ajouter des couches `MaxPooling2D` pour réduire la dimensionnalité.
- Ajouter du `Dropout` pour limiter le surapprentissage.
- Augmenter le nombre d'épochs ou utiliser un `EarlyStopping`.
- Sauvegarder le modèle entraîné avec `modele.save()`.
- Tester le modèle sur des images dessinées à la main (hors MNIST).

## Licence

Projet libre d'utilisation à des fins pédagogiques.