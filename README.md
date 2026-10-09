# Reconnaissance d'expressions faciales avec un CNN

Projet de **Fondamentaux du Deep Learning** (2026-2027), CY Tech.
**Équipe :** Nathan Le Nestour, YU Xinyu

## Objectif

Reconnaître l'expression d'un visage à partir d'une image en niveaux de gris de 48×48 pixels. Il y a 7 classes : colère, dégoût, peur, joie, neutre, tristesse et surprise.

On part d'un modèle de référence (MLP), on construit un réseau de neurones convolutif (CNN), puis on l'améliore étape par étape.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `Projet_DL_Expressions_faciales_v2.ipynb` | Notebook complet : données, modèles, entraînement, évaluation et expériences. Chaque partie contient des explications et une analyse des résultats. |
| `README.md` | Ce fichier |

## Dataset : FER-2013

- **Source :** [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013), téléchargé automatiquement par le notebook avec `kagglehub`
- **Auteurs :** Pierre-Luc Carrier et Aaron Courville (concours Kaggle de 2013)
- **Licence :** Open Database License (ODbL)
- **Taille :** 35 887 images, dont 28 709 dans `train/` et 7 178 dans `test/`
- **Déséquilibre :** la Joie représente 25 % des images, le Dégoût seulement 1,5 %

**Découpage utilisé :**
- le dossier `train/` est divisé en 85 % pour l'entraînement (24 402 images) et 15 % pour la validation (4 307 images), de façon stratifiée ;
- le dossier `test/` (7 178 images) n'est utilisé **qu'une seule fois**, à la fin, pour évaluer le modèle final.

## Lancer le notebook sur Google Colab

1. Ouvrir le notebook dans Google Colab.
2. Activer le GPU : *Exécution → Modifier le type d'exécution → GPU*.
3. Exécuter les cellules dans l'ordre (*Exécution → Tout exécuter*).
4. Autoriser l'accès à Google Drive quand Colab le demande.

**Sauvegarde des modèles :** chaque modèle entraîné est enregistré dans Google Drive, dans le dossier `MyDrive/projet_dl_expressions/` :
- `models/` contient les fichiers `.keras` ;
- `histories/` contient les historiques d'entraînement au format `.json`.

Si un modèle existe déjà dans ce dossier, le notebook le **recharge au lieu de le réentraîner**. La première exécution est donc lente, car les 7 modèles sont entraînés. Les exécutions suivantes sont rapides. Pour réentraîner un modèle, il suffit de supprimer son fichier dans Drive.

**Bibliothèques :** TensorFlow / Keras, NumPy, pandas, scikit-learn, Matplotlib, seaborn, Pillow et kagglehub. Elles sont toutes déjà installées sur Colab, sauf `kagglehub`, que la première cellule installe.

## Structure du notebook

1. **Données :** présentation du dataset, prétraitement (niveaux de gris, 48×48, pixels divisés par 255), exemples d'images, encodage one-hot et découpage.
2. **Modèle de référence (MLP) :** Flatten, Dense 256, Dense 128, Softmax. La propagation avant est aussi refaite avec NumPy pour vérifier le calcul.
3. **CNN :** notions (convolution, padding, stride, pooling), exemple de convolution faite à la main, architecture et nombre de paramètres couche par couche.
4. **Entraînement :** Adam (learning rate 0,001), batch de 64, perte `categorical_crossentropy`, Early Stopping et réduction du learning rate.
5. **Évaluation :** précision, rappel, F1, matrice de confusion et exemples d'erreurs.
6. **Expériences :** six variantes du CNN, choix du modèle final et évaluation sur le jeu de test.

## Architecture du CNN

```
Entrée 48×48×1
Bloc 1 : Conv 3×3 (32)  → Conv 3×3 (32)  → MaxPool 2×2   → 24×24×32
Bloc 2 : Conv 3×3 (64)  → Conv 3×3 (64)  → MaxPool 2×2   → 12×12×64
Bloc 3 : Conv 3×3 (128) → Conv 3×3 (128) → MaxPool 2×2   → 6×6×128
Flatten (4608) → Dense 256 → Dense 7 + Softmax
```

## Expériences

À chaque expérience, on change **une seule chose** par rapport à la précédente.

| Modèle | Modification | Accuracy val | Macro-F1 val |
|---|---|---|---|
| MLP | modèle de référence | 0,410 | 0,324 |
| E0 | CNN de base, sans régularisation | 0,547 | 0,514 |
| E1 | + Dropout | 0,606 | 0,577 |
| **E2** | **+ Batch Normalization** | **0,647** | **0,623** |
| E3 | + Data Augmentation | 0,632 | 0,568 |
| E4 | + 2 fois plus de filtres | 0,661 | 0,612 |
| E5 | + pondération des classes | 0,642 | 0,597 |

**Choix du modèle final :** E2, parce qu'il a le **meilleur macro-F1 sur la validation**. Le dataset est déséquilibré : avec l'accuracy, la classe Joie compte beaucoup plus que les autres. Le macro-F1 donne le même poids aux 7 expressions.

## Résultats sur le jeu de test

| Modèle | Accuracy test | Macro-F1 test |
|---|---|---|
| MLP | 0,407 | 0,326 |
| CNN de base (E0) | 0,555 | 0,525 |
| **Modèle final (E2)** | **0,655** | **0,643** |

- **Classes les mieux reconnues :** Joie (F1 = 0,841) et Surprise (F1 = 0,794)
- **Classes les plus difficiles :** Peur (F1 = 0,495) et Tristesse (F1 = 0,522)
- **Confusions principales :** Peur et Tristesse, Tristesse et Neutre, Dégoût et Colère

Pour comparaison, la précision humaine sur FER-2013 est d'environ 65 %.

## Conclusion et pistes d'amélioration

Le CNN est beaucoup plus performant que le MLP, parce qu'il exploite la structure spatiale de l'image. Le CNN de base faisait beaucoup de surapprentissage (95 % en entraînement contre 55 % en validation). Le Dropout et la Batch Normalization ont nettement amélioré la généralisation.

**Pistes d'amélioration :**
- transfer learning avec un modèle pré-entraîné (VGG, ResNet) ;
- réseau plus profond ou ensemble de plusieurs modèles ;
- Data Augmentation mieux réglée ;
- appliquer la pondération des classes à E2, le modèle qui a le meilleur macro-F1.
