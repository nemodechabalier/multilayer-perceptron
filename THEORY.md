# Théorie mathématique du perceptron multicouche (MLP)

Notes de révision du projet 42 Multilayer Perceptron : tout ce qu'il faut savoir expliquer en défense.

## 1. Notations

- **m** : taille d'un batch ; **n** : nombre total d'exemples
- **X** : entrées, forme (m, n_features) ; **A₀ = X**
- **Y** : labels one-hot, forme (m, 2), avec M = [1, 0] et B = [0, 1]
- **Wₗ, bₗ** : poids de forme (n_in, n_out) et biais de forme (n_out,) de la couche l
- **Zₗ** : somme pondérée (pré-activation) de la couche l
- **Aₗ** : activation (sortie) de la couche l
- **P** : sortie du réseau (probabilités), forme (m, 2)
- **α** : learning rate

## 2. Le neurone

Pour un exemple x et un neurone :

```
z = Σₖ xₖ·wₖ + b        (somme pondérée)
a = f(z)                 (activation)
```

C'est exactement la régression logistique de DSLR (un neurone + sigmoïde). Un MLP empile des neurones : les 24 neurones d'une couche sont 24 vecteurs w placés côte à côte, ce qui forme une matrice.

## 3. Architecture et dimensions

Pour une couche : **Zₗ = Aₗ₋₁ · Wₗ + bₗ**, puis **Aₗ = f(Zₗ)**.

Règle du produit matriciel : (a, b) @ (b, c) → (a, c). La dimension du milieu doit être identique et disparaît. Le biais (n_out,) est ajouté à chaque ligne par broadcasting.

Exemple avec sizes = [30, 24, 18, 2] et un batch de 8 :

| Couche | Calcul | Forme de sortie |
|---|---|---|
| Entrée | X | (8, 30) |
| 1 | (8,30) @ (30,24) | (8, 24) |
| 2 | (8,24) @ (24,18) | (8, 18) |
| Sortie | (8,18) @ (18,2) | (8, 2) |

Seule la dimension des neurones change (30 → 24 → 18 → 2). La dimension du batch reste identique.

**Nombre de paramètres** : Σ (n_in · n_out + n_out). Ici : 720 + 432 + 36 = 1188 poids et 24 + 18 + 2 = 44 biais, soit 1232 paramètres.

**Pourquoi une activation non linéaire ?** Sans elle, empiler deux couches revient à une seule : (A·W₁)·W₂ = A·(W₁W₂), soit encore une transformation linéaire. La non-linéarité permet de modéliser des frontières complexes.

## 4. Prétraitement

**Standardisation** : x' = (x − μ) / σ, calculée par feature.

- Les features ont des échelles très différentes (0.01 à 1000). Sans standardisation, le gradient est dominé par les grandes valeurs et l'apprentissage est lent ou instable.
- μ et σ sont calculés **uniquement sur le train**, puis appliqués à la validation et à la prédiction. Les recalculer sur la validation est une fuite de données (data leakage).
- Ils sont sauvegardés avec le modèle.

**One-hot** : M → [1, 0], B → [0, 1]. Il correspond aux deux neurones de sortie du softmax.

## 5. Initialisation

- **Poids aléatoires** : si tous les poids étaient identiques, tous les neurones d'une couche calculeraient la même chose, recevraient le même gradient et resteraient identiques (problème de symétrie).
- **Biais à 0** : sans risque, car ce sont les poids qui brisent la symétrie.
- **He uniform** : poids tirés dans [−limit, +limit] avec limit = √(6 / n_in). Cette formule est conçue pour ReLU. Pour la sigmoïde, Glorot (limit = √(6 / (n_in + n_out))) est le choix classique, mais les deux fonctionnent ici.

## 6. Fonctions d'activation

**Sigmoïde**

```
σ(z) = 1 / (1 + e^(−z))        sortie dans ]0, 1[
σ'(z) = σ(z) · (1 − σ(z))
```

Puisqu'on a déjà a = σ(z) en mémoire, la dérivée vaut simplement **a · (1 − a)**. Pour |z| grand, la sigmoïde sature (σ' ≈ 0) : les gradients deviennent minuscules, c'est le problème du gradient qui s'évanouit (vanishing gradient).

**Softmax** (couche de sortie)

```
pₖ = e^(zₖ) / Σⱼ e^(zⱼ)
```

Les sorties sont positives et leur somme vaut 1 : c'est une distribution de probabilité. **Stabilité numérique** : softmax(z) = softmax(z − max(z)), car le facteur e^(−max) se simplifie. On soustrait le max pour éviter le dépassement de capacité de l'exponentielle.

## 7. Feedforward (passe avant)

```
A₀ = X
pour chaque couche l :
    Zₗ = Aₗ₋₁ · Wₗ + bₗ
    Aₗ = sigmoïde(Zₗ)        (couches cachées)
    Aₗ = softmax(Zₗ)         (dernière couche)
P = A_dernière
```

On **conserve toutes les activations** [A₀, A₁, ..., P] : la backpropagation en a besoin.

Prédiction : classe = argmax de chaque ligne de P (colonne 0 = M, colonne 1 = B).

## 8. Fonctions de coût

**Cross-entropy catégorielle** (entraînement)

```
L = −(1/m) Σᵢ Σₖ yᵢₖ · log(pᵢₖ)
```

Avec un label one-hot, un seul terme survit par exemple : −log(probabilité donnée à la bonne classe). Plus cette probabilité est faible, plus l'erreur est grande. Avec des poids aléatoires, P ≈ [0.5, 0.5] et **L ≈ ln 2 ≈ 0.693** : c'est le loss initial attendu. On borne P dans [ε, 1 − ε] pour éviter log(0).

**Binary cross-entropy** (formule du sujet)

```
E = −(1/N) Σₙ [ yₙ · log(pₙ) + (1 − yₙ) · log(1 − pₙ) ]
```

Avec deux sorties qui somment à 1 (p₁ = 1 − p₀) et un one-hot (y₁ = 1 − y₀), la cross-entropy catégorielle se réécrit exactement comme la binary cross-entropy, avec p = probabilité de M. C'est pourquoi les deux valeurs sont identiques dans le projet.

## 9. Backpropagation

Objectif : calculer ∂L/∂W et ∂L/∂b pour chaque couche. On utilise la règle de la chaîne, en remontant de la sortie vers l'entrée.

**Erreur de sortie (softmax + cross-entropy)**

```
∂L/∂zⱼ = pⱼ − yⱼ        donc   δ_sortie = P − Y
```

Démonstration : pour le softmax, ∂pₖ/∂zⱼ = pₖ(δₖⱼ − pⱼ). Donc ∂L/∂zⱼ = −Σₖ yₖ(δₖⱼ − pⱼ) = −yⱼ + pⱼ·Σₖ yₖ = pⱼ − yⱼ, car Σₖ yₖ = 1. Toutes les dérivées compliquées se simplifient.

**Gradients d'une couche l** (δₗ = erreur sur Zₗ)

```
∂L/∂Wₗ = Aₗ₋₁ᵀ · δₗ / m
∂L/∂bₗ = Σ des lignes de δₗ / m
```

La division par m donne la moyenne sur le batch. ∂L/∂Wₗ a bien la forme de Wₗ : (n_in, m) @ (m, n_out) = (n_in, n_out).

**Propagation vers la couche précédente**

```
δₗ₋₁ = (δₗ · Wₗᵀ) ⊙ Aₗ₋₁ ⊙ (1 − Aₗ₋₁)
```

- δₗ · Wₗᵀ répartit l'erreur sur les neurones de la couche précédente, selon le poids de chaque connexion.
- Aₗ₋₁ ⊙ (1 − Aₗ₋₁) est la dérivée de la sigmoïde (⊙ = produit terme à terme).
- On utilise le Wₗ **avant** sa mise à jour : on calcule tous les gradients, puis on met à jour.

**Vérification (gradient checking)** : pour un poids θ, (L(θ + h) − L(θ − h)) / (2h) avec h ≈ 10⁻⁵ doit être quasi identique au gradient calculé.

## 10. Descente de gradient

```
θ ← θ − α · ∂L/∂θ
```

On déplace chaque paramètre dans la direction qui fait baisser le loss.

| Variante | Exemples par mise à jour |
|---|---|
| Batch | tout le train |
| Stochastique | 1 |
| **Mini-batch** | batch_size (ex. 8) |

- **Epoch** : un passage complet sur le train. Avec 455 exemples et batch_size = 8, une epoch fait environ 57 mises à jour, le dernier batch étant plus petit.
- **Shuffle** à chaque epoch : évite que l'ordre des données biaise l'apprentissage et décorrèle les batchs.
- **Learning rate** : trop grand, le loss oscille ou diverge (nan) ; trop petit, l'apprentissage est très lent.

## 11. Optimiseurs

Notation : g = gradient du paramètre θ, t = nombre de mises à jour effectuées (incrémenté à chaque batch).

**SGD** : θ ← θ − α·g

**Momentum** (moyenne mobile des gradients)

```
v = β·v + g
θ ← θ − α·v            (β ≈ 0.9)
```

Les gradients successifs de même sens s'additionnent, ceux qui oscillent s'annulent : on avance plus vite dans la bonne direction.

**RMSprop** (moyenne mobile des gradients au carré)

```
s = ρ·s + (1 − ρ)·g²
θ ← θ − α · g / (√s + ε)        (ρ ≈ 0.9)
```

Diviser par √s adapte le pas à chaque paramètre : un paramètre aux gradients grands avance par petits pas, et inversement.

**Adam** = momentum + RMSprop + correction de biais

```
m = β₁·m + (1 − β₁)·g
v = β₂·v + (1 − β₂)·g²
m̂ = m / (1 − β₁ᵗ)
v̂ = v / (1 − β₂ᵗ)
θ ← θ − α · m̂ / (√v̂ + ε)
```

Valeurs usuelles : β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, α ≈ 0.001.

**Pourquoi la correction de biais ?** m et v démarrent à 0. Au premier pas, m = (1 − β₁)·g = 0.1·g, bien en dessous de la vraie moyenne g. Diviser par (1 − β₁ᵗ) corrige exactement ce biais ; le facteur tend vers 1 quand t grandit. À t = 1, m̂ = g et v̂ = g², donc le pas vaut environ ±α pour chaque paramètre.

Chaque paramètre (chaque W et chaque b) possède ses propres m et v, de même forme que lui.

## 12. Métriques d'évaluation

**Accuracy** = (TP + TN) / total. Elle ne distingue pas le type d'erreur.

Avec M comme classe positive :

| | Réel M | Réel B |
|---|---|---|
| **Prédit M** | TP | FP |
| **Prédit B** | FN | TN |

```
Précision = TP / (TP + FP)       quand le modèle dit M, a-t-il raison ?
Rappel    = TP / (TP + FN)       quelle part des vraies tumeurs M est détectée ?
F1        = 2·P·R / (P + R)      élevé seulement si P et R le sont
```

Exemple : 100 tumeurs dont 40 M, 30 prédites M dont 27 correctes : TP = 27, FP = 3, FN = 13, TN = 57. Précision = 0.90, rappel = 0.675, accuracy = 0.84.

En contexte médical, un **FN** (tumeur maligne classée bénigne) est l'erreur la plus grave : le rappel est la métrique clé. On protège les divisions par zéro (dénominateur nul → 0).

## 13. Généralisation, surapprentissage, early stopping

- Le jeu de **validation** mesure la performance sur des données jamais vues pendant l'entraînement.
- **Surapprentissage** : le loss d'entraînement continue de baisser mais val_loss remonte, le modèle mémorise le train.
- **Early stopping** : on arrête quand val_loss ne s'améliore plus pendant *patience* epochs consécutives.
- **Restauration du meilleur modèle** : on garde une copie profonde des poids à la meilleure val_loss et on la restaure à la fin. Une simple affectation ne copie rien, car les poids sont modifiés en place.

## 14. Questions probables en défense

| Question | Réponse courte |
|---|---|
| Qu'est-ce que le feedforward ? | Propagation des données de l'entrée vers la sortie : Z = A·W + b puis activation, couche après couche. |
| Qu'est-ce que la backpropagation ? | Calcul des gradients du loss par rapport à chaque poids avec la règle de la chaîne, de la sortie vers l'entrée. |
| Et la descente de gradient ? | Mise à jour θ ← θ − α·gradient pour faire baisser le loss. |
| Pourquoi δ = P − Y en sortie ? | La dérivée de softmax + cross-entropy se simplifie en p − y. |
| D'où vient le terme A(1 − A) ? | C'est la dérivée de la sigmoïde, calculée depuis l'activation stockée. |
| Pourquoi initialiser au hasard ? | Pour briser la symétrie entre neurones. |
| Pourquoi normaliser avec les stats du train ? | Pour éviter la fuite de données et traiter prédiction et validation comme des données nouvelles. |
| Pourquoi le loss initial vaut-il environ 0.69 ? | Sorties proches de [0.5, 0.5] : −log(0.5) = ln 2. |
| Pourquoi BCE = cross-entropy ici ? | Deux sorties sommant à 1 et un one-hot : la formule se réécrit identiquement. |
| Pourquoi Adam converge plus vite ? | Il lisse la direction (momentum) et adapte le pas par paramètre (RMSprop). |
| Pourquoi la correction de biais d'Adam ? | m et v partent de 0 et sont sous-estimés au début. |
| Early stopping contre restauration du meilleur modèle ? | Le premier arrête l'entraînement après *patience* epochs sans progrès ; la seconde remet les poids de la meilleure epoch. |
