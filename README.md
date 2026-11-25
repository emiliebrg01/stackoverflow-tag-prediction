# Prédiction de tags Stack Overflow — TF-IDF vs BERT

Projet du cours **From PoC to Prod** (EPF 5A).

À partir du **titre** d'une question Stack Overflow, on prédit automatiquement ses **tags** (`python`, `c#`, `sql`…). Une question peut avoir plusieurs tags : c'est de la **classification multi-label**.

Deux approches sont comparées sur **le même découpage des données** :

1. **Data Science classique** : TF-IDF (mots + n-grammes de caractères) + régression logistique *One-vs-Rest*.
2. **Deep Learning** : fine-tuning de **BERT** (`bert-base-uncased`).

## Données

`data/stackoverflow_posts.csv` contient une ligne par couple (question, tag).

| | |
|---|---|
| Questions | 4 609 |
| Tags différents | 681 |
| Tags par question | 2,2 en moyenne (1 à 5) |
| Longueur des titres | ~8 mots |

**Préparation** (`src/preprocessing.py`) :

- regroupement par question ;
- conservation des tags vus **au moins 20 fois**, soit 86 tags couvrant 89 % des questions ;
- découpage **stratifié multi-label** en 70 / 15 / 15 (train / validation / test), avec `seed=42`.

## Résultats (jeu de test, 611 questions)

| Métrique | TF-IDF + LogReg | BERT fine-tuné |
|---|---|---|
| **F1 micro** | **0.561** | _à compléter_ |
| F1 macro | 0.512 | _à compléter_ |
| Précision (micro) | 0.616 | _à compléter_ |
| Rappel (micro) | 0.516 | _à compléter_ |
| Précision@1 | 0.655 | _à compléter_ |
| Temps d'entraînement | ~8 s (CPU) | _à compléter_ (GPU T4) |
| Temps de prédiction | ~0.2 ms / question | _à compléter_ |
| Taille du modèle | 13.5 Mo | ~440 Mo |

Les résultats détaillés sont dans [results/](results/) et l'analyse dans les notebooks.

### Optimisations mises en place

- **Seuil de décision** optimisé sur la validation, au lieu de 0,5. Au moins un tag est toujours prédit.
- **TF-IDF** :
  - combinaison mots (1–2 grammes) et caractères (2–5 grammes) ;
  - nettoyage qui préserve `c#`, `c++`, `.net` ;
  - `class_weight="balanced"` ;
  - choix de `C` sur la validation.
  - Variantes testées et écartées : SVM linéaire, régression logistique sans rééquilibrage.
- **BERT** :
  - deux learning rates (3e-5 pour BERT, 1e-3 pour la couche de sortie) ;
  - warmup puis décroissance linéaire ;
  - précision mixte fp16 ;
  - gradient clipping ;
  - early stopping sur le F1 de validation ;
  - `max_length=48`, adapté aux titres courts.

## Structure du projet

```
├── data/stackoverflow_posts.csv
├── notebooks/
│   ├── 01_exploration_tfidf.ipynb   # exploration + modèle TF-IDF (exécutable en local)
│   └── 02_bert_finetuning.ipynb     # fine-tuning BERT + comparaison (Google Colab, GPU)
├── src/
│   ├── preprocessing.py   # chargement, filtrage des tags, découpage stratifié
│   ├── metrics.py         # métriques et optimisation du seuil (communs aux 2 modèles)
│   ├── tfidf_model.py     # entraînement du modèle TF-IDF
│   ├── bert_model.py      # entraînement et prédiction BERT
│   └── predict.py         # prédiction en ligne de commande (TF-IDF)
├── tests/test_pipeline.py # tests automatiques (pytest)
├── results/               # métriques au format JSON / CSV, graphique de comparaison
└── requirements.txt
```

## Utilisation

```bash
pip install -r requirements.txt

python -m src.tfidf_model                                 # entraîne le modèle TF-IDF
python -m src.predict "How to center a div with CSS?"     # -> [('css', 0.993), ('html', 0.909), ...]
python -m pytest                                          # lance les tests
```

**BERT** : ouvrir `notebooks/02_bert_finetuning.ipynb` dans Google Colab, activer le GPU (*Exécution → Modifier le type d'exécution → T4 GPU*), renseigner l'URL du repo dans la 1ʳᵉ cellule, puis *Tout exécuter*.
