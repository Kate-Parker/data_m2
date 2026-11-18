# Data M2 — préparation et segmentation

Ce dépôt contient le travail sur FoodData Central (préparation d'un catalogue produit) et une analyse séparée de segmentation des clients du jeu public Instacart Market Basket Analysis.

## Installation (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Données locales

Les données ne sont pas versionnées dans Git :

- Placer les CSV FoodData Central dans `data/raw/`.
- Conserver les CSV Instacart séparément. Le script de segmentation attend `orders.csv`, `order_products__prior.csv`, `products.csv` et `departments.csv`.

Les dossiers `data/raw/`, `data/processed/` et les exports de modèles sont ignorés par Git afin de ne pas pousser de gros fichiers de données.

## Commandes

Préparer le jeu FoodData Central :

```powershell
python .\preparer_donnees.py
```

Entraîner le classificateur exploratoire de catégorie produit :

```powershell
python .\entrainer_classification.py
```

Segmenter les clients Instacart (adapter le chemin d'entrée) :

```powershell
python .\segmenter_clients_instacart.py --input-dir "C:\chemin\vers\archive (1)"
```

Les notebooks EDA et segmentation se trouvent dans `notebooks/`.

## Périmètre et interprétation

Le classificateur FoodData Central prédit `branded_food_category`; il ne remplace pas le modèle métier de réachat attendu sur les commandes Instacart. La segmentation Instacart utilise uniquement les commandes `prior`, sans les commandes d'évaluation `train` et `test`.

Sur l'exécution de référence, MiniBatchKMeans a retenu `k=2` (silhouette `0,3326` sur un échantillon de 2 500 clients). Les clusters sont descriptifs du jeu public Instacart et ne représentent pas des clients réels de Panier Vert. La source et sa licence doivent être citées dans le livrable d'équipe.
