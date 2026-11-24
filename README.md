# Instacart — préparation des données

## Arborescence

```text
data_m2/
├── notebook/
│   ├── nettoyage_donnees.ipynb
│   ├── clustering/
│   │   ├── exploration_donnees.ipynb
│   │   └── segmentation_clients.ipynb
│   ├── associations/
│   │   └── regles_association.ipynb
│   └── prediction/
│       └── prediction_reachat.ipynb
├── data/
│   ├── raw/          # six fichiers sources, conservés tels quels
│   └── processed/   # copies préparées et rapports générés
└── README.md
```

Les CSV bruts et préparés restent sur la machine et ne sont pas ajoutés à Git,
car ils sont volumineux.

## Exécuter le travail

1. Ouvrir `notebook/nettoyage_donnees.ipynb` dans VS Code.
2. Choisir l'environnement Python du projet (`.venv`) comme kernel.
3. Exécuter les cellules dans l'ordre.

Installer les bibliothèques nécessaires dans l'environnement Python si besoin :

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Le notebook contient le code et les explications : il montre les fichiers,
applique les corrections prudentes, effectue les tests de qualité et écrit le
rapport dans `data/processed/rapport_qualite_donnees.txt`.

Après le nettoyage, exécuter `notebook/clustering/exploration_donnees.ipynb`.
Il explore les habitudes de commande, les tailles des paniers, les produits
fréquents et les réachats. Cette étape prépare les analyses de segmentation et
d'associations, sans entraîner de modèle à ce stade.

Le notebook `notebook/clustering/segmentation_clients.ipynb` compare ensuite
K-means et DBSCAN sur un échantillon reproductible de 5 000 clients. Les groupes
sont exploratoires et leur qualité doit être validée avant toute décision métier.

Pour explorer les produits achetés ensemble, exécuter ensuite
`notebook/associations/regles_association.ipynb`. Ce notebook utilise FP-Growth
sur un échantillon reproductible de 50 000 paniers et les 50 produits candidats
les plus fréquents. Il faut aussi installer `mlxtend` :

```powershell
python -m pip install mlxtend
```

Le fichier `order_products__prior.csv` dépasse 32 millions de lignes ; son
traitement peut prendre plusieurs minutes. La progression s'affiche pendant
l'analyse. Ne pas interrompre la cellule avant sa fin.

Pour tester la prédiction du réachat, exécuter ensuite
`notebook/prediction/prediction_reachat.ipynb`. Le notebook entraîne une
régression logistique sur un échantillon reproductible de 20 000 commandes
`train`, en utilisant uniquement l'historique `prior`. Les candidats incluent
les produits déjà achetés par le client et les 20, 50 ou 100 produits les plus
fréquents dans `prior`. Le notebook compare la couverture, plusieurs stratégies
de classement et les performances pour ces trois tailles, puis les compare à
une règle simple. Il n'utilise pas les commandes `test` sans étiquettes.

Les données brutes ne sont jamais modifiées. Les espaces superflus des champs
texte sont retirés dans les copies préparées. Les délais manquants de
`days_since_prior_order` sur la première commande d'un client sont attendus et
ne sont pas remplacés arbitrairement.
