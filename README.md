# Analyse des rappels de produits (RappelConso)

Ce projet explore les rappels de produits publiés sur RappelConso. Il comprend
un notebook de préparation/exploration et un premier essai de regroupement des
rappels par motif avec TF-IDF et K-means.

> **Projet exploratoire :** les groupes calculés ne sont pas des catégories
> officielles et ne doivent pas être interprétés comme une classification
> définitive. Les résultats dépendent du texte, de sa normalisation et des
> paramètres du modèle.

## Contenu du dépôt

```text
.
├── notebook/
│   ├── nettoyage_donnees.ipynb
│   └── clustering/
│       └── exploration_donnees.ipynb
├── data/
│   └── raw/
│       └── rappelconso-v2.csv   # à récupérer séparément (non versionné)
└── README.md
```

Les chemins sont en minuscules dans GitHub. Selon le système d’exploitation,
le dossier peut s’afficher avec une casse différente.

## Données

Le notebook utilise le fichier CSV public `rappelconso-v2.csv`, séparé par des
points-virgules (`;`). Le fichier présent lors de l’analyse contenait **19 258
lignes et 34 colonnes**.

Le CSV n’est pas inclus dans le dépôt : `data/raw/` est ignoré par Git. Après
avoir cloné le projet :

1. Récupérer le fichier CSV depuis la source publique RappelConso
   (https://data.economie.gouv.fr/).
2. Créer le dossier `data/raw/` à la racine du projet s’il n’existe pas.
3. Y placer le fichier sous le nom exact `rappelconso-v2.csv`.

Le fichier de données ne sera donc pas téléchargé automatiquement lors d’un
`git clone`. Il faut aussi vérifier les conditions de réutilisation et la date
de mise à jour de la version utilisée.

## Installation

Prérequis : Git et Python 3.10 ou plus récent. La version utilisée pour la
dernière exécution locale était Python 3.13.

Depuis la racine du projet, créer et activer un environnement virtuel :

```bash
python3 -m venv .venv
```

macOS / Linux :

```bash
source .venv/bin/activate
```

Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Installer les bibliothèques nécessaires :

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn jupyter
```

Dans VS Code, ouvrir le dépôt, installer/activer l’extension Jupyter si
nécessaire, puis choisir l’interpréteur Python de l’environnement virtuel comme
kernel du notebook.

## Exécution des notebooks

Exécuter les cellules dans l’ordre, dans le même notebook/kernel.

### 1. Nettoyage et exploration

Ouvrir `notebook/nettoyage_donnees.ipynb`. Ce notebook charge le CSV et commence
son exploration/nettoyage.

Le chemin de données écrit dans ce notebook est `../data/raw/rappelconso-v2.csv`.
Il suppose que le répertoire de travail du kernel est `notebook/`. Si le fichier
est introuvable, vérifier le répertoire de travail du kernel et la présence du
CSV à l’emplacement indiqué.

### 2. Exploration du clustering

Ouvrir `notebook/clustering/exploration_donnees.ipynb`. Ce notebook charge et
prépare les données, puis compare plusieurs modèles K-means sur la colonne
`motif_rappel`.

Son chemin de données est `../../data/raw/rappelconso-v2.csv`. Il suppose que le
répertoire de travail du kernel est `notebook/clustering/`.

Les chemins relatifs sont résolus depuis le **répertoire de travail du kernel**,
qui peut différer de l’emplacement du fichier notebook. Si le notebook ne trouve
pas les données, vérifier ce répertoire ou adapter le chemin d’entrée.

## Méthode de clustering actuelle

Le notebook de clustering :

1. normalise les accents et la ponctuation, et harmonise quelques variantes
   orthographiques explicitement prévues (notamment Listeria, Salmonelle et
   *E. coli*) ;
2. regroupe les formulations qui deviennent identiques après normalisation ;
3. transforme les textes en vecteurs TF-IDF (unigrammes et bigrammes) ;
4. entraîne K-means pour `k = 4, 6, 8, 10, 12` ;
5. compare les modèles avec un score de silhouette calculé en distance cosinus
   sur un échantillon de formulations distinctes ;
6. affiche les tailles des groupes, les termes représentatifs et les motifs
   fréquents du modèle ayant le meilleur score parmi les valeurs testées.

La colonne `cluster_motif` est ajoutée au DataFrame en mémoire lors de
l’exécution. Elle n’est pas exportée automatiquement dans un fichier.

### Résultats de la dernière exécution enregistrée

La normalisation a ramené les formulations distinctes de **9 657 à 8 915**.
Les scores obtenus étaient :

| Nombre de groupes (`k`) | Silhouette cosinus |
|---:|---:|
| 4 | 0,0193 |
| 6 | 0,0297 |
| 8 | 0,0378 |
| 10 | 0,0371 |
| 12 | **0,0398** |

`k = 12` est le meilleur score parmi ces essais, mais **0,0398 reste faible**.
Le plus grand groupe représente environ **37,1 %** des rappels et mélange des
motifs variés. D’autres groupes sont plus faciles à interpréter, par exemple
Listeria, Salmonelle, oxyde d’éthylène, corps étrangers ou certains risques
d’étouffement.

Les numéros de groupes (par exemple « groupe 1 ») sont arbitraires : ils
peuvent changer si les paramètres, les données ou l’environnement logiciel
changent. Le score de silhouette ne suffit pas à lui seul pour choisir un
modèle ; il faut aussi examiner les exemples et vérifier que les groupes ont un
sens métier.

## Suite conseillée

#1. Examiner les motifs fréquents et des exemples de chaque groupe, en particulier
   le grand groupe hétérogène.
#2. Vérifier la normalisation avec une personne connaissant le domaine : ne pas
   fusionner automatiquement des formulations comme « suspicion », « présence »
   et « détection », qui peuvent avoir des sens différents.
#3. Essayer de segmenter séparément les grandes catégories de produits ou de
   risques, puis comparer l’interprétabilité des groupes.
#4. Tester d’autres représentations textuelles et méthodes, en conservant une
   évaluation qualitative et des métriques comparables.
#5. Une fois une segmentation jugée utile, exporter les identifiants de rappel
   et les groupes dans un fichier de résultats versionné ou régénérable.



## Reproductibilité et limites

- `random_state=42` est utilisé pour K-means et pour l’échantillon du score de
  silhouette ; cela améliore la répétabilité dans un environnement identique.
- Les versions des bibliothèques ne sont pas encore verrouillées dans un fichier
  de dépendances. Les résultats peuvent donc légèrement varier selon les
  versions.
- Les catégories, identifiants et autres variables du dataset ne sont pas
  utilisées dans cette première segmentation : elle porte uniquement sur le
  texte de `motif_rappel`.
- L’analyse ne remplace ni une validation métier ni les informations officielles
  des fiches de rappel.
