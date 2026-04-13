# Lending Club — Credit Risk Scoring

## Données

Les données brutes (~1 Go) ne sont pas versionnées. Suivre les deux étapes ci-dessous pour les reconstituer localement.

### Prérequis

1. **Compte Kaggle** avec une clé API : générer le fichier `~/.config/kaggle/kaggle.json` depuis [kaggle.com/settings](https://www.kaggle.com/settings).
2. **Environnement Python** installé via `uv` :

```bash
uv sync
```

### Étape 1 — Télécharger le dataset

```bash
uv run python download_data.py
```

Le script appelle `kagglehub` pour télécharger le dataset [`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club) et déplace les fichiers CSV dans `data/`.

### Étape 2 — Convertir en Parquet

```bash
uv run python process_data.py
```

Le script lit `data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` via **Polars** (scan lazy) et produit `data/lending_club.parquet`, nettement plus compact et rapide à charger.

> Le dossier `data/` est exclu du dépôt git (`.gitignore`). Le fichier Parquet doit être regénéré localement après clonage.
