"""Fonctions utilitaires et transformateurs personnalisés pour le pipeline de Credit Scoring."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


# ── Transformateurs sklearn ────────────────────────────────────────────────────

class Winsorizer(BaseEstimator, TransformerMixin):
    """Écrêtage des valeurs extrêmes aux percentiles lower/upper, fitté sur le train."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper * 100, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])
        return np.array(input_features)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Transformation log1p avec support de get_feature_names_out pour ColumnTransformer."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])
        return np.array(input_features)


# ── Fonctions de visualisation EDA ────────────────────────────────────────────

def plot_missing(df, label=""):
    """Affiche un tableau des variables avec des valeurs manquantes (taux et effectif)."""
    missing = df.isnull().mean().rename("taux")
    missing_n = df.isnull().sum().rename("effectif")
    result = pd.concat([missing, missing_n], axis=1)
    result = result[result["taux"] > 0].sort_values("taux", ascending=False)
    if result.empty:
        print(f"{label} — Aucune valeur manquante.")
    else:
        print(f"{label} — {len(result)} variables avec valeurs manquantes :")
        display(result.style.format({"taux": "{:.1%}"}))


def plot_distributions(df, num_cols, cat_cols):
    """Histogrammes (numériques) et diagrammes en barres (catégorielles)."""
    if num_cols:
        n_c, n_r = 4, (len(num_cols) + 3) // 4
        fig, axes = plt.subplots(n_r, n_c, figsize=(16, n_r * 3.5))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), ax=axes[i], bins=30, kde=True)
            axes[i].set_title(col)
            axes[i].set_ylabel("Effectif")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    if cat_cols:
        n_c, n_r = 3, (len(cat_cols) + 2) // 3
        fig, axes = plt.subplots(n_r, n_c, figsize=(16, n_r * 4))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            order = df[col].value_counts().iloc[:15].index
            sns.countplot(data=df, y=col, ax=axes[i], order=order)
            axes[i].set_title(col)
            axes[i].set_xlabel("Effectif")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


def plot_qq(df, num_cols, sample_size=5000):
    """Q-Q plots (vs loi normale) pour chaque variable numérique, sur un sous-échantillon."""
    n_c, n_r = 4, (len(num_cols) + 3) // 4
    fig, axes = plt.subplots(n_r, n_c, figsize=(16, n_r * 3.5))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sample = df[col].dropna()
        if len(sample) > sample_size:
            sample = sample.sample(sample_size, random_state=42)
        stats.probplot(sample, dist="norm", plot=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel("Quantiles théoriques")
        axes[idx].set_ylabel("Quantiles observés")
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_boxplots(X, y, num_cols, sample_per_class=25000):
    """Boxplots stratifiés (Défaut / Non-défaut) sur échantillon équilibré."""
    df_viz = X.copy()
    df_viz["target"] = y.values
    df_viz["Classe"] = df_viz["target"].map({0: "Non-défaut", 1: "Défaut"})
    n_def = min(sample_per_class, df_viz["target"].eq(1).sum())
    df_sample = pd.concat([
        df_viz[df_viz["target"] == 1].sample(n_def, random_state=42),
        df_viz[df_viz["target"] == 0].sample(sample_per_class, random_state=42),
    ])
    n_c, n_r = 4, (len(num_cols) + 3) // 4
    fig, axes = plt.subplots(n_r, n_c, figsize=(16, n_r * 4))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.boxplot(data=df_sample, x="Classe", y=col, ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel("")
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_correlation(df, num_cols, threshold=0.7):
    """Heatmap triangulaire inférieure de Spearman, filtrée aux variables corrélées au-delà du seuil."""
    corr = df[num_cols].corr(method="spearman")
    corr_vals = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_vals, 0)
    corr_zeroed = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
    highly_corr_mask = (corr_zeroed.abs() > threshold).any(axis=1)
    highly_corr_cols = corr.index[highly_corr_mask]

    if len(highly_corr_cols) > 0:
        filtered_corr = df[highly_corr_cols].corr(method="spearman")
        mask = np.triu(np.ones_like(filtered_corr, dtype=bool), k=1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(filtered_corr, mask=mask, cmap="coolwarm",
                    linewidths=0.5, linecolor="white")
        plt.title(f"Corrélation de Spearman (|r| > {threshold})")
        plt.show()
    else:
        print(f"Aucune paire ne dépasse le seuil de {threshold}.")


def get_corr_pairs(df, num_cols, threshold=0.7):
    """Retourne un DataFrame des paires de variables numériques avec |Spearman| > threshold."""
    corr = df[num_cols].corr(method="spearman")
    upper = corr.where(np.triu(np.ones_like(corr, dtype=bool), k=1))
    pairs = (
        upper.stack().rename("correlation").reset_index()
        .rename(columns={"level_0": "Variable 1", "level_1": "Variable 2"})
    )
    pairs["abs_corr"] = pairs["correlation"].abs()
    pairs = (pairs[pairs["abs_corr"] > threshold]
             .sort_values("abs_corr", ascending=False)
             .drop(columns="abs_corr"))
    return pairs.reset_index(drop=True)
