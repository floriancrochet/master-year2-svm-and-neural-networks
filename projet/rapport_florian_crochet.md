# Rapport de Projet : Modélisation et Prédiction du Risque de Défaut de Crédit

**Auteur :** Florian Crochet  
**Cadre :** M2 Économétrie et Statistiques (Machine Learning, SVM, Explicabilité) — Application métier en Asset Management Quantitative.

---

## 1. Introduction et Objectifs

Dans le contexte de la gestion d'actifs quantitative, l'optimisation du couple rendement/risque passe par une modélisation rigoureuse du risque de crédit. Ce projet conçoit un modèle algorithmique capable d'évaluer la probabilité de défaut de paiement d'emprunteurs (*Credit Scoring*) au sein d'un portefeuille obligataire de dette privée.

L'étude compare quatre algorithmes de Machine Learning (Régression Logistique, SVM linéaire calibré, Random Forest et HistGradientBoosting) et justifie les décisions du modèle retenu via des méthodes d'interprétabilité globale (Feature Importance, Permutation Feature Importance, PDP+ICE, SHAP Beeswarm) et locale (LIME, SHAP Waterfall), conformément aux exigences réglementaires de l'industrie financière institutionnelle.

## 2. Présentation des données

L'ensemble des données comprend 2 260 701 observations et 151 variables (101 numériques, 50 catégorielles), soit 108 millions de valeurs manquantes au total. Il provient de la plateforme *Lending Club* et est ingéré via le fichier `lending_club.parquet` (312 Mo), converti depuis le CSV brut par `polars`.

## 3. Préparation des données

### Nettoyage initial

Les variables post-octroi (pénalités, recouvrements, montants remboursés) et les variables non exploitables (identifiants, URL, intitulés de poste) sont supprimées pour éviter tout *data leakage*. 32 doublons sont détectés et supprimés. La variable cible `loan_status` est binarisée en « Défaut » (1) et « Non-défaut » (0), aboutissant à un déséquilibre de 87 % / 13 %. Les modalités rares de `home_ownership` sont fusionnées, `sub_grade` et `addr_state` supprimées, et les variables ordinales (`emp_length`) et temporelles (`issue_d`, `earliest_cr_line`) converties en valeurs numériques. Ces opérations déterministes ne comportent aucun risque de fuite de données.

### Séparation des données

Le jeu de données est séparé **avant toute transformation dépendante des données** en 80 % d'apprentissage (1 808 535 observations) et 20 % de test (452 134 observations), avec stratification sur la cible (13,06 % de défauts dans les deux jeux) et graine fixe (`random_state=42`).

### Valeurs manquantes

L'analyse préalable révèle que 0 % des observations sont complètes et 65 % présentent plus de 20 % de valeurs manquantes, ce qui justifie un traitement en deux temps. Les 56 259 observations présentant plus de 50 % de valeurs manquantes (soit 3,1 % du train) sont supprimées en premier — elles sont trop lacunaires pour être informatives. En second, les 35 variables dont le taux de manquants dépasse 30 % sont supprimées des deux jeux : 16 variables de demandes conjointes (`sec_app_*`, `*_joint`, 94–98 % de manquants), 6 variables d'historiques d'incidents (`mths_since_*`, 51–84 %), et 13 variables de crédit installment (`il_util`, `open_il_*`, `total_bal_il`, etc., 36–46 %). Pour celles-ci, l'imputation par la médiane serait fallacieuse. Le jeu de variables passe de 103 à 68. L'imputation numérique par la médiane et catégorielle par la constante « Unknown » est fittée exclusivement sur le train, puis appliquée au test.

### Statistiques descriptives

Les statistiques de position et de dispersion (*skewness*, *kurtosis*) pour chaque variable numérique sur le train révèlent des asymétries extrêmes (ex. `annual_inc` : skewness = 534, `tot_coll_amt` : skewness = 936) et des valeurs aberrantes (`dti` max = 999). La distribution de chacune des 8 variables catégorielles est analysée par leurs fréquences relatives.

### Distributions

Les histogrammes (avec estimateur de densité à noyau) des 60 variables numériques et les diagrammes en barres des 8 variables catégorielles confirment des distributions très asymétriques et des déséquilibres de modalités prononcés (ex. `purpose` : 57 % de consolidation de dette, `application_type` : 94 % de demandes individuelles).

### Q-Q plots

Les Q-Q plots, calculés sur un sous-échantillon de 5 000 observations par variable, révèlent des écarts systématiques à la normalité pour l'ensemble des 60 variables numériques : queues épaisses pour les variables monétaires (ex. `annual_inc`, `revol_bal`), patrons en escalier pour les variables discrètes de comptage (ex. `delinq_2yrs`, `pub_rec`), et discontinuités pour les variables à pics. Ces observations confirment l'usage de la corrélation de Spearman et justifient la Winsorization.

### Valeurs atypiques

Des boxplots stratifiés (Défaut / Non-défaut), produits sur un sous-échantillon équilibré de 25 000 observations par classe (50 000 au total), permettent d'identifier visuellement les distributions différentielles. Les variables les plus discriminantes sont le taux d'intérêt (`int_rate`), le score FICO et le ratio d'utilisation du crédit revolving (`revol_util`), dont les médianes diffèrent significativement entre les deux classes.

### Corrélations

La matrice de corrélation de Spearman identifie 43 paires avec |r| > 0,7, dont plusieurs quasi-doublons parfaits (ex. `fico_range_low` / `fico_range_high` : r = 1,000 ; `loan_amnt` / `funded_amnt` : r = 0,9999 ; `open_acc` / `num_sats` : r = 0,999). Huit variables redondantes confirmées sont supprimées des deux jeux (`funded_amnt`, `funded_amnt_inv`, `fico_range_high`, `last_fico_range_high`, `num_sats`, `tot_cur_bal`, `total_bal_ex_mort`, `total_il_high_credit_limit`), ramenant le jeu à 52 variables numériques et 8 catégorielles.

### Pipeline de prétraitement

Les transformations dépendantes des données sont encapsulées dans un `ColumnTransformer` sklearn fitté exclusivement sur le train, produisant 75 features en sortie. `Winsorizer` (écrêteur aux 1er/99e centiles) neutralise les valeurs aberrantes avant standardisation, tandis que `Log1pTransformer` (transformation log(1+x)) symétrise les distributions monétaires fortement asymétriques. Ces deux transformateurs sont implémentés comme classes sklearn personnalisées (héritant de `BaseEstimator` et `TransformerMixin`) avec `get_feature_names_out()` pour la compatibilité complète avec le pipeline :
- 9 variables monétaires : `Winsorizer` → `Log1pTransformer` → `StandardScaler`.
- 43 autres numériques : `Winsorizer` → `StandardScaler`.
- 4 variables binaires et `grade` : `OrdinalEncoder`.
- 3 variables nominales (`home_ownership`, `verification_status`, `purpose`) : `OneHotEncoder` (drop first) → 18 features.

## 4. Modélisation, Explicabilité et Interprétabilité

La distribution de la cible (87 % non-défaut, 13 % défaut) présente un déséquilibre modéré. Le paramètre `class_weight='balanced'` est activé sur les quatre modèles. Les métriques privilégiées sont le Recall et l'AUC-ROC.

Pour chaque modèle, trois analyses sont conduites :
- **Modélisation** : optimisation des hyperparamètres via `GridSearchCV` (3-fold, scoring AUC-ROC), rapport de classification et matrice de confusion.
- **Interprétabilité** : mécanismes natifs du modèle (coefficients ou Feature Importance) et Permutation Feature Importance modèle-agnostique.
- **Explicabilité** : SHAP Beeswarm (global) et SHAP Waterfall (local), ainsi que LIME (approximation locale linéaire).

### Régression Logistique

Modèle de référence linéaire. Les coefficients standardisés identifient directement les facteurs de risque. `GridSearchCV` optimise la régularisation (`C`, `penalty`) sur un sous-échantillon de 100 000 observations, puis le modèle est ré-entraîné sur le train complet. `LinearExplainer` SHAP est l'explainer le plus approprié.

#### Modélisation

Les meilleurs hyperparamètres sélectionnés sont C = 0,1 avec régularisation L1 (solveur SAGA), produisant un modèle parcimonieux. L'AUC-ROC en validation croisée sur le sous-échantillon atteint 0,950. Sur le test complet, le Recall sur la classe Défaut est de 89 % — priorité métier — pour une Precision de 57 % (compromis acceptable au vu du déséquilibre de classes).

#### Interprétabilité

Les coefficients L1 standardisés montrent que le score FICO (`last_fico_range_low`, `fico_range_low`) est la variable la plus protectrice contre le défaut (coefficient négatif fort), suivi du taux d’intérêt (`int_rate`, coefficient positif — plus le taux est élevé, plus le risque est fort). La Permutation Feature Importance valide le même classement, confirmant la stabilité de l’interprétation.

#### Explicabilité

Le SHAP Beeswarm confirme la dominance du score FICO et du taux d'intérêt à l'échelle globale. Sur l'individu testé localement (SHAP Waterfall + LIME), le score FICO très élevé (`last_fico_range_low` = −1 après standardisation) et un montant emprunté faible poussent fortement la prédiction vers le Non-défaut (probabilité de 87 %).


### Support Vector Machine (SVM)

Le `SVC` à noyau RBF présente une complexité quadratique O(n²) impraticable (>50 minutes sur 50 000 obs.). Le `LinearSVC` est substitué : sa complexité O(n·d) est compatible avec les ressources disponibles. La calibration isotonique (`CalibratedClassifierCV`) lui confère `predict_proba`. L'interprétabilité est directe via les coefficients du modèle linéaire. `LinearExplainer` SHAP est utilisé.

**Modélisation**

L’hyperparamètre optimal sélectionné est C = 0,1. L’AUC-ROC en validation croisée atteint 0,949. Sur le test, la précision globale est de 92 % mais le Recall Défaut est de 68 % — inférieur à la Régression Logistique (89 %) —, car la calibration isotonique tend à lisser les probabilités, rendant le modèle plus conservateur dans la détection d’incidents.

**Interprétabilité**

Les coefficients du `LinearSVC` montrent deux signaux dominants : `last_fico_range_low` (coefficient fortement négatif — le FICO protège du défaut) et les modalités à risque de `purpose` telles que `small_business`, `medical`, `house` (coefficients positifs élevés). Ce résultat souligne le rôle de la finalité du prêt dans la détection du risque, absent du classement de la Régression Logistique. La Permutation Feature Importance est ultraconcentrée sur `last_fico_range_low` (≈ 0,4 de diminution AUC-ROC), les autres variables étant quasiment nulles.

**Explicabilité**

Le SHAP Beeswarm confirme la dominance absolue de `last_fico_range_low` : les valeurs élevées du FICO (points rouges) génèrent des contributions SHAP négatives importantes (vers le Non-défaut), avec un impact pouvant dépasser +6 en valeur absolue pour les cas extrêmes. Le SHAP Waterfall de l’individu testé donne f(x) = 0,53, soit une prédiction légèrement au-dessus de la moyenne (E[f(X)] = −0,424), tirée vers le défaut par `last_fico_range_low` (+1,04 SHAP) malgré un versement mensuel faible (−0,29). Le LIME prédit Non-défaut à 44 % / Défaut à 56 % pour cet individu, avec `last_fico_range_low` comme principal facteur discriminant (poids 0,64).

### Random Forest

Modèle d'ensemble par bagging de 200 arbres (`max_depth=None`, `min_samples_split=5`), entraîné via `GridSearchCV` sur 100 000 observations puis ré-entraîné sur le train complet. L'interprétabilité est assurée par la Feature Importance Gini, la Permutation Feature Importance et les PDP + ICE. L'explicabilité repose sur `TreeExplainer` SHAP (Beeswarm, Waterfall) et LIME.

**Modélisation**

Les meilleurs hyperparamètres sont `max_depth=None`, `min_samples_split=5`, `n_estimators=200`. L'AUC-ROC en validation croisée atteint 0,943. Sur le test, le Recall Défaut est de 65 % pour une Precision de 74 % — profil inverse de la Régression Logistique (Recall = 89 %, Precision = 57 %) — traduisant une plus grande sélectivité dans les alertes de défaut. La matrice de confusion révèle 38 296 vrais positifs et 20 739 faux négatifs.

**Interprétabilité**

La Feature Importance Gini est fortement concentrée sur `last_fico_range_low` (≈ 0,53, contre ≤0,04 pour toutes les autres variables). La Permutation Feature Importance confirme : `last_fico_range_low` capte ≈ 0,37 de diminution AUC-ROC, `num__issue_d` et `num__int_rate` suivent loin derrière (≤0,01). Les PDP + ICE révèlent un effet monotone décroissant du FICO sur la probabilité de défaut, un effet non linéaire en cloche de `num__issue_d`, et un effet quasi-plat de `num__int_rate`.

**Explicabilité**

`TreeExplainer` SHAP (Beeswarm, Waterfall, Scatter) et LIME permettent de décomposer les prédictions à l'échelle globale et locale, en cohérence avec les importances Gini et la Permutation Feature Importance.


### HistGradientBoosting

Implémentation native sklearn du Gradient Boosting histogrammique (équivalent LightGBM). Sa complexité O(n·d) permet l'entraînement sur le train complet (1,75 M obs.). `early_stopping=True` (avec `validation_fraction=0.1`, `n_iter_no_change=10`) évite le sur-apprentissage sans nécessiter de jeu de validation séparé. L'explicabilité repose sur `TreeExplainer` SHAP et LIME.

## 5. Synthèse et sélection du meilleur modèle

Un tableau récapitulatif compare les quatre modèles sur cinq métriques (Accuracy, Precision, Recall, F1, AUC-ROC). Les courbes ROC superposées offrent une comparaison visuelle directe. Le meilleur modèle est sélectionné sur l'AUC-ROC et ses hyperparamètres finaux sont documentés avec sa matrice de confusion.

## 6. Conclusions et Pistes d'Amélioration

Ce projet a permis de construire un pipeline complet de Credit Scoring avec une rigueur absolue en matière de prévention du *data leakage*. Le code est structuré en modules réutilisables (`utils.py`) et en pipelines sklearn. La comparaison des quatre algorithmes met en évidence les compromis entre performance, temps d'entraînement et interprétabilité. Parmi les pistes d'amélioration : enrichissement par des données macroéconomiques, optimisation des seuils de classification par méthode de coût-bénéfice, ou déploiement via une API de scoring en temps réel.
