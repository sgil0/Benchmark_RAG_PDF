# Benchmark RAG : Évaluation Comparative de Solutions d'IA

Ce projet est une infrastructure de test permettant d'évaluer et de comparer la performance de différentes solutions de **RAG (Retrieval-Augmented Generation)**. Il automatise l'interrogation de plusieurs assistants, l'évaluation de la pertinence de leurs réponses via un LLM "Juge" (GPT-4o), et la génération de rapports statistiques détaillés.

## Contexte de Test
Le dataset est composé de 213 fichiers (212 pdfs / 1 word) plus ou moins complexeset de 103 questions classés en différents types
Afin de garantir l'équité du benchmark, tous les assistants sont évalués selon un protocole strictement identique :

* **Approche "Out-of-the-box"** : Aucune instruction spécifique visant à orienter le comportement, le ton ou le format de réponse n'est fournie aux assistants testés.
* **Documents Bruts** : Le corpus documentaire est fourni aux assistants sans prétraitement manuel ni indexation spécifique autre que celle par défaut de la solution.
* **Interrogation Neutre** : Les questions sont posées directement, sans exemples (*Zero-Shot*) ni contexte additionnel.

L'objectif est de mesurer la performance native et l'autonomie de la brique RAG (Retrieval + Génération) de chaque solution, sans optimisation avancée ("Prompt Engineering" ou réglages d'hyperparamètres).

## Jeu de Données (Dataset)

Le benchmark s'appuie sur un corpus hétérogène et un jeu de questions calibré pour tester différentes capacités cognitives des modèles.

### Corpus Documentaire
Le dataset est composé de **213 fichiers** (212 PDFs et 1 document Word) de complexité variable, incluant :
* **Documents techniques** (manuels automobiles, anciens scans, spécifications électroniques).
* **Documents administratifs** (factures, fiches de paie, extraits Kbis).
* **Recherche académique** (papiers scientifiques sur l'IA et les sciences cognitives).
* **Contenus divers** (menus de restaurants, recettes de cuisine).

### Typologie des Questions
Le benchmark comporte **103 questions** classées en 5 catégories de difficulté croissante. Chaque type de question rapporte un nombre de points spécifique reflétant la complexité du raisonnement attendu :

| Type | Pts | Description |
| :--- | :---: | :--- | 
| **Boolean** | **2** | Questions fermées nécessitant une vérification factuelle simple (Vrai/Faux, Oui/Non).|
| **Attribute** | **3** | Extraction d'une valeur précise, d'une entité nommée ou d'une caractéristique explicite. |
| **Constraint** | **5** | Recherche multicritère nécessitant de filtrer les résultats (dates, montants, seuils).|
| **Ranking** | **5** | Nécessite d'identifier plusieurs éléments et de les trier selon un ordre logique (chronologique, alphabétique, valeur).|
| **Multi-hop** | **10** | **Niveau Expert**. Nécessite de croiser des informations situées dans plusieurs documents ou paragraphes différents pour déduire la réponse (raisonnement en plusieurs étapes).|

Cette gradation permet d'identifier précisément le seuil de compétence de chaque assistant.

## Fonctionnalités

* **Multi-Assistant** : Support intégré pour trois architectures différentes :
    * **ChatGPT** (via OpenAI Assistants API avec `file_search`).
    * **Dimarc** (Solution propriétaire via API REST).
    * **Vertex AI** (Google Discovery Engine + Gemini Pro).
* **Évaluation Automatisée** : Utilisation de **GPT-4o** comme juge impartial pour noter les réponses générées par rapport à une vérité terrain fournie par l'humain.
* **Métriques Précises** :
    * Taux de réponses correctes.
    * Score pondéré.
    * Détection des "Non-réponses".
* **Analyse de Stabilité** : Exécution de tests multiples (runs séquentiels) pour calculer la variance et l'écart-type des performances.
* **Rapports Détaillés** : Export automatique des résultats en JSON (détail par question) et CSV (tableaux récapitulatifs).

## Structure du Projet

L'arborescence principale s'organise autour des dossiers par assistant :

```text
benchmark_rag/
├── benchmark_100_query_pdf/
│   ├── input/
│   │   ├── benchmark_PDF_WORD_querys_200.json  # Jeu de données (103 questions + réponses attendues)
│   │   └── promptEvaluateur.txt                # Instructions système pour le Juge (GPT-4o)
│   ├── ChatGPT/
│   │   ├── benchmark_ChatGPT.py                # Script d'exécution pour ChatGPT
│   │   └── ChatGPT_output/                     # Rapports bruts générés
│   ├── Dimarc/
│   │   ├── benchmark_Dimarc.py                 # Script d'exécution pour Dimarc
│   │   └── Dimarc_output/
│   └── VertexAI_google/
│       ├── benchmark_VertexAI_google.py        # Script d'exécution pour Vertex AI
│       └── VertexAI_google_output/
├── résultats/
│   ├── synthese_resultats.webp                 # Synthèse visuelle des performances comparées
│   └── lien_figma.txt                          # Document contenant le lien vers le Figma des résultats détaillés
└── requirements.txt                            # Dépendances Python
```


## Installation

### Prérequis
* Python 3.10+
* Clés API pour les services utilisés (OpenAI, Google Cloud, Dimarc).

### Installation des dépendances
Installez les paquets requis via `pip` :

```bash
pip install -r requirements.txt
```

## Configuration
Chaque script de benchmark attend des variables d'environnement spécifiques. 
Vous devez créer un fichier .env à la racine ou configurer vos variables système.

**Variables Communes :**
- `OPENAI_API_KEY` : Clé API OpenAI (requise pour le Juge GPT-4o).

**Variables Spécifiques par Assistant :**

| Assistant | Variable Requise | Description |
| :--- | :--- | :--- |
| **ChatGPT** | `ASSISTANT_ID` | ID de l'assistant OpenAI configuré avec les documents. |
| **Dimarc** | `CLIENT_ID` | Identifiant client pour l'auth Dimarc. |
| | `CLIENT_SECRET` | Secret client pour l'auth Dimarc. |
| | `DIMARC_URL` | URL de base de l'API (prod/dev). |
| | `DOCUMENTALIST_ID` | ID de l'agent documentaliste cible. |
| **Vertex AI**| `PROJECT_ID` | ID du projet Google Cloud. |
| | `DATA_STORE_ID` | ID du Data Store (Discovery Engine). |
| | `GOOGLE_APPLICATION_CREDENTIALS` | Chemin vers le fichier JSON de clé de service GCP. |


## Utilisation
Pour lancer un benchmark, naviguez vers le dossier de l'assistant souhaité et exécutez le script Python correspondant.
Exemple avec ChatGPT :
```Bash
cd benchmark_100_query_pdf/ChatGPT
python benchmark_ChatGPT.py
```

Exemple avec Vertex AI :
```Bash
cd benchmark_100_query_pdf/VertexAI_google
python benchmark_VertexAI_google.py
```

*Le script va exécuter par défaut une série de 10 itérations (runs) pour assurer la significativité statistique des résultats.*

## Méthodologie d'Évaluation

1.  **Interrogation (Step 1)** : Le script charge les questions depuis `benchmark_PDF_WORD_querys_200.json`. Chaque question possède un `id`, une `query`, une `expected_answer` et un nombre de `points`.
2.  **Génération** : L'assistant génère une réponse basée sur les documents fournis.
3.  **Notation (Step 2)** : Le script envoie la triplette (Question, Réponse Assistant, Réponse Attendue) à **GPT-4o**.
4.  **Critères** : Le Juge évalue la réponse selon le prompt `promptEvaluateur.txt` :
    * **Correct** : Contient les informations clés, pas d'hallucination, chiffres exacts.
    * **Incorrect** : Omission d'information, contre-sens ou réponse "Je ne sais pas".
5.  **Agrégation** : Un fichier CSV global est mis à jour à la fin de l'exécution pour calculer la moyenne et l'écart-type des scores sur l'ensemble des runs.

## Résultats (Output)

Les résultats sont stockés dans le dossier `_output` de chaque assistant :
* **`json/`** : Fichiers détaillés contenant chaque question, la réponse générée, la justification du juge et la note attribuée.
* **`csv/`** : Tableaux récapitulatifs par "Run" (performance par catégorie de question).
* **`global_summary_*.csv`** : Fichier consolidé présentant les moyennes et écarts-types (ex: `83.5% ± 2.1`) pour juger de la fiabilité.

Ces résultats sont consultables directements via le `liens_figma.txt` ou `synthese_resultats_webp` mais il faut les mettre à jour manuellement avec les informations de `global_summary_*.csv`.

## Auteur

Ce benchmark a été réalisé par **Samuel Gillon**, apprenti ingénieur en IA, sous la direction de **Bryan Fruchart**, docteur en sciences cognitives.

