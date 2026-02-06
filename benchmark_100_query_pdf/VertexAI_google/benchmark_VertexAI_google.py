import json
import asyncio
import glob
import statistics
import os
import csv
import re
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from datetime import datetime
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import asyncio
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

# Configuration
load_dotenv(find_dotenv())

# Google Vertex AI
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION", "global")
DATA_STORE_ID = os.getenv("GOOGLE_DATA_STORE_ID")
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if google_creds_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path

# Initialisation unique de Gemini Pro
vertexai.init(project=PROJECT_ID, location="us-central1") # Ou "us-central1"
model_gemini = GenerativeModel("gemini-2.5-pro")


MODEL_JUDGE = "gpt-4o"
TESTED_ASSISTANT = "VertexAI_google"

# Fichiers
BENCHMARK_QUERYS_JSON = "../input/benchmark_PDF_WORD_querys_200.json"
PROMPT_EVALUATEUR_PATH = "../input/promptEvaluateur.txt"
OUTPUT_PATH_JSON = f"{TESTED_ASSISTANT}_output/json/"
OUTPUT_PATH_CSV = f"{TESTED_ASSISTANT}_output/csv/"
OUTPUT_PATH_CSV_GLOBAL = f"{TESTED_ASSISTANT}_output/"

# Tools
def load_json(path_str):
    with open(path_str, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_text(path_str):
    with open(path_str, 'r', encoding='utf-8') as f:
        return f.read().strip()

def save_json(data, path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def ask_vertex_one(question_obj, semaphore, serving_config, client):
    
    async with semaphore:
        max_retries = 5
        base_delay = 5

        print(f"Pose {question_obj.get('id')} (Vertex + Gemini)...")

        for attempt in range(max_retries):
            try:
                # 1. RECHERCHE DANS LE VECTOR STORE (RAG)
                request = discoveryengine.SearchRequest(
                    serving_config=serving_config,
                    query=question_obj["query"],
                    page_size=5, # 5 Documents
                    content_search_spec={
                        # On désactive le résumé flou
                        "snippet_spec": {"return_snippet": False},
                        # On active les SEGMENTS PRÉCIS (Layout Parser)
                        "extractive_content_spec": {
                            "max_extractive_segment_count": 5, # 5 segments par document max
                            "return_extractive_segment_score": True
                        }
                    }
                )
                
                response = await client.search(request)

                # 2. PRÉPARATION DU CONTEXTE 
                context_segments = []
                
                if response.results:
                    for result in response.results:
                        data = result.document.derived_struct_data
                        
                        # On lit 'extractive_segments'
                        if "extractive_segments" in data:
                            for segment in data["extractive_segments"]:
                                content = segment.get("content", "").strip()
                                if content:
                                    context_segments.append(content)
                
                # On nettoie et on assemble
                if not context_segments:
                    question_obj["answer"] = "Aucune information pertinente trouvée dans les documents."
                    return question_obj

                # On crée un bloc de texte propre
                context_text = "\n\n--- DOCUMENT EXTRACT ---\n".join(context_segments)


                # 3. GÉNÉRATION (GEMINI PRO)
                prompt = f"""
                Tu es un auditeur expert. Utilise EXCLUSIVEMENT les extraits de documents ci-dessous pour répondre.
                
                RÈGLES :
                1. Cite le contexte si possible.
                2. Si la réponse n'est pas dans les extraits, dis "Information non trouvée".
                3. Sois précis sur les chiffres.

                --- CONTEXTE DOCUMENTAIRE ---
                {context_text}
                -----------------------------
                
                QUESTION : {question_obj["query"]}
                
                RÉPONSE :
                """

                # Appel asynchrone propre
                def call_gemini():
                    # Attention : assurez-vous que model_gemini est bien initialisé ailleurs
                    return model_gemini.generate_content(prompt).text

                response_text = await asyncio.to_thread(call_gemini)
                question_obj["answer"] = response_text
                return question_obj

            # GESTION DES QUOTAS
            except Exception as e:
                # On capture tout (ResourceExhausted est parfois encapsulé)
                error_str = str(e)
                if "429" in error_str or "Quota" in error_str or "ResourceExhausted" in str(type(e)):
                    if attempt == max_retries - 1:
                        print(f"❌ Abandon Q{question_obj['id']} après {max_retries} essais.")
                        question_obj["answer"] = f"Erreur Quota: {e}"
                        return question_obj
                    
                    wait_time = base_delay * (2 ** attempt)
                    print(f"⚠️ Quota Vertex/Gemini. Pause {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Autre erreur (ex: bug code) -> on arrête tout de suite
                    print(f"❌ Erreur technique sur Q{question_obj['id']}: {e}")
                    question_obj["answer"] = f"Erreur technique: {e}"
                    return question_obj

    return question_obj
        
# Étape 1 : Benchmark - Interrogation de Gemini
async def step_1_run_benchmark(questions: list) -> list:
    # Interroge l’assistant et renvoie les réponses
    semaphore = asyncio.Semaphore(5)

    # Configuration du client
    client_options = (
        ClientOptions(api_endpoint=f"{LOCATION}-discoveryengine.googleapis.com")
    )
    # Création du client asynchrone
    client = discoveryengine.SearchServiceAsyncClient(client_options=client_options)

    # Construction de la ressource (le chemin vers ton moteur)
    serving_config = client.serving_config_path(
        project=PROJECT_ID,
        location=LOCATION,
        data_store=DATA_STORE_ID,
        serving_config="default_config",
    )

    tasks = [ask_vertex_one(query, semaphore, serving_config, client) for query in questions]
    results = await asyncio.gather(*tasks)    
    return results

# Étape 2 : Évaluation
async def grade_one(client, question_data, system_prompt, semaphore):
    async with semaphore:
        print(f"Note {question_data.get('id')} par ({MODEL_JUDGE})...")
        
        user_prompt = """
        ## Question posée
        {initial_query}
        ## Réponse à évaluer
        {answer}
        ## Correction de référence
        {correction}
        ---
        Évalue si la réponse est correcte selon les critères définis. Réponds en JSON.""".format(
            initial_query=question_data.get('query'),
            answer=question_data.get('answer'),
            correction=question_data.get('expected_answer')
        )
    
        
        q_type = question_data.get("type", "General")

        try:
            completion = await client.chat.completions.create(
                model=MODEL_JUDGE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            grade = json.loads(completion.choices[0].message.content)
            
            return {
                "id": question_data.get("id"),
                "type": q_type, 
                "query": question_data.get("query"),
                "expected_answer": question_data.get("expected_answer"),
                "answer": question_data.get("answer"),
                "justification": grade.get("justification", ""),
                "is_correct": grade.get("is_correct"),
                "points_awarded": question_data.get("query_points", 0) if grade.get("is_correct") else 0,
                "query_points": question_data.get("query_points", 0),
            }
        except Exception as e:
            print(f"Erreur notation: {e}")
            return {
                "id": question_data.get("id"), 
                "type": q_type,
                "is_correct": "erreur", 
                "points_awarded": 0,
                "query_points": question_data.get("query_points", 0)
            }

async def step_2_run_evaluation(benchmark_results):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(10)
    prompt = load_text(PROMPT_EVALUATEUR_PATH)
    
    tasks = [grade_one(client, q, prompt, semaphore) for q in benchmark_results]
    evaluations = await asyncio.gather(*tasks)
    
    # Calculs pour le JSON de sauvegarde
    total_earned = sum(e["points_awarded"] for e in evaluations)
    total_possible = sum(e["query_points"] for e in evaluations)
    
    total_correct = sum(e["is_correct"] for e in evaluations)
    
    final_score = ((total_correct * 100) / len(evaluations))
    
    summary = {
        "global_summary": True,
        "nomAssistant": TESTED_ASSISTANT,
        "note_finale": round(final_score, 2),
        "details": f"{total_earned}/{total_possible} pts"
    }
    
    return evaluations + [summary]

# Calcul des stats
def calculate_statistics(evaluations):
    """Calcule les statistiques agrégées par type + le total."""
    stats_by_type = {}
    global_stats = {"correct": 0, "total": 0, "pts_ok": 0, "pts_max": 0, "No_response": 0}
    
    for item in evaluations:
        if item.get("global_summary"): continue
            
        q_type = item.get("type", "Autre")
        if q_type not in stats_by_type:
            stats_by_type[q_type] = {"correct": 0, "total": 0, "pts_ok": 0, "pts_max": 0, "No_response": 0}
        
        s = stats_by_type[q_type]
        
        # Mise à jour locale
        s["total"] += 1
        s["pts_ok"] += item["points_awarded"]
        s["pts_max"] += item["query_points"]

        # Compteur de non-réponses
        if item["answer"] == "" :
            s["No_response"] += 1
            global_stats["No_response"] += 1

        
        # Mise à jour globale
        global_stats["total"] += 1
        global_stats["pts_ok"] += item["points_awarded"]
        global_stats["pts_max"] += item["query_points"]
        
        if item["is_correct"]:
            s["correct"] += 1
            global_stats["correct"] += 1

    # On ajoute le TOTAL à la liste des types pour faciliter l'export
    stats_by_type["TOTAL"] = global_stats
    return stats_by_type

# AFFICHAGE MARKDOWN (Gemini)
def print_markdown_table(stats_by_type, assistant_name):
    sorted_types = sorted(stats_by_type.keys())
    # On force TOTAL à la fin si présent
    if "TOTAL" in sorted_types:
        sorted_types.remove("TOTAL")
        sorted_types.append("TOTAL")
    
    # Largeurs des colonnes
    col_width = 25  # Un peu moins large car moins de texte par cellule
    name_width = 25 # Plus large pour les libellés des lignes
    
    # 1. Initialisation des lignes (En-tête + 3 lignes de données)
    header = f"| {assistant_name:<{name_width}} |"
    sep_line = f"| {'-'*name_width} |"
    
    row_correct = f"| {'Correct / Total':<{name_width}} |"
    row_score   = f"| {'Pts ok / Pts max':<{name_width}} |"
    row_no_resp = f"| {'No response / Total':<{name_width}} |"
    
    # 2. Boucle sur chaque type de question pour remplir les colonnes
    for t in sorted_types:
        s = stats_by_type[t]
        
        # --- Calculs ---
        pct_c = (s['correct'] / s['total'] * 100) if s['total'] else 0
        pct_p = (s['pts_ok'] / s['pts_max'] * 100) if s['pts_max'] else 0
        pct_error = (s['No_response'] / s['total'] * 100) if s['total'] else 0
        
        # --- Construction de l'en-tête et du séparateur ---
        header += f" {t:<{col_width}} |"
        sep_line += f" {'-'*col_width} |"

        # --- Construction des valeurs pour chaque ligne ---
        val_c = f"{s['correct']}/{s['total']} ({pct_c:.0f}%)"
        val_p = f"{s['pts_ok']}/{s['pts_max']} ({pct_p:.0f}%)"
        val_e = f"{s['No_response']}/{s['total']} ({pct_error:.0f}%)"
        
        # --- Ajout aux lignes respectives ---
        row_correct += f" {val_c:<{col_width}} |"
        row_score   += f" {val_p:<{col_width}} |"
        row_no_resp += f" {val_e:<{col_width}} |"

    # 3. Affichage final
    full_sep = "-" * len(header)
    
    print("\n" + full_sep)
    print(header)      # Ligne 1: Titres
    print(full_sep)    # Séparateur
    print(row_correct) # Ligne 2: Correct query
    print(row_score)   # Ligne 3: Points
    print(row_no_resp) # Ligne 4: No response
    print(full_sep + "\n")

# EXPORT CSV
def save_stats_to_csv(stats_by_type, assistant_name, output_path):
    # Préparation des colonnes (Tri alphabétique + TOTAL à la fin)
    keys = sorted([k for k in stats_by_type.keys() if k != "TOTAL"])
    keys.append("TOTAL")
    
    # Création du fichier
    # csv.QUOTE_MINIMAL permet de gérer les caractères spéciaux si besoin
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';') # Point-virgule pour Excel fr, sinon ','
        
        # Ligne d'en-tête
        headers = [assistant_name] + keys
        writer.writerow(headers)
        
        # Ligne de données
        row = ["Pdf / word"] # Ou "Pdf / word" selon votre préférence
        
        for t in keys:
            s = stats_by_type[t]
            # correct query / total query
            pct_c = (s['correct']/s['total']*100) if s['total'] else 0
            # pts ok / pts max
            pct_p = (s['pts_ok']/s['pts_max']*100) if s['pts_max'] else 0
            # no response / total query
            pct_error = (s['No_response']/s['total']*100) if s['total'] else 0
            
            # Format: "10/15 (66%) | 30/45 (66%) | 2/15 (13%)"
            cell_value = f"{s['correct']}/{s['total']} ({pct_c:.0f}%) | {s['pts_ok']}/{s['pts_max']} ({pct_p:.0f}%) | {s['No_response']}/{s['total']} ({pct_error:.0f}%)"
            row.append(cell_value)
            
        writer.writerow(row)
    
    print(f"Tableau CSV exporté : {output_path}")

def extract_full_data_regex(text_part):
    """
    Extrait numérateur, dénominateur et pourcentage via Regex pour plus de robustesse.
    Format attendu : "10/12 (83%)" ou "10.5/12 (83.3%)"
    """
    # Regex : cherche un nombre, un slash, un nombre, des parenthèses et un %
    pattern = r"([\d\.]+)\s*/\s*([\d\.]+)\s*\(\s*([\d\.]+)\s*%\s*\)"
    match = re.search(pattern, text_part)
    
    if match:
        try:
            num = float(match.group(1))
            denom = float(match.group(2))
            pct = float(match.group(3))
            return num, denom, pct
        except ValueError:
            return None # Indique un échec
    return None

def save_global_sats_to_csv(csv_folder_path, assistant_name, output_path):
    print(f"Génération du Global Summary (avec Écart-type)...")
    
    # 1. Récupération des fichiers
    pattern = os.path.join(csv_folder_path, f"summary_{assistant_name}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("Aucun fichier CSV trouvé.")
        return

    # Structure : data[Col][Metrique] = {'num': [], 'denom': [], 'pct': []}
    aggregated_data = {}
    headers = []
    valid_files_count = 0

    # 2. Extraction des données
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
            if len(rows) < 2: continue
            
            current_headers = rows[0]
            data_row = rows[1]

            # Initialisation des en-têtes au premier passage
            if not headers:
                headers = current_headers 

            # Vérification de la cohérence des colonnes
            if len(data_row) != len(current_headers):
                print(f"Attention: fichier ignoré (colonnes incohérentes) : {file_path}")
                continue

            valid_files_count += 1

            for idx, col_name in enumerate(current_headers):
                if idx == 0: continue # On saute la première colonne (Label)
                
                # Initialisation si nouvelle colonne
                if col_name not in aggregated_data:
                    aggregated_data[col_name] = {
                        'correct': {'num': [], 'denom': [], 'pct': []},
                        'score':   {'num': [], 'denom': [], 'pct': []},
                        'no_resp': {'num': [], 'denom': [], 'pct': []}
                    }
                
                cell_value = data_row[idx]
                parts = cell_value.split('|')
                
                if len(parts) == 3:
                    # Fonction helper pour éviter la répétition
                    def process_part(part_str, metric_key):
                        extracted = extract_full_data_regex(part_str)
                        if extracted:
                            n, d, p = extracted
                            aggregated_data[col_name][metric_key]['num'].append(n)
                            aggregated_data[col_name][metric_key]['denom'].append(d)
                            aggregated_data[col_name][metric_key]['pct'].append(p)

                    process_part(parts[0], 'correct')
                    process_part(parts[1], 'score')
                    process_part(parts[2], 'no_resp')

    # 3. Écriture du Global
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        
        # En-tête
        new_headers = list(headers)
        new_headers[0] = f"GLOBAL ({valid_files_count} runs)"
        writer.writerow(new_headers)
        
        # Titres des lignes
        row_correct = ["Correct / Total (Moy +/- SD)"]
        row_score   = ["Pts ok / Pts max (Moy +/- SD)"]
        row_no_resp = ["No response / Total (Moy +/- SD)"]
        
        for col_name in headers[1:]:
            col_data = aggregated_data.get(col_name)
            
            def format_cell(data_dict):
                # S'il n'y a pas assez de données
                if not data_dict or not data_dict['pct']: 
                    return "N/A"
                
                # Moyennes
                moy_num = statistics.mean(data_dict['num'])
                moy_denom = statistics.mean(data_dict['denom'])
                moy_pct = statistics.mean(data_dict['pct'])
                
                # Écart-type (Standard Deviation)
                # Nécessite au moins 2 valeurs
                if len(data_dict['pct']) > 1:
                    stdev_val = statistics.stdev(data_dict['pct'])
                else:
                    stdev_val = 0.0
                
                # Format: 10.5/12.0 (83.5%) +/- 2.1
                return f"{moy_num:.1f}/{moy_denom:.1f} ({moy_pct:.1f}%) ± {stdev_val:.1f}"

            row_correct.append(format_cell(col_data['correct']))
            row_score.append(format_cell(col_data['score']))
            row_no_resp.append(format_cell(col_data['no_resp']))
            
        writer.writerow(row_correct)
        writer.writerow(row_score)
        writer.writerow(row_no_resp)

    print(f"Tableau Global CSV généré : {output_path}")

async def main():
    # Chargement du dict de questions
    raw_questions = load_json(BENCHMARK_QUERYS_JSON)

    x=0
    while (x<10):
        CURRENT_TIME = datetime.now()
        print(f"\n--- BENCHMARK : {TESTED_ASSISTANT} ---")
        # Step 1 : On interroge Dimarc et on récup chaque réponse
        benchmark_results = await step_1_run_benchmark(raw_questions)

        print(f"\n--- ÉVALUATION ---")
        # Step 2 : On fait évaluer chaque réponse de Dimarc par un LLM
        final_report = await step_2_run_evaluation(benchmark_results)

        # Noms de fichiers basés sur l'heure
        file_time = CURRENT_TIME.strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde JSON
        path_json = f"{OUTPUT_PATH_JSON}benchmark_{TESTED_ASSISTANT}_{file_time}.json"
        save_json(final_report, path_json)

        # Calcul des stats
        stats = calculate_statistics(final_report)

        # Affichage Console
        print_markdown_table(stats, TESTED_ASSISTANT)
        
        # Sauvegarde CSV (Tableau récapitulatif)
        path_csv = f"{OUTPUT_PATH_CSV}summary_{TESTED_ASSISTANT}_{file_time}.csv"
        save_stats_to_csv(stats, TESTED_ASSISTANT, path_csv)

        x+=1

    print(f"\n--- FIN DU BENCHMARK ---")

    # Génération du CSV GLOBAL
    path_csv_global = f"{OUTPUT_PATH_CSV_GLOBAL}global_summary_{TESTED_ASSISTANT}.csv"

    # On appelle la fonction en lui donnant le DOSSIER où chercher les CSV (OUTPUT_PATH_CSV)
    # Sauvegarde CSV (Tableau récapitulatif GLOBAL : variable et moyenne)
    save_global_sats_to_csv(OUTPUT_PATH_CSV, TESTED_ASSISTANT, path_csv_global)



if __name__ == "__main__":
    asyncio.run(main())