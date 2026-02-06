import json
import asyncio
import aiohttp
import requests
import csv
import glob
import statistics
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from datetime import datetime

# Configuration
load_dotenv(find_dotenv())

# Récupération sécurisée
CLIENT_ID = os.getenv("DIMARC_CLIENT_ID")
CLIENT_SECRET = os.getenv("DIMARC_CLIENT_SECRET")
DIMARC_URL = os.getenv("DIMARC_URL")
DOCUMENTALIST_ID = os.getenv("DIMARC_AGENT_ID")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("Erreur: Les identifiants DIMARC sont manquants dans le fichier .env")

MODEL_JUDGE = "gpt-4o"
TESTED_ASSISTANT = "Dimarc"

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

def get_dimarc_token():
    try:
        url = f"{DIMARC_URL}/token"
        params = {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"Erreur Auth Dimarc: {e}")
        return None

# Étape 1 : Benchmark
async def ask_dimarc_one(session, agent_id, api_key, question_obj, semaphore):
    url = f"{DIMARC_URL}/documentalist/{agent_id}"
    headers = {'x-api-key': api_key}
    
    async with semaphore:
        try:
            print(f"Pose {question_obj.get('id')} ({TESTED_ASSISTANT})...")
            
            # --- TIMER START ---
            start_time = time.time() 
            
            async with session.post(url, headers=headers, json={"query": question_obj["query"]}) as response:
                full_text = ""
                async for chunk in response.content.iter_chunked(1024):
                    if chunk: full_text += chunk.decode('utf-8')
                
                # --- TIMER END ---
                end_time = time.time()
                duration = end_time - start_time
                
                question_obj["answer"] = full_text
                question_obj["response_time"] = duration # Stockage du temps
                
                return question_obj
        except Exception as e:
            question_obj["answer"] = f"Error: {str(e)}"
            question_obj["response_time"] = 0 # Valeur par défaut
            return question_obj

async def step_1_run_benchmark(token, questions):
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = [ask_dimarc_one(session, DOCUMENTALIST_ID, token, q, semaphore) for q in questions]
        results = await asyncio.gather(*tasks)
    return results

# Étape 2 : Évaluation
async def grade_one(client, question_data, system_prompt, semaphore):
    async with semaphore:
        print(f"Evaluation {question_data.get('id')} par ({MODEL_JUDGE})...")
        
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
                "response_time": question_data.get("response_time", 0), # On passe le temps
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
                "is_correct": False, 
                "points_awarded": 0,
                "response_time": question_data.get("response_time", 0),
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
    """Calcule les statistiques agrégées par type + le total + le temps."""
    stats_by_type = {}
    global_stats = {"correct": 0, "total": 0, "pts_ok": 0, "pts_max": 0, "No_response": 0, "total_time": 0.0}
    
    for item in evaluations:
        if item.get("global_summary"): continue
            
        q_type = item.get("type", "Autre")
        if q_type not in stats_by_type:
            stats_by_type[q_type] = {"correct": 0, "total": 0, "pts_ok": 0, "pts_max": 0, "No_response": 0, "total_time": 0.0}
        
        s = stats_by_type[q_type]
        
        # Récupération du temps
        r_time = item.get("response_time", 0)

        # Mise à jour locale
        s["total"] += 1
        s["pts_ok"] += item["points_awarded"]
        s["pts_max"] += item["query_points"]
        s["total_time"] += r_time

        # Compteur de non-réponses
        if item["answer"] == "" :
            s["No_response"] += 1
            global_stats["No_response"] += 1
        
        # Mise à jour globale
        global_stats["total"] += 1
        global_stats["pts_ok"] += item["points_awarded"]
        global_stats["pts_max"] += item["query_points"]
        global_stats["total_time"] += r_time
        
        if item["is_correct"]:
            s["correct"] += 1
            global_stats["correct"] += 1

    # On ajoute le TOTAL à la liste des types pour faciliter l'export
    stats_by_type["TOTAL"] = global_stats
    return stats_by_type

# AFFICHAGE MARKDOWN
def print_markdown_table(stats_by_type, assistant_name):
    sorted_types = sorted(stats_by_type.keys())
    if "TOTAL" in sorted_types:
        sorted_types.remove("TOTAL")
        sorted_types.append("TOTAL")
    
    # Largeurs des colonnes
    col_width = 25
    name_width = 25 
    
    # 1. Initialisation des lignes
    header = f"| {assistant_name:<{name_width}} |"
    sep_line = f"| {'-'*name_width} |"
    
    row_correct = f"| {'Correct / Total':<{name_width}} |"
    row_score   = f"| {'Pts ok / Pts max':<{name_width}} |"
    row_no_resp = f"| {'No response / Total':<{name_width}} |"
    row_time    = f"| {'Avg Response Time':<{name_width}} |" # Nouvelle ligne
    
    # 2. Boucle sur chaque type
    for t in sorted_types:
        s = stats_by_type[t]
        
        pct_c = (s['correct'] / s['total'] * 100) if s['total'] else 0
        pct_p = (s['pts_ok'] / s['pts_max'] * 100) if s['pts_max'] else 0
        pct_error = (s['No_response'] / s['total'] * 100) if s['total'] else 0
        avg_time = (s['total_time'] / s['total']) if s['total'] else 0
        
        header += f" {t:<{col_width}} |"
        sep_line += f" {'-'*col_width} |"

        val_c = f"{s['correct']}/{s['total']} ({pct_c:.0f}%)"
        val_p = f"{s['pts_ok']}/{s['pts_max']} ({pct_p:.0f}%)"
        val_e = f"{s['No_response']}/{s['total']} ({pct_error:.0f}%)"
        val_t = f"{avg_time:.2f}s"
        
        row_correct += f" {val_c:<{col_width}} |"
        row_score   += f" {val_p:<{col_width}} |"
        row_no_resp += f" {val_e:<{col_width}} |"
        row_time    += f" {val_t:<{col_width}} |"

    # 3. Affichage final
    full_sep = "-" * len(header)
    
    print("\n" + full_sep)
    print(header)
    print(full_sep)
    print(row_correct)
    print(row_score)
    print(row_no_resp)
    print(row_time) # Affichage du temps
    print(full_sep + "\n")

# EXPORT CSV UNITAIRE
def save_stats_to_csv(stats_by_type, assistant_name, output_path):
    keys = sorted([k for k in stats_by_type.keys() if k != "TOTAL"])
    keys.append("TOTAL")
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        
        headers = [assistant_name] + keys
        writer.writerow(headers)
        
        row = ["Pdf / word"]
        
        for t in keys:
            s = stats_by_type[t]
            pct_c = (s['correct']/s['total']*100) if s['total'] else 0
            pct_p = (s['pts_ok']/s['pts_max']*100) if s['pts_max'] else 0
            pct_error = (s['No_response']/s['total']*100) if s['total'] else 0
            avg_time = (s['total_time'] / s['total']) if s['total'] else 0
            
            # Format: "Correct | Score | NoResp | Time"
            cell_value = f"{s['correct']}/{s['total']} ({pct_c:.0f}%) | {s['pts_ok']}/{s['pts_max']} ({pct_p:.0f}%) | {s['No_response']}/{s['total']} ({pct_error:.0f}%) | {avg_time:.2f}s"
            row.append(cell_value)
            
        writer.writerow(row)
    
    print(f"Tableau CSV exporté : {output_path}")

def extract_full_data_regex(text_part):
    """Extrait num, denom, pct via Regex."""
    pattern = r"([\d\.]+)\s*/\s*([\d\.]+)\s*\(\s*([\d\.]+)\s*%\s*\)"
    match = re.search(pattern, text_part)
    if match:
        try:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        except ValueError:
            return None
    return None

# EXPORT CSV GLOBAL
def save_global_sats_to_csv(csv_folder_path, assistant_name, output_path):
    print(f"Génération du Global Summary (avec Écart-type & Temps)...")
    
    pattern = os.path.join(csv_folder_path, f"summary_{assistant_name}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("Aucun fichier CSV trouvé.")
        return

    aggregated_data = {}
    headers = []
    valid_files_count = 0

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            rows = list(reader)
            if len(rows) < 2: continue
            
            current_headers = rows[0]
            data_row = rows[1]

            if not headers: headers = current_headers 
            if len(data_row) != len(current_headers): continue

            valid_files_count += 1

            for idx, col_name in enumerate(current_headers):
                if idx == 0: continue 
                
                if col_name not in aggregated_data:
                    aggregated_data[col_name] = {
                        'correct': {'num': [], 'denom': [], 'pct': []},
                        'score':   {'num': [], 'denom': [], 'pct': []},
                        'no_resp': {'num': [], 'denom': [], 'pct': []},
                        'time': [] 
                    }
                
                cell_value = data_row[idx]
                parts = cell_value.split('|')
                
                # Parties standards
                if len(parts) >= 3:
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
                
                # Partie Temps (4ème)
                if len(parts) >= 4:
                    try:
                        time_str = parts[3].replace('s', '').strip()
                        aggregated_data[col_name]['time'].append(float(time_str))
                    except ValueError:
                        pass

    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        
        new_headers = list(headers)
        new_headers[0] = f"GLOBAL ({valid_files_count} runs)"
        writer.writerow(new_headers)
        
        row_correct = ["Correct / Total (Moy +/- SD)"]
        row_score   = ["Pts ok / Pts max (Moy +/- SD)"]
        row_no_resp = ["No response / Total (Moy +/- SD)"]
        row_time    = ["Temps de réponse moyen (Moy +/- SD)"] 
        
        for col_name in headers[1:]:
            col_data = aggregated_data.get(col_name)
            
            def format_complex_cell(data_dict):
                if not data_dict or not data_dict['pct']: return "N/A"
                moy_num = statistics.mean(data_dict['num'])
                moy_denom = statistics.mean(data_dict['denom'])
                moy_pct = statistics.mean(data_dict['pct'])
                stdev_val = statistics.stdev(data_dict['pct']) if len(data_dict['pct']) > 1 else 0.0
                return f"{moy_num:.1f}/{moy_denom:.1f} ({moy_pct:.1f}%) ± {stdev_val:.1f}"

            def format_time_cell(time_list):
                if not time_list: return "N/A"
                moy_time = statistics.mean(time_list)
                stdev_time = statistics.stdev(time_list) if len(time_list) > 1 else 0.0
                return f"{moy_time:.2f}s ± {stdev_time:.2f}"

            row_correct.append(format_complex_cell(col_data['correct']))
            row_score.append(format_complex_cell(col_data['score']))
            row_no_resp.append(format_complex_cell(col_data['no_resp']))
            row_time.append(format_time_cell(col_data['time']))
            
        writer.writerow(row_correct)
        writer.writerow(row_score)
        writer.writerow(row_no_resp)
        writer.writerow(row_time)

    print(f"Tableau Global CSV généré : {output_path}")

async def main():
    token = get_dimarc_token()
    raw_questions = load_json(BENCHMARK_QUERYS_JSON)

    if not token or not raw_questions:
        print("Arrêt : Token manquant ou fichier vide.")
        return
    
    x=0
    while (x<10):
        CURRENT_TIME = datetime.now()
        print(f"\n--- BENCHMARK : {TESTED_ASSISTANT} (Run {x+1}/10) ---")
        
        # Step 1
        benchmark_results = await step_1_run_benchmark(token, raw_questions)

        print(f"\n--- ÉVALUATION ---")
        # Step 2
        final_report = await step_2_run_evaluation(benchmark_results)

        file_time = CURRENT_TIME.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        path_json = f"{OUTPUT_PATH_JSON}benchmark_{TESTED_ASSISTANT}_{file_time}.json"
        save_json(final_report, path_json)

        # Calcul Stats
        stats = calculate_statistics(final_report)

        # Affichage Console
        print_markdown_table(stats, TESTED_ASSISTANT)
        
        # Save CSV Unit
        path_csv = f"{OUTPUT_PATH_CSV}summary_{TESTED_ASSISTANT}_{file_time}.csv"
        save_stats_to_csv(stats, TESTED_ASSISTANT, path_csv)

        x+=1

    print(f"\n--- FIN DU BENCHMARK ---")

    # CSV GLOBAL
    path_csv_global = f"{OUTPUT_PATH_CSV_GLOBAL}global_summary_{TESTED_ASSISTANT}.csv"
    save_global_sats_to_csv(OUTPUT_PATH_CSV, TESTED_ASSISTANT, path_csv_global)

if __name__ == "__main__":
    asyncio.run(main())