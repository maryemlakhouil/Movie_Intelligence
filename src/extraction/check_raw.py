import json
from pathlib import Path

# Contrôle qualité des JSON bruts 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
files = list((PROJECT_ROOT / "data" / "raw" / "movies").glob("*.json"))  #glob Trouver tous les fichiers .json.

n = len(files)
print(f"Films téléchargés : {n}")
if n == 0:
    raise SystemExit("Aucun fichier : lancer d'abord extract_tmdb.py")

# compteurs 
zero_budget = zero_revenue = no_overview = no_keywords = no_runtime = no_date = 0

for f in files:
    m = json.loads(f.read_text(encoding="utf-8")) # lire et transformer en dict 
    zero_budget += m.get("budget", 0) == 0    # si budget n'existe pas, utiliser 0 true or false en trf on 0 ou 1
    zero_revenue += m.get("revenue", 0) == 0
    no_overview += not m.get("overview")
    no_keywords += not m.get("keywords", {}).get("keywords")
    no_runtime += not m.get("runtime")
    no_date += not m.get("release_date")

for label, v in [("budget = 0", zero_budget), ("revenue = 0", zero_revenue),
                 ("overview vide", no_overview), ("sans mots-clés", no_keywords),
                 ("runtime nul/0", no_runtime), ("sans date", no_date)]:
    print(f"{label:16s}: {v:5d} ({v / n:.1%})")    # Réserve 16 caractères. 
