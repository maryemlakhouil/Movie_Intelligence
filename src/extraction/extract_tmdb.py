"""
Usage :
    python src/extraction/extract_tmdb.py --n-movies 2000
    argparse : t9dar tjib parmetre de donnes ghir terminal  exemple --movies : 2000 
    logging afficher les messages de erreur ,information 
"""
import json
import logging
import os
import argparse
import time 
from pathlib import Path
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter # connection http 
from urllib3.util.retry import Retry    # Configurer les tentatives automatiques si une requête échoue


# Racine du projet (movie-recommender/), quel que soit le dossier de lancement
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")  

BASE_URL = "https://api.themoviedb.org/3"
RAW_DIR = Path(os.getenv("RAW_DIR", PROJECT_ROOT / "data" / "raw"))
DISCOVER_DIR = RAW_DIR / "discover"
MOVIES_DIR = RAW_DIR / "movies"
MAX_PAGES = 500  # limite imposée par TMDB sur /discover

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tmdb")


# Client HTTP : un outil personnalisé pour communiquer avec l'API TMDB.
# Une Session permet de garder une connexion commune pour plusieurs requêtes. 

class TMDBClient:

    def __init__(self, delay: float = 0.05, timeout: int = 10):
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.params = {}

        token = os.getenv("TMDB_READ_TOKEN")
        api_key = os.getenv("TMDB_API_KEY")
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif api_key:
            self.params["api_key"] = api_key
        else:
            raise EnvironmentError("Définir TMDB_READ_TOKEN ou TMDB_API_KEY dans le fichier .env")

        # Configuration des retries 
        # Trop de requêtes , erreur serveur ,mauvais reponce , serice indisponible , Timeout  serveur/proxy
        retry = Retry(
            total=5,
            backoff_factor=1,   #kola mara katsana secende 1 zayda 1s → 2s → 4s → ...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,  # Attendez 10 secondes avant de réessayer. rescept 
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # GET avec gestion d'erreurs. Retourne un dict, ou None en cas d'échec/vide 

    def get(self, endpoint: str, params: dict | None = None) -> dict | None:
        # throttling ou limitation de debit
        time.sleep(self.delay)  # attent 0.05 Pour éviter d'envoyer trop de requêtes rapidement à TMDB
        all_params = {**self.params, **(params or {})} # fusiennr 2 dict 
        url = f"{BASE_URL}{endpoint}"

        try:
            resp = self.session.get(url, params=all_params, timeout=self.timeout)
        except requests.RequestException as exc:   # probleme internet , connecion 
            logger.error("Erreur réseau sur %s : %s", endpoint, exc)
            return None

        if resp.status_code in (401, 403): # non autorisee 
            raise PermissionError("Authentification refusée : vérifier la clé/token TMDB")
        if resp.status_code == 404:
            logger.warning("404 introuvable : %s", endpoint)
            return None
        if not resp.ok:
            logger.error("HTTP %s sur %s", resp.status_code, endpoint)
            return None

        try:
            data = resp.json()
        except ValueError:
            logger.error("Réponse non JSON sur %s", endpoint)
            return None

        if not data:
            logger.warning("Réponse vide sur %s", endpoint)
            return None
        return data


#sauvegarder un dictionnaire Python dans un fichier JSON

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)  # json plus lisible 

# fichier -> disc 

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Phase A : récupérer les IDs des films depuis TMDB 

def discover_movie_ids(client: TMDBClient, n_movies: int) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set() #les doublons.
    page = 1

    while len(ids) < n_movies and page <= MAX_PAGES:
        path = DISCOVER_DIR / f"page_{page:04d}.json"

        if path.exists():  # cache local : pas d'appel API
            data = load_json(path)
        else:
            data = client.get(
                "/discover/movie",
                {
                    "page": page,
                    "sort_by": "popularity.desc",
                    "vote_count.gte": 10,
                    "include_adult": "false",
                    "language": "en-US",
                },
            )
            if data:
                save_json(data, path)

        if not data or not data.get("results"):
            logger.warning("Page %d vide ou en erreur : arrêt de la pagination", page)
            break

        for movie in data["results"]:
            if movie["id"] not in seen:  # évite les doublons
                seen.add(movie["id"])
                ids.append(movie["id"])

        page += 1

    logger.info("%d IDs collectés (%d pages)", len(ids), page - 1)
    return ids[:n_movies]


# Phase B : détails de chaque film

def fetch_movie_details(client: TMDBClient, movie_id: int) -> bool:

    path = MOVIES_DIR / f"{movie_id}.json"
    if path.exists():
        return True  

    data = client.get(
        f"/movie/{movie_id}",
        {"append_to_response": "keywords", "language": "en-US"},
    )
    if data:
        save_json(data, path)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction TMDB")
    parser.add_argument("--n-movies", type=int, default=2000)
    args = parser.parse_args()

    client = TMDBClient()
    ids = discover_movie_ids(client, args.n_movies)
    #Compteurs 
    ok, failed = 0, []
    for i, movie_id in enumerate(ids, start=1):
        if fetch_movie_details(client, movie_id):
            ok += 1
        else:
            failed.append(movie_id)
        if i % 100 == 0:
            logger.info("Progression : %d/%d", i, len(ids)) # 100/2000 

    logger.info("Terminé : %d films OK, %d échecs", ok, len(failed))
    if failed:
        logger.warning("IDs en échec : %s", failed)


if __name__ == "__main__":
    main()
