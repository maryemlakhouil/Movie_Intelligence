"""
Pipeline complet : Extraction -> Nettoyage -> Features -> MongoDB -> NLP -> ML.

"""
import argparse
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # pas d'écran dans un conteneur

logger = logging.getLogger("pipeline")


def extract() -> None:
    from src.extraction.extract_tmdb import run_extraction
    ok, failed = run_extraction(int(os.getenv("N_MOVIES", "2000")))
    if ok == 0:
        raise RuntimeError("Extraction : aucun film téléchargé (token TMDB ? réseau ?)")
    if len(failed) > 0.10 * (ok + len(failed)):
        raise RuntimeError(f"Extraction : trop d'échecs ({len(failed)} sur {ok + len(failed)})")


def clean() -> None:
    from src.cleaning.clean_data import run_cleaning
    run_cleaning()


def features() -> None:
    from src.features.build_features import run_feature_engineering
    run_feature_engineering()


def mongo() -> None:
    from src.database.mongo_loader import load_to_mongo
    logger.info("Films en base : %d", load_to_mongo())


def tfidf() -> None:
    from src.nlp.tfidf import print_summary, run_tfidf
    print_summary(run_tfidf(include_keywords=os.getenv("TFIDF_KEYWORDS", "1") == "1"))


def classifier() -> None:
    from src.models.classification import THRESHOLD
    from src.models.validation import main
    main(THRESHOLD, None, 5)


def clustering() -> None:
    from src.models.clustering import main
    main()


# L'ordre du dictionnaire EST l'ordre d'exécution
STAGES = {
    "extract": extract, "clean": clean, "features": features, "mongo": mongo,
    "tfidf": tfidf, "classifier": classifier, "clustering": clustering,
}


def run_stage(name: str) -> None:
    if name not in STAGES:
        raise KeyError(f"Étape inconnue : {name!r} (choix : {', '.join(STAGES)})")
    try:
        STAGES[name]()
    except SystemExit as exc:  
        raise RuntimeError(f"Étape {name} : {exc}") from None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pipeline films TMDB")
    ap.add_argument("stages", nargs="*", help=f"étapes à lancer (défaut : toutes) : {', '.join(STAGES)}")
    names = ap.parse_args(argv).stages or list(STAGES)
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        ap.error(f"étape(s) inconnue(s) : {unknown}")
    for name in names:
        logger.info("========== %s ==========", name)
        run_stage(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
