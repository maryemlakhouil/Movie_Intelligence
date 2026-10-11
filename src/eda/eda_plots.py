import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.cleaning.clean_data import NUMERIC_COLS, load_clean, missing_summary  # noqa: E402

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
INTERP_MD = PROJECT_ROOT / "reports" / "eda_interpretations.md"
BLUE, ORANGE = "#3b6ea5", "#e08e45"

sns.set_theme(style="whitegrid", context="notebook")


# Utilitaires

def _finish(fig, name: str, show: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.show() if show else plt.close(fig)


def _skew_word(s: pd.Series) -> str:
    sk = s.dropna().skew()
    if sk > 1:
        return "très asymétrique vers la droite (longue queue de valeurs élevées)"
    if sk > 0.5:
        return "asymétrique vers la droite"
    if sk < -1:
        return "très asymétrique vers la gauche (longue queue de valeurs faibles)"
    if sk < -0.5:
        return "asymétrique vers la gauche"
    return "à peu près symétrique"


def _pct(x: float) -> str:
    return f"{x:.0f} %"


def _exploded_genres(df: pd.DataFrame) -> pd.DataFrame:
    return df.explode("genres").dropna(subset=["genres"])


# Graphiques

def plot_missing(df, show=False):
    miss = missing_summary(df)["pourcentage"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(miss.index, miss.values, color=BLUE)
    ax.set_xlabel("% de valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne")
    _finish(fig, "01_missing_values", show)
    high = miss[miss > 10].sort_values(ascending=False)
    if high.empty:
        return "Aucune colonne ne dépasse 10 % de valeurs manquantes : les données sont globalement complètes."
    cols = ", ".join(f"{c} ({v:.0f} %)" for c, v in high.items())
    return (f"Colonnes avec plus de 10 % de manquants : {cols}. Ces valeurs sont des informations inconnues "
            "de TMDB, pas des zéros : elles seront imputées dans un Pipeline (sur le train uniquement) "
            "et signalées par l'indicateur `has_budget`.")


def plot_vote_average(df, show=False):
    s = df["vote_average"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(s, bins=30, kde=True, color=BLUE, ax=ax)
    ax.axvline(s.mean(), color="red", ls="--", label=f"moyenne = {s.mean():.2f}")
    ax.axvline(s.median(), color="green", ls=":", label=f"médiane = {s.median():.2f}")
    ax.set(title="Distribution des notes (vote_average)", xlabel="Note moyenne (0-10)", ylabel="Nombre de films")
    ax.legend()
    _finish(fig, "02_vote_average_distribution", show)
    inside = ((s >= 6) & (s <= 8)).mean() * 100
    return (f"La note moyenne est {s.mean():.2f} (médiane {s.median():.2f}, écart-type {s.std():.2f}). "
            f"La distribution est {_skew_word(s)} et {_pct(inside)} des films ont une note entre 6 et 8"
            + (" : les notes extrêmes sont rares, la note seule discrimine peu les films." if inside >= 60 else "."))


def plot_popularity(df, show=False):
    s = df["popularity"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(s, bins=40, color=ORANGE, ax=axes[0])
    axes[0].set(title="Popularité (échelle linéaire)", xlabel="popularity", ylabel="Nombre de films")
    sns.histplot(np.log10(s.clip(lower=0.01)), bins=40, color=ORANGE, ax=axes[1])
    axes[1].set(title="Popularité (log10)", xlabel="log10(popularity)", ylabel="")
    _finish(fig, "03_popularity_distribution", show)
    return (f"La popularité est {_skew_word(s)} : médiane {s.median():.1f} contre un maximum de {s.max():.0f}. "
            "Quelques films concentrent l'essentiel de la popularité, ce qui écrase le graphique linéaire. "
            "L'échelle logarithmique (à droite) rend la distribution lisible : une transformation log sera utile.")


def plot_vote_count(df, show=False):
    s = df["vote_count"].dropna()
    q = s.quantile([0.5, 0.75, 0.9])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log10(s.clip(lower=1)), bins=40, color=BLUE, ax=ax)
    for p, color in zip(q.index, ["green", "orange", "red"]):
        ax.axvline(np.log10(max(q[p], 1)), color=color, ls="--", label=f"{int(p * 100)}e centile = {q[p]:.0f} votes")
    ax.set(title="Distribution du nombre de votes (log10)", xlabel="log10(vote_count)", ylabel="Nombre de films")
    ax.legend()
    _finish(fig, "04_vote_count_distribution", show)
    return (f"Le nombre de votes est {_skew_word(s)} : médiane {q[0.5]:.0f}, 75e centile {q[0.75]:.0f}, "
            f"90e centile {q[0.9]:.0f}, maximum {s.max():.0f}. Cette distribution guidera le choix du seuil de "
            "`high_engagement` au Jour 5 : la médiane donne des classes équilibrées, un centile plus élevé "
            "isole les films vraiment à fort engagement (classes déséquilibrées).")


def plot_genres(df, show=False):
    g = _exploded_genres(df)["genres"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=g.values, y=g.index, color=BLUE, ax=ax)
    ax.set(title="Nombre de films par genre", xlabel="Nombre de films", ylabel="")
    _finish(fig, "05_films_by_genre", show)
    top = g.head(3)
    share = top / len(df) * 100
    names = ", ".join(f"{k} ({v} films, {share[k]:.0f} % du catalogue)" for k, v in top.items())
    return (f"Les genres les plus présents sont {names}. Un film pouvant avoir plusieurs genres, "
            f"la somme des barres dépasse le nombre de films ({len(df)}). "
            f"Le genre le plus rare est {g.index[-1]} ({g.iloc[-1]} films) : les déséquilibres entre genres "
            "peuvent influencer les clusters et les modèles.")


def plot_years(df, show=False):
    y = df["release_date"].dt.year.dropna().astype(int)
    counts = y.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(counts.index, counts.values, color=BLUE, width=0.85)
    axes[0].set(title="Sorties par année", xlabel="Année", ylabel="Nombre de films")
    dec = (y // 10 * 10).value_counts().sort_index()
    axes[1].bar(dec.index.astype(str) + "s", dec.values, color=ORANGE)
    axes[1].set(title="Sorties par décennie", xlabel="Décennie", ylabel="")
    axes[1].tick_params(axis="x", rotation=45)
    _finish(fig, "06_releases_by_year", show)
    recent = (y >= 2000).mean() * 100
    return (f"Le catalogue couvre de {y.min()} à {y.max()}. L'année la plus représentée est {counts.idxmax()} "
            f"({counts.max()} films) et {_pct(recent)} des films sont sortis depuis 2000. "
            + ("Le catalogue est donc fortement orienté vers les films récents (effet du tri par popularité à "
               "l'extraction), biais à mentionner : les conclusions valent surtout pour le cinéma récent."
               if recent >= 60 else "Les périodes sont relativement équilibrées."))


def plot_runtime(df, show=False):
    s = df["runtime"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    sns.histplot(s, bins=40, color=BLUE, ax=axes[0])
    axes[0].axvline(s.median(), color="red", ls="--", label=f"médiane = {s.median():.0f} min")
    axes[0].set(title="Distribution de la durée", xlabel="Durée (minutes)", ylabel="Nombre de films")
    axes[0].legend()
    sns.boxplot(y=s, color=ORANGE, ax=axes[1])
    axes[1].set(title="Boxplot de la durée", ylabel="Durée (minutes)")
    _finish(fig, "07_runtime", show)
    q1, q3 = s.quantile([0.25, 0.75])
    n_out = int((s > q3 + 1.5 * (q3 - q1)).sum() + (s < q1 - 1.5 * (q3 - q1)).sum())
    return (f"La durée médiane est de {s.median():.0f} minutes, la moitié des films durant entre {q1:.0f} et {q3:.0f} min. "
            f"La distribution est {_skew_word(s)}. Le boxplot signale {n_out} valeurs aberrantes (durées très courtes ou "
            f"très longues, maximum {s.max():.0f} min) : ce sont de vrais films atypiques, on les conserve, "
            "et les seuils fixes de `runtime_category` limitent leur influence.")


def plot_budget_revenue(df, show=False):
    d = df.dropna(subset=["budget", "revenue"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d["budget"], d["revenue"], alpha=0.4, color=BLUE, s=18)
    lim = [max(min(d["budget"].min(), d["revenue"].min()), 1), max(d["budget"].max(), d["revenue"].max())]
    ax.plot(lim, lim, "r--", label="revenue = budget (seuil de rentabilité brute)")
    ax.set(xscale="log", yscale="log", xlabel="Budget ($, log)", ylabel="Revenus ($, log)",
           title="Budget vs revenus")
    ax.legend(loc="upper left")
    _finish(fig, "08_budget_vs_revenue", show)
    rho = d["budget"].corr(d["revenue"], method="spearman")
    above = (d["revenue"] > d["budget"]).mean() * 100
    return (f"Sur les {len(d)} films dont budget et revenus sont connus ({len(d) / len(df) * 100:.0f} % du catalogue), "
            f"la corrélation de Spearman est de {rho:.2f}"
            + (" : plus le budget est élevé, plus les revenus tendent à l'être. " if rho >= 0.3 else " : le lien est faible. ")
            + f"{_pct(above)} des films sont au-dessus de la droite revenue = budget. Observez la dispersion verticale : "
            "à budget égal, les revenus varient fortement. Les films sans budget ou revenus connus sont exclus du graphique.")


def plot_votes_popularity(df, show=False):
    d = df.dropna(subset=["vote_count", "popularity"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d["vote_count"].clip(lower=1), d["popularity"].clip(lower=0.01), alpha=0.4, color=ORANGE, s=18)
    ax.set(xscale="log", yscale="log", xlabel="Nombre de votes (log)", ylabel="Popularité (log)",
           title="Votes vs popularité")
    _finish(fig, "09_votes_vs_popularity", show)
    rho = d["vote_count"].corr(d["popularity"], method="spearman")
    strength = "forte" if abs(rho) >= 0.6 else "modérée" if abs(rho) >= 0.3 else "faible"
    return (f"La corrélation de Spearman entre votes et popularité est {rho:.2f} ({strength}). "
            "Les deux mesures capturent en partie la même chose (l'attention du public). "
            "Point crucial pour la classification : si la cible est construite à partir de `vote_count`, "
            "`popularity` en est un proxy et pourrait provoquer un Data Leakage : nous l'écartons des features.")


def plot_boxplots(df, show=False):
    d = df.dropna(subset=["release_date"]).copy()
    d["decade"] = (d["release_date"].dt.year // 10 * 10).astype(int)
    dec_ok = d["decade"].value_counts()
    dec_ok = dec_ok[dec_ok >= 10].index
    d1 = d[d["decade"].isin(dec_ok)].sort_values("decade")
    d1["decade"] = d1["decade"].astype(str) + "s"

    g = _exploded_genres(df)
    top = g["genres"].value_counts().head(8).index
    g = g[g["genres"].isin(top)].copy()
    g["log_votes"] = np.log10(g["vote_count"].clip(lower=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=d1, x="decade", y="vote_average", color=BLUE, ax=axes[0])
    axes[0].set(title="Note par décennie", xlabel="Décennie", ylabel="vote_average")
    axes[0].tick_params(axis="x", rotation=45)
    order = g.groupby("genres")["log_votes"].median().sort_values(ascending=False).index
    sns.boxplot(data=g, x="genres", y="log_votes", order=order, color=ORANGE, ax=axes[1])
    axes[1].set(title="Nombre de votes (log10) par genre", xlabel="", ylabel="log10(vote_count)")
    axes[1].tick_params(axis="x", rotation=45)
    _finish(fig, "10_boxplots", show)

    med_dec = d1.groupby("decade")["vote_average"].median()
    med_gen = g.groupby("genres")["log_votes"].median().sort_values(ascending=False)
    txt = (f"Note médiane par décennie : la plus élevée pour les {med_dec.idxmax()} ({med_dec.max():.2f}), "
           f"la plus basse pour les {med_dec.idxmin()} ({med_dec.min():.2f}). ")
    if int(med_dec.idxmax()[:4]) < 2000:
        txt += ("Les films anciens encore présents dans le catalogue sont probablement des classiques : "
                "biais de survie à mentionner. ")
    ratio = 10 ** (med_gen.iloc[0] - med_gen.iloc[-1])
    txt += (f"Côté engagement, le genre avec le plus de votes en médiane est {med_gen.index[0]} "
            f"(≈ {10 ** med_gen.iloc[0]:.0f} votes) et le moins engageant {med_gen.index[-1]} "
            f"(≈ {10 ** med_gen.iloc[-1]:.0f} votes), soit un rapport de {ratio:.1f}. ")
    txt += ("Le genre semble donc lié à l'engagement et sera une feature utile."
            if ratio >= 1.5 else "L'écart entre genres est modeste : le genre seul est un faible prédicteur.")
    return txt


def plot_correlation(df, show=False):
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Corrélations de Spearman (variables numériques)")
    _finish(fig, "11_correlation_heatmap", show)
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(3)
    txt = ", ".join(f"{a} / {b} ({v:.2f})" for (a, b), v in top.items())
    return (f"Les paires les plus corrélées sont : {txt}. On utilise Spearman car les variables sont très asymétriques "
            "(Pearson serait dominé par les valeurs extrêmes). Les variables fortement corrélées entre elles "
            "sont redondantes pour les modèles linéaires, et celles liées à `vote_count` sont des sources potentielles de leakage.")


PLOTS = [
    ("Valeurs manquantes", plot_missing),
    ("Distribution des notes", plot_vote_average),
    ("Distribution de la popularité", plot_popularity),
    ("Distribution du nombre de votes", plot_vote_count),
    ("Films par genre", plot_genres),
    ("Sorties par année", plot_years),
    ("Durée", plot_runtime),
    ("Budget vs revenus", plot_budget_revenue),
    ("Votes vs popularité", plot_votes_popularity),
    ("Boxplots", plot_boxplots),
    ("Corrélations", plot_correlation),
]


def run_eda(df: pd.DataFrame | None = None, show: bool = False) -> dict:
    df = load_clean() if df is None else df
    results = {}
    for title, fn in PLOTS:
        results[title] = fn(df, show=show)
    lines = ["# Interprétations de l'EDA", "",
             f"Jeu de données : {len(df)} films. Textes générés à partir des chiffres réels ; à relire et reformuler.", ""]
    for i, (title, text) in enumerate(results.items()):
        lines += [f"## {title}", "", text, ""]
    INTERP_MD.parent.mkdir(parents=True, exist_ok=True)
    INTERP_MD.write_text("\n".join(lines), encoding="utf-8")
    return results


if __name__ == "__main__":
    plt.switch_backend("Agg")
    res = run_eda()
    for k, v in res.items():
        print(f"\n[{k}]\n{v}")
    print(f"\nFigures : {FIG_DIR}\nTextes  : {INTERP_MD}")
