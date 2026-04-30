# -*- coding: utf-8 -*-
"""
Analyse de plusieurs séries temporelles financières
===================================================

Ce script peut être exécuté tel quel (python timeseries_financial_analysis.py)
ou copié par cellules dans un notebook Jupyter.

Hypothèses sur les données :
- Première colonne : dates
- Colonnes suivantes : prix, NAV ou indices (une colonne = une série)

Les statistiques de performance (Sharpe, volatilité, rendement annualisé, etc.)
sont calculées à partir des **rendements journaliers** (voir docstrings).

Auteur : généré pour un usage pédagogique — à adapter librement.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paramètres globaux (faciles à modifier)
# ---------------------------------------------------------------------------

# Nombre de jours de bourse par an (marchés US / Europe courants)
TRADING_DAYS_PER_YEAR = 252

# Taux sans risque annualisé (ex. 0.02 = 2 % par an). Utilisé pour le Sharpe.
# Par défaut : 0 (comme demandé).
RISK_FREE_ANNUAL = 0.0

# Chemin du CSV réel (None = on génère des données fictives)
CSV_PATH: Optional[str] = None  # ex. : r"C:\chemin\vers\donnees.csv"

# Export des tableaux (None = pas d'export)
EXPORT_DIR: Optional[str] = None  # ex. : r"C:\chemin\vers\exports"


# =============================================================================
# 1. PRÉPARATION DES DONNÉES
# =============================================================================


def load_or_build_prices(
    csv_path: Optional[str] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Charge un CSV ou construit un jeu de données fictif pour tester.

    Retourne un DataFrame avec :
    - un index Datetime trié
    - uniquement des colonnes numériques (les séries)

    La première colonne du fichier est supposée être la date (nom par défaut
    'date' si besoin de renommer — adapter selon votre fichier).
    """
    if csv_path is not None and pathlib.Path(csv_path).is_file():
        df = pd.read_csv(csv_path)
        # Si la première colonne n'a pas de nom "date", on prend la 1ère colonne
        if date_column not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: date_column})
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
        numeric = df.select_dtypes(include=[np.number])
        return numeric

    # --- Données fictives : 3 séries corrélées sur ~2 ans de jours ouvrés ---
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2022-01-01", periods=520, freq="B")
    n = len(dates)
    # Bruit commun + spécifique pour chaque actif
    common = rng.normal(0.0003, 0.008, size=n)
    rets = np.column_stack(
        [
            common + rng.normal(0.0002, 0.006, size=n),
            common * 0.7 + rng.normal(0.0001, 0.010, size=n),
            rng.normal(0.0004, 0.012, size=n),
        ]
    )
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(
        {"Serie_A": prices[:, 0], "Serie_B": prices[:, 1], "Serie_C": prices[:, 2]},
        index=dates,
    )


def prepare_price_frame(
    df: pd.DataFrame,
    *,
    fill_method: str = "ffill",
    drop_remaining_na: bool = True,
) -> pd.DataFrame:
    """
    Nettoie un DataFrame de prix déjà indexé par datetime.

    - fill_method : 'ffill' (propager la dernière valeur) ou 'bfill' ou None
    - drop_remaining_na : supprime les lignes où il reste des NaN après fill

    Vérifie que les colonnes sont numériques exploitables.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index doit être un DatetimeIndex.")

    out = df.sort_index().copy()
    # On ne garde que les colonnes numériques
    out = out.select_dtypes(include=[np.number])
    if out.empty:
        raise ValueError("Aucune colonne numérique trouvée.")

    if fill_method == "ffill":
        out = out.ffill()
    elif fill_method == "bfill":
        out = out.bfill()
    elif fill_method is not None:
        raise ValueError("fill_method doit être 'ffill', 'bfill' ou None.")

    if drop_remaining_na:
        out = out.dropna(how="any")

    if out.empty:
        raise ValueError("Après nettoyage, le DataFrame est vide.")

    return out


# =============================================================================
# 2. GRAPHIQUE EN BASE 100
# =============================================================================


def rebase_to_100(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rebaser chaque colonne à 100 sur sa **première valeur non-NaN**.

    Formule : 100 * prix_t / prix_premier_observation
    """
    base = prices.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return 100.0 * prices.div(base, axis=1)


def plot_base100(
    rebased: pd.DataFrame,
    title: str = "Évolution en base 100",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Trace toutes les séries rebasées sur un même graphique.

    - save_path : si renseigné, enregistre l'image (pratique en exécution .py sans GUI).
    - show : affiche la fenêtre matplotlib (typique en notebook / IDE interactif).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in rebased.columns:
        ax.plot(rebased.index, rebased[col], label=col, linewidth=1.5)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Indice (base 100 au premier point)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graphique enregistré : {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# 3. RENDEMENTS MENSUELS (tableau)
# =============================================================================


def monthly_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements mensuels simples entre **derniers cours disponibles** de chaque mois.

    - Les mois sont en lignes (index de type fin de mois / Period M)
    - Les séries restent en colonnes

    Méthode : dernier cours du mois (resample 'ME' = month end), puis pct_change.
    """
    month_end = prices.resample("ME").last()
    monthly_rets = month_end.pct_change()
    # Index plus lisible : année-mois
    monthly_rets.index = monthly_rets.index.to_period("M").astype(str)
    return monthly_rets.iloc[1:]  # première ligne NaN (pas de mois précédent)


def format_monthly_returns_for_display(
    monthly_rets: pd.DataFrame,
    pct_decimals: int = 2,
) -> pd.DataFrame:
    """Version affichage : pourcentages arrondis (copie, ne modifie pas les données brutes)."""
    return (monthly_rets * 100).round(pct_decimals)


def monthly_performance_table(
    monthly_rets: pd.DataFrame,
    *,
    series_name: Optional[str] = None,
    pct_decimals: int = 2,
) -> pd.DataFrame:
    """
    Construit un tableau annuel des performances mensuelles :
    - colonnes : Jan, Feb, ..., Dec, Year
    - lignes : annees

    monthly_rets est attendu avec un index mensuel (PeriodIndex "M" ou string "YYYY-MM")
    et une colonne par serie. Le tableau est en pourcentage pour l'affichage.
    """
    if monthly_rets.empty:
        return pd.DataFrame(columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"])

    data = monthly_rets.copy()
    if isinstance(data.index, pd.PeriodIndex):
        period_idx = data.index.asfreq("M")
    else:
        period_idx = pd.PeriodIndex(data.index.astype(str), freq="M")

    if series_name is None:
        series_name = str(data.columns[0])
    if series_name not in data.columns:
        raise KeyError(f"Serie introuvable dans monthly_rets : {series_name}")

    s = data[series_name].copy()
    s.index = period_idx
    work = pd.DataFrame(
        {
            "year": s.index.year,
            "month": s.index.month,
            "ret": s.values,
        }
    )

    pivot = work.pivot(index="year", columns="month", values="ret")
    month_order = list(range(1, 13))
    pivot = pivot.reindex(columns=month_order)
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    annual = work.groupby("year")["ret"].apply(lambda x: (1.0 + x).prod() - 1.0)
    table = pivot.join(annual.rename("Year"))

    return (table * 100).round(pct_decimals)


# =============================================================================
# 4. STATISTIQUES DE PERFORMANCE (à partir des rendements **journaliers**)
# =============================================================================


def daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements journaliers simples : (P_t / P_{t-1}) - 1

    **Important (pédagogique)** : les statistiques annualisées ci-dessous
    (volatilité, Sharpe, rendement annualisé géométrique) utilisent ces
    rendements **journaliers**, puis un facteur sqrt(252) ou 252 selon le cas.
    Le tableau des performances **mensuelles** est séparé (fonction dédiée).
    """
    return prices.pct_change().dropna(how="all")


def annualized_geometric_return(daily: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Performance annualisée **géométrique** à partir des rendements journaliers :

    ( ∏(1+r_i) )^(252/n) - 1

    où n = nombre d'observations journalières par série.
    """
    n = daily.shape[0]
    if n == 0:
        return pd.Series(dtype=float)
    wealth = (1 + daily).prod(axis=0)
    return wealth ** (trading_days / n) - 1


def annualized_volatility(daily: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Volatilité annualisée : écart-type des rendements journaliers * sqrt(252)."""
    return daily.std(axis=0, ddof=1) * np.sqrt(trading_days)


def sharpe_ratio(
    daily: pd.DataFrame,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Sharpe annualisé (rendements journaliers, taux sans risque annualisé converti en journalier).

    Sharpe = (E[r_jour] - rf_jour) / σ_jour * sqrt(252)

    avec rf_jour ≈ (1+rf_annual)^(1/252) - 1 (composition), ou approximation rf_annual/252.
    Ici on utilise la **composition** pour rester cohérent avec de petits taux.
    """
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / trading_days) - 1.0
    excess = daily - rf_daily
    mu = excess.mean(axis=0)
    sigma = daily.std(axis=0, ddof=1)
    # Éviter division par zéro
    sharpe = np.where(sigma > 0, mu / sigma * np.sqrt(trading_days), np.nan)
    return pd.Series(sharpe, index=daily.columns)


def max_drawdown_from_prices(prices: pd.DataFrame) -> pd.Series:
    """
    Drawdown maximum sur la courbe de **wealth** normalisée (premier prix = 1).

    DD_t = W_t / max(W_0..W_t) - 1 ; max drawdown = min(DD_t) (valeur négative).
    """
    out = {}
    for col in prices.columns:
        s = prices[col].dropna()
        if s.empty:
            out[col] = np.nan
            continue
        w = s / s.iloc[0]
        running_max = w.cummax()
        dd = w / running_max - 1.0
        out[col] = dd.min()
    return pd.Series(out)


def skewness_kurtosis(daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Skewness et kurtosis (excess kurtosis = Fisher) par colonne, NaN si série constante."""
    skew = daily.apply(lambda s: scipy_stats.skew(s.dropna(), bias=False))
    kurt = daily.apply(lambda s: scipy_stats.kurtosis(s.dropna(), fisher=True, bias=False))
    return skew, kurt


def performance_stats_table(
    prices: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    round_decimals: int = 4,
) -> pd.DataFrame:
    """
    Tableau unique : une ligne par série, colonnes = métriques demandées.

    Les rendements sous-jacents pour Sharpe / vol / rendement ann. / skew / kurt
    sont les **journaliers**. Le max drawdown est calculé sur les **prix** (NAV).
    """
    ann_ret = annualized_geometric_return(daily, trading_days)
    ann_vol = annualized_volatility(daily, trading_days)
    sharpe = sharpe_ratio(daily, risk_free_annual, trading_days)
    mdd = max_drawdown_from_prices(prices)
    skew, kurt = skewness_kurtosis(daily)

    table = pd.DataFrame(
        {
            "perf_annualisee": ann_ret,
            "volatilite_annualisee": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "skewness": skew,
            "kurtosis_excess": kurt,
        }
    )
    # Arrondi pour l'affichage (copie)
    return table.round(round_decimals)


# =============================================================================
# 5. EXPORT (CSV / Excel)
# =============================================================================


def export_tables(
    monthly_rets: pd.DataFrame,
    stats: pd.DataFrame,
    directory: Union[str, pathlib.Path],
    prefix: str = "analyse_ts",
) -> None:
    """Exporte les tableaux en CSV ; Excel si openpyxl est disponible."""
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    monthly_path = path / f"{prefix}_rendements_mensuels.csv"
    stats_path = path / f"{prefix}_statistiques.csv"
    monthly_rets.to_csv(monthly_path, encoding="utf-8-sig")
    stats.to_csv(stats_path, encoding="utf-8-sig")
    print(f"CSV écrits : {monthly_path} , {stats_path}")

    try:
        xlsx = path / f"{prefix}_rapport.xlsx"
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            monthly_rets.to_excel(writer, sheet_name="mensuel")
            stats.to_excel(writer, sheet_name="stats")
        print(f"Excel écrit : {xlsx}")
    except ImportError:
        print("(openpyxl non installé : export Excel ignoré. pip install openpyxl)")


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================


def run_full_financial_pipeline(
    prices: pd.DataFrame,
    *,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    export_dir: Optional[str] = EXPORT_DIR,
    figure_save_path: Optional[str] = None,
    show_figure: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Enchaîne : base 100, mensuel, stats.

    Retourne (rebased_100, monthly_returns, stats_table).

    figure_save_path : enregistre le graphique (recommandé pour un script .py sans affichage).
    show_figure : True pour plt.show() (notebook / environnement interactif).
    """
    prices_clean = prepare_price_frame(prices)

    # --- Graphique base 100 ---
    rebased = rebase_to_100(prices_clean)
    plot_base100(
        rebased,
        title="Comparaison des séries (base 100)",
        save_path=figure_save_path,
        show=show_figure,
    )

    # --- Mensuel ---
    monthly = monthly_returns_from_prices(prices_clean)
    monthly_display = format_monthly_returns_for_display(monthly)
    print("\n=== Performances mensuelles (en %, arrondi) ===")
    print(monthly_display.to_string())
    print("\n=== Tableau de performance mensuelle (annee x mois) ===")
    for col in monthly.columns:
        monthly_year_table = monthly_performance_table(monthly, series_name=col)
        print(f"\n--- {col} ---")
        print(monthly_year_table.to_string())
    print(
        "\n(NB : valeurs en % pour lecture ; les exports CSV gardent les rendements en decimal.)"
    )

    # --- Stats (journaliers) ---
    daily = daily_returns_from_prices(prices_clean)
    stats = performance_stats_table(
        prices_clean,
        daily,
        risk_free_annual=risk_free_annual,
        round_decimals=4,
    )
    print("\n=== Statistiques de performance ===")
    print(
        "Metriques basees sur les rendements journaliers "
        "(vol annualisee = ecart-type journalier * sqrt(252), etc.). "
        "Max drawdown calcule sur les prix (wealth relatif)."
    )
    print(stats.to_string())

    if export_dir:
        export_tables(monthly, stats, export_dir)

    return rebased, monthly, stats


def main() -> None:
    """Point d'entrée : charge les données, lance l'analyse."""
    prices = load_or_build_prices(csv_path=CSV_PATH)
    # En ligne de commande : on sauvegarde le graphique et on évite plt.show() qui peut bloquer.
    out_png = pathlib.Path(__file__).with_name("base100_comparaison.png")
    run_full_financial_pipeline(
        prices,
        risk_free_annual=RISK_FREE_ANNUAL,
        export_dir=EXPORT_DIR,
        figure_save_path=str(out_png),
        show_figure=False,
    )


if __name__ == "__main__":
    main()
