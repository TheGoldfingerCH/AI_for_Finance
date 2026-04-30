import pandas as pd
import numpy as np
from typing import Callable

def run_backtest(
    df: pd.DataFrame,
    model,
    features: list[str],
    horizon: int = 5,
    start_date: str = None,
    duration_days: int = 365,
    initial_capital: float = 10000.0,
    strategies: list[str] = ['long_only', 'signal', 'perfect', 'random'],
    extra_strategies: dict[str, Callable] = {}
) -> dict:
    """
    Simule plusieurs stratégies de trading sur une période donnée.

    Args:
        df: DataFrame OHLCV avec features déjà calculées
        model: modèle entraîné avec méthode predict()
        features: liste des features utilisées par le modèle
        horizon: nombre de jours de prédiction
        start_date: date de début (str 'YYYY-MM-DD'), défaut = début du test set
        duration_days: durée de la simulation en jours
        initial_capital: capital initial par batch
        strategies: liste des stratégies à simuler
        extra_strategies: dict {'nom': callable(row) -> bool} pour stratégies custom

    Returns:
        dict avec 'results' (DataFrame des trades) et 'summary' (DataFrame des performances)
    """

    df = df.copy().sort_index()
    df.index = pd.to_datetime(df.index)

    # Calculer target réel
    df['target_real'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df['return_fwd'] = df['Close'].shift(-horizon) / df['Close'] - 1

    # Prédictions modèle
    df['signal'] = np.nan
    valid_idx = df[features].dropna().index
    df.loc[valid_idx, 'signal'] = model.predict(df.loc[valid_idx, features])

    df = df.dropna(subset=['signal', 'target_real', 'return_fwd'])

    # Définir la période
    max_end = df.index[-(horizon + 1)]
    if start_date:
        start = pd.Timestamp(start_date)
    else:
        start = df.index[int(len(df) * 0.8)]

    end = start + pd.Timedelta(days=duration_days)
    if end > max_end:
        end = max_end
    df_sim = df.loc[start:end]

    if len(df_sim) == 0:
        raise ValueError(f"Aucune donnée entre {start.date()} et {end.date()}")

    # Batches : on découpe en cycles de `horizon` jours
    batches = [df_sim.iloc[i:i+horizon] for i in range(0, len(df_sim), horizon)]
    batches = [b for b in batches if len(b) == horizon]

    print(f"Période  : {start.date()} → {end.date()}")
    print(f"Batches  : {len(batches)} × {horizon} jours")

    # Définition des stratégies
    strategy_signals = {
        'long_only': lambda row, batch: True,
        'signal':    lambda row, batch: bool(batch['signal'].iloc[0]),
        'perfect':   lambda row, batch: bool(batch['target_real'].iloc[0]),
        'random':    lambda row, batch: np.random.choice([True, False]),
    }
    strategy_signals.update(extra_strategies)

    active_strategies = {k: v for k, v in strategy_signals.items() if k in strategies or k in extra_strategies}

    # Simulation
    portfolio = {s: initial_capital for s in active_strategies}
    records = []

    for batch in batches:
        date = batch.index[0]
        actual_return = batch['return_fwd'].iloc[0]
        price_start = batch['Close'].iloc[0]
        price_end = batch['Close'].iloc[-1]

        for strat_name, signal_fn in active_strategies.items():
            invest = signal_fn(None, batch)
            if invest:
                pnl_pct = actual_return
            else:
                pnl_pct = 0.0

            pnl_abs = portfolio[strat_name] * pnl_pct
            portfolio[strat_name] += pnl_abs

            records.append({
                'date': date,
                'strategy': strat_name,
                'invested': invest,
                'return_pct': round(pnl_pct * 100, 4),
                'pnl': round(pnl_abs, 2),
                'portfolio_value': round(portfolio[strat_name], 2),
                'price_start': round(price_start, 2),
                'price_end': round(price_end, 2),
            })

    results = pd.DataFrame(records)

    # Résumé par stratégie
    summary_rows = []
    for strat in active_strategies:
        s = results[results['strategy'] == strat]
        final = s['portfolio_value'].iloc[-1]
        total_return = (final - initial_capital) / initial_capital * 100
        win_rate = s[s['invested']]['return_pct'].gt(0).mean() * 100
        n_invested = s['invested'].sum()
        summary_rows.append({
            'strategy': strat,
            'final_value': round(final, 2),
            'total_return_%': round(total_return, 2),
            'win_rate_%': round(win_rate, 2),
            'n_trades': int(n_invested),
            'n_batches': len(s),
        })

    summary = pd.DataFrame(summary_rows).sort_values('total_return_%', ascending=False)

    return {'results': results, 'summary': summary}
