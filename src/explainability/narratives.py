"""Plain-English prediction narratives.

Turns SHAP contributions into the 'Explain Why' text shown on the
Prediction page. Wording is generated from data — the narrative always
reflects the model, never an opinion.
"""

from __future__ import annotations

import pandas as pd

#: Human-readable phrasing per feature, keyed by direction of support.
_FEATURE_PHRASES: dict[str, str] = {
    "elo_diff": "the Elo rating gap between the sides",
    "home_elo_pre": "{home}'s overall Elo strength",
    "away_elo_pre": "{away}'s overall Elo strength",
    "form_diff": "the difference in recent form",
    "home_form_win_rate": "{home}'s recent win rate",
    "away_form_win_rate": "{away}'s recent win rate",
    "home_form_goals_for": "{home}'s recent scoring output",
    "away_form_goals_for": "{away}'s recent scoring output",
    "home_form_goals_against": "goals {home} has been conceding lately",
    "away_form_goals_against": "goals {away} has been conceding lately",
    "home_clean_sheet_rate": "{home}'s clean-sheet record",
    "away_clean_sheet_rate": "{away}'s clean-sheet record",
    "attack_diff": "the attacking strength differential",
    "defense_diff": "the defensive strength differential",
    "h2h_balance": "the head-to-head history between these teams",
    "importance": "the stakes of a World Cup fixture",
    "neutral": "the venue situation",
}

_OUTCOME_LABELS = {0: "{away} win", 1: "draw", 2: "{home} win"}


def generate_match_narrative(
    home: str,
    away: str,
    probabilities: tuple[float, float, float],
    contributions: pd.DataFrame,
    top_k: int = 4,
) -> str:
    """Build the 'Explain Why' paragraph for one fixture.

    Args:
        home: Home team name.
        away: Away team name.
        probabilities: (p_home_win, p_draw, p_away_win).
        contributions: Output of ``ModelExplainer.explain_prediction``.
        top_k: Number of drivers to mention.
    """
    p_home, p_draw, p_away = probabilities
    outcomes = {2: p_home, 1: p_draw, 0: p_away}
    predicted = max(outcomes, key=lambda key: outcomes[key])
    outcome_text = _OUTCOME_LABELS[predicted].format(home=home, away=away)
    confidence = 100.0 * outcomes[predicted]

    drivers: list[str] = []
    for row in contributions.head(top_k).itertuples(index=False):
        phrase = _FEATURE_PHRASES.get(str(row.feature), str(row.feature))
        direction = "supports" if row.contribution > 0 else "works against"
        drivers.append(
            f"{phrase.format(home=home, away=away)} ({direction} this outcome, "
            f"SHAP {row.contribution:+.3f})"
        )

    driver_text = "; ".join(drivers)
    return (
        f"The model predicts a {outcome_text} with {confidence:.1f}% confidence "
        f"(win {100 * p_home:.1f}% / draw {100 * p_draw:.1f}% / loss {100 * p_away:.1f}% "
        f"from {home}'s perspective). "
        f"The strongest drivers behind this prediction are: {driver_text}."
    )
