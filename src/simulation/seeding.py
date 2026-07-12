"""Tournament group construction.

Groups come from ``data/wc2026_groups.csv`` (columns: ``group,team``)
when the official draw is known. Until then, the top 48 teams by
current Elo are seeded into 12 groups of 4 using pot-based snake
seeding — a data-driven stand-in, not a prediction.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

GROUPS_FILE = Path("data") / "wc2026_groups.csv"


def load_or_seed_groups(
    elo_ratings: pd.DataFrame,
    groups_file: Path = GROUPS_FILE,
    n_groups: int = 12,
    group_size: int = 4,
) -> dict[str, list[str]]:
    """Return tournament groups from file if present, else Elo pot seeding."""
    if groups_file.exists():
        frame = pd.read_csv(groups_file)
        if not {"group", "team"}.issubset(frame.columns):
            raise ValueError(f"{groups_file} must have 'group' and 'team' columns")
        groups = {
            str(name): members["team"].tolist()
            for name, members in frame.groupby("group", sort=True)
        }
        logger.info("Loaded %d groups from %s", len(groups), groups_file)
        return groups

    top = elo_ratings.sort_values("elo", ascending=False).head(n_groups * group_size)
    teams = top["team"].tolist()
    group_names = list(string.ascii_uppercase[:n_groups])
    groups = {name: [] for name in group_names}
    # Snake seeding: pot 1 fills A->L, pot 2 fills L->A, and so on.
    for pot in range(group_size):
        pot_teams = teams[pot * n_groups:(pot + 1) * n_groups]
        order = group_names if pot % 2 == 0 else list(reversed(group_names))
        for name, team in zip(order, pot_teams):
            groups[name].append(team)
    logger.info("Seeded %d groups from Elo (no groups file found)", n_groups)
    return groups
