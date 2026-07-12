"""Shared UI building blocks for the dashboard.

Centralizes page setup, the glassmorphism CSS injection, the hero
section, and metric cards so every page stays visually consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"


def setup_page(title: str, icon: str = "\u26bd") -> None:
    """Configure the page and inject the shared theme. Call first on every page."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    st.set_page_config(page_title=f"{title} \u00b7 FIFA 2026 Analytics",
                       page_icon=icon, layout="wide")
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Full-width gradient hero section."""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    """Row of glass metric cards. Each item: (label, value, delta_text)."""
    columns = st.columns(len(metrics))
    for column, (label, value, delta) in zip(columns, metrics):
        delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
        column.markdown(
            f'<div class="glass-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{delta_html}</div>',
            unsafe_allow_html=True,
        )


def section(title: str) -> None:
    """Styled section heading."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def missing_data_warning(step: str) -> None:
    """Consistent guidance when a pipeline artifact is absent."""
    st.warning(f"Data not found. Run `{step}` from the repository root, then refresh.")
