"""Reusable Plotly chart builders with the platform's dark styling.

Every dashboard page builds charts through this module so the visual
language stays consistent (fonts, transparency, accent colors).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ACCENT = "#00c896"
ACCENT_2 = "#7c4dff"
PALETTE = [ACCENT, ACCENT_2, "#ff8a65", "#4fc3f7", "#ffd54f", "#f06292"]


def _style(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared dark, transparent styling."""
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="sans-serif"),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.line(frame, x=x, y=y, color_discrete_sequence=[ACCENT])
    fig.update_traces(line_width=2.2)
    return _style(fig, title)


def bar_chart(frame: pd.DataFrame, x: str, y: str, title: str,
              horizontal: bool = False, color: str = ACCENT) -> go.Figure:
    if horizontal:
        fig = px.bar(frame, x=y, y=x, orientation="h", color_discrete_sequence=[color])
    else:
        fig = px.bar(frame, x=x, y=y, color_discrete_sequence=[color])
    return _style(fig, title)


def pie_chart(frame: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
    fig = px.pie(frame, names=names, values=values, hole=0.45,
                 color_discrete_sequence=PALETTE)
    return _style(fig, title)


def treemap(frame: pd.DataFrame, label: str, value: str, title: str) -> go.Figure:
    fig = px.treemap(frame, path=[label], values=value,
                     color=value, color_continuous_scale=["#131a2a", ACCENT])
    return _style(fig, title)


def correlation_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            colorscale="RdBu", zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        )
    )
    return _style(fig, title)


def radar_compare(categories: list[str], series: dict[str, list[float]], title: str) -> go.Figure:
    """Radar chart comparing multiple teams over shared axes (0-1 scaled)."""
    fig = go.Figure()
    for (name, values), color in zip(series.items(), PALETTE):
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1], theta=categories + categories[:1],
                fill="toself", name=name, line_color=color, opacity=0.75,
            )
        )
    fig.update_layout(polar=dict(
        bgcolor="rgba(255,255,255,0.03)",
        radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.12)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.12)"),
    ))
    return _style(fig, title)


def histogram(frame: pd.DataFrame, x: str, title: str, nbins: int = 30) -> go.Figure:
    fig = px.histogram(frame, x=x, nbins=nbins, color_discrete_sequence=[ACCENT_2])
    return _style(fig, title)


def grouped_bars(frame: pd.DataFrame, x: str, ys: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for y, color in zip(ys, PALETTE):
        fig.add_trace(go.Bar(name=y, x=frame[x], y=frame[y], marker_color=color))
    fig.update_layout(barmode="group")
    return _style(fig, title)
