"""AI PDF Report Generator: Compile analytics into a professional downloadable document."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import tempfile
import os
from datetime import datetime

from app.ui import hero, missing_data_warning, setup_page

setup_page("Report Generator", "📄")

from app.data_access import get_config, load_model, load_table, get_prediction_engine

try:
    from fpdf import FPDF
except ImportError:
    st.error("fpdf2 is not installed. Please run `pip install fpdf2`.")
    st.stop()

hero(
    "AI PDF Report Generator",
    "Compile model evaluations, tournament simulations, and team analytics into a professional PDF document.",
)

sims = load_table("simulation_probabilities")
loaded = load_model()
engine = get_prediction_engine()

if sims is None or loaded is None or engine is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

model_obj, metadata = loaded
config = get_config()


class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(11, 36, 71) # Deep corporate blue
        self.rect(0, 0, 210, 20, "F")
        self.set_font("helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 5, f"Advanced Analytics: {config.tournament.name} {config.tournament.year}", border=False, align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Proprietary & Confidential - Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("helvetica", "B", 18)
        self.set_text_color(11, 36, 71)
        self.cell(0, 12, title, ln=True)
        self.set_fill_color(0, 200, 150)
        self.rect(self.get_x(), self.get_y(), 190, 1, "F")
        self.ln(5)

    def body_text(self, text):
        self.set_font("helvetica", "", 11)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 7, text)
        self.ln(5)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.write("Click below to generate a comprehensive PDF report containing:")
st.markdown("""
- **Executive Summary** (Automated Insights)
- **Tournament Simulation Results** (Top Contenders)
- **Top 3 Analytics Charts Embedded**
- **Machine Learning Model Performance**
""")
st.markdown("</div>", unsafe_allow_html=True)

if st.button("Generate Professional PDF Report", type="primary"):
    with st.spinner("Compiling Analytics & Building PDF..."):
        pdf = PDFReport()
        pdf.add_page()

        # Title
        pdf.set_font("helvetica", "B", 26)
        pdf.set_text_color(11, 36, 71)
        pdf.cell(0, 20, "Executive Summary", ln=True)
        
        best_attack = max(engine._states.items(), key=lambda x: x[1].attack_strength)[0]
        best_defense = max(engine._states.items(), key=lambda x: x[1].defense_strength)[0]

        # Summary text
        pdf.body_text(
            f"This executive report outlines the AI-driven probabilistic forecast for the {config.tournament.name} {config.tournament.year}. "
            f"Leveraging 100,000 Monte Carlo simulations and an advanced machine learning pipeline ({metadata['model_name']}), "
            f"this analysis strictly adheres to empiric data spanning over a century of international football."
        )
        pdf.body_text(
            f"Key Findings:\n"
            f"• Tournament Favorite: {sims.iloc[0]['team']} ({sims.iloc[0]['champion_prob']:.1%} Win Probability)\n"
            f"• Global Model Accuracy: {metadata['leaderboard'][0]['accuracy']:.1%} (Out-of-sample)\n"
            f"• Highest Attack Index: {best_attack}\n"
            f"• Most Resilient Defense: {best_defense}"
        )
        pdf.ln(5)

        # Simulation Results
        pdf.section_title("Simulation Outcomes (Top 10 Contenders)")

        # Table Header
        pdf.set_fill_color(240, 245, 250)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(11, 36, 71)
        pdf.cell(60, 10, "Nation", border=1, fill=True)
        pdf.cell(40, 10, "Win Probability", border=1, fill=True, align="C")
        pdf.cell(40, 10, "Final Prob", border=1, fill=True, align="C")
        pdf.cell(40, 10, "Semi Prob", border=1, fill=True, align="C")
        pdf.ln()

        # Table Rows
        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(50, 50, 50)
        for _, row in sims.head(10).iterrows():
            pdf.cell(60, 10, row["team"], border=1)
            pdf.cell(40, 10, f"{row['champion_prob']:.1%}", border=1, align="C")
            pdf.cell(40, 10, f"{row['final_prob']:.1%}", border=1, align="C")
            pdf.cell(40, 10, f"{row['semifinal_prob']:.1%}", border=1, align="C")
            pdf.ln()

        pdf.add_page()

        # Embed Charts
        pdf.section_title("Visual Analytics & Model Performance")
        
        figures = Path("reports") / "figures"
        champ_chart = figures / "champion_probabilities.png"
        confusion = figures / "confusion_matrix.png"
        roc = figures / "roc_curves.png"

        if champ_chart.exists():
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "1. Tournament Win Probabilities", ln=True)
            pdf.image(str(champ_chart), w=160)
            pdf.ln(5)

        if roc.exists():
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "2. One-vs-Rest ROC Curves", ln=True)
            pdf.image(str(roc), w=140)
            pdf.ln(5)

        if confusion.exists():
            pdf.add_page()
            pdf.section_title("Performance Metrics")
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 10, "3. Champion Model Confusion Matrix", ln=True)
            pdf.image(str(confusion), w=140)
            pdf.ln(10)

        # Model Metrics
        pdf.body_text(
            f"The predictive engine is powered by {metadata['model_name']}, selected over 5 other algorithms. "
            f"It was rigorously evaluated using Time-Series Cross Validation."
        )

        pdf.set_font("helvetica", "B", 11)
        for metric, val in metadata["leaderboard"][0].items():
            if isinstance(val, float):
                pdf.cell(0, 8, f"• {metric.replace('_', ' ').title()}: {val:.3f}", ln=True)

        # Export
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        pdf.output(tmp.name)

        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

        st.success("Professional PDF generated successfully!")
        st.download_button(
            label="⬇️ Download Executive Report (PDF)",
            data=pdf_bytes,
            file_name=f"FIFA_2026_Analytics_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            type="primary",
        )

        os.unlink(tmp.name)
