"""AI PDF Report Generator: Compile analytics into a professional downloadable document."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import tempfile
import os

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("Report Generator", "📄")

from app.data_access import get_config, load_model, load_table

try:
    from fpdf import FPDF
except ImportError:
    st.error("fpdf2 is not installed. Please run `pip install fpdf2`.")
    st.stop()

hero("AI PDF Report Generator", "Compile model evaluations, tournament simulations, and team analytics into a professional PDF document.")

sims = load_table("simulation_probabilities")
loaded = load_model()

if sims is None or loaded is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

model_obj, metadata = loaded
config = get_config()

class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.set_text_color(0, 200, 150)
        self.cell(0, 10, f"{config.tournament.name} {config.tournament.year} - AI Analytics Report", border=False, align="C")
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.write("Click below to generate a comprehensive PDF report containing:")
st.markdown("""
- **Executive Summary**
- **Tournament Simulation Results (Top Contenders)**
- **Machine Learning Model Performance (Champion Model Metrics)**
- **Historical Analysis Overview**
""")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Generate Professional PDF Report", type="primary"):
    with st.spinner("Compiling Analytics & Building PDF..."):
        pdf = PDFReport()
        pdf.add_page()
        
        # Title
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(11, 15, 25)
        pdf.cell(0, 20, "Executive Summary", ln=True)
        
        # Summary text
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 8, f"This report outlines the AI-driven forecast for the {config.tournament.name} {config.tournament.year}. Based on 100,000 Monte Carlo simulations utilizing a {metadata['model_name']} model (Accuracy: {metadata['leaderboard'][0]['accuracy']:.1%}), we have evaluated the probabilities of all participating nations.")
        pdf.ln(10)
        
        # Simulation Results
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, "Top Contenders (Simulation Probabilities)", ln=True)
        pdf.set_font("helvetica", "B", 12)
        
        # Table Header
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(60, 10, "Team", border=1, fill=True)
        pdf.cell(40, 10, "Win Probability", border=1, fill=True)
        pdf.cell(40, 10, "Final Prob", border=1, fill=True)
        pdf.cell(40, 10, "Semi Prob", border=1, fill=True)
        pdf.ln()
        
        # Table Rows
        pdf.set_font("helvetica", "", 12)
        for _, row in sims.head(10).iterrows():
            pdf.cell(60, 10, row["team"], border=1)
            pdf.cell(40, 10, f"{row['champion_prob']:.1%}", border=1)
            pdf.cell(40, 10, f"{row['final_prob']:.1%}", border=1)
            pdf.cell(40, 10, f"{row['semifinal_prob']:.1%}", border=1)
            pdf.ln()
            
        pdf.add_page()
        
        # Model Metrics
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, "Model Observatory & Evaluation", ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 8, f"The predictive engine is powered by {metadata['model_name']}, selected over 5 other algorithms. It was evaluated using Time-Series Cross Validation.")
        pdf.ln(5)
        
        for metric, val in metadata["leaderboard"][0].items():
            if isinstance(val, float):
                pdf.cell(0, 8, f"- {metric.replace('_', ' ').title()}: {val:.3f}", ln=True)
                
        # Embed existing matplotlib charts if available
        figures = Path("reports") / "figures"
        confusion = figures / "confusion_matrix.png"
        roc = figures / "roc_curves.png"
        
        if confusion.exists():
            pdf.ln(10)
            pdf.set_font("helvetica", "B", 14)
            pdf.cell(0, 10, "Confusion Matrix", ln=True)
            pdf.image(str(confusion), w=150)
            
        if roc.exists():
            pdf.add_page()
            pdf.set_font("helvetica", "B", 14)
            pdf.cell(0, 10, "ROC Curves", ln=True)
            pdf.image(str(roc), w=150)
            
        # Export
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        pdf.output(tmp.name)
        
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
            
        st.success("PDF generated successfully!")
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"FIFA_2026_Analytics_Report.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        os.unlink(tmp.name)
