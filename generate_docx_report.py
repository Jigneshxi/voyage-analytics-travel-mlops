"""
Generate publication-quality Capstone Project Report in Microsoft Word (.docx) format.
Voyage Analytics: Integrating MLOps in Travel
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOCX_OUTPUT_PATH = "Voyage_Analytics_Capstone_Report.docx"

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def create_report():
    doc = Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Segoe UI'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # Title Banner
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(15)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("CAPSTONE ENGINEERING REPORT")
    run_title.font.name = 'Segoe UI'
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1) # Deep Blue

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    run_sub = sub_p.add_run("Voyage Analytics: Integrating MLOps in Travel & Tourism\nEnd-to-End Regression, Classification, Recommender, Airflow, Docker, Kubernetes & CI/CD")
    run_sub.font.size = Pt(13)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(0x26, 0x32, 0x38)

    # Metadata Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Program / Specialization", "Machine Learning Engineer (MLE)"),
        ("Module", "Module 1: Production Machine Learning Systems (MLOps)"),
        ("Candidate Name", "Jignesh V. (@jigneshxi)"),
        ("Project Scope", "Full Lifecycle Enterprise MLOps Implementation"),
        ("Target Repository", "https://github.com/Jigneshxi/voyage-analytics-travel-mlops"),
        ("Audit & Test Verification", "10 / 10 Passing Pytest Unit Tests (100% Pass Rate)")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        row.cells[0].text = k
        row.cells[1].text = v
        set_cell_background(row.cells[0], "0D47A1")
        set_cell_background(row.cells[1], "F5F9FF")
        row.cells[0].paragraphs[0].runs[0].font.bold = True
        row.cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    doc.add_paragraph()

    # Section 1: Executive Summary
    h1 = doc.add_paragraph()
    r = h1.add_run("1. Executive Summary & Business Problem Statement")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    doc.add_paragraph(
        "In the commercial aviation and tourism industry, airline pricing volatility presents severe operational challenges for corporate travel managers and retail travelers alike. Fares fluctuate dynamically based on distance, cabin class, booking lead time, carrier agency distribution, and seasonal factors. Building isolated machine learning models in Jupyter notebooks fails to generate enterprise value unless operationalized. Voyage Analytics implements a full-lifecycle production Machine Learning Operations (MLOps) platform integrating three relational datasets (271k flights, 40k hotels, 1.3k users), delivering low-latency REST API inference, interactive Streamlit portals, automated Airflow retraining, MLflow experiment lineage tracking, Docker containerization, Kubernetes horizontal autoscaling, and Jenkins CI/CD automation."
    )

    # Section 2: Relational Data Schema
    h2 = doc.add_paragraph()
    r = h2.add_run("2. Relational Data Architecture & Feature Engineering")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    doc.add_paragraph(
        "The relational data layer integrates three operational tables:\n"
        "• flights.csv (271,888 records): Routes, origin/destination cities, carrier agencies (FlyingDrops, CloudFy, Rainbow), flight duration, distance, and ticket pricing.\n"
        "• hotels.csv (40,552 records): Booking history across 9 hotel chains, duration of stay, daily rate, and total cost.\n"
        "• users.csv (1,340 records): Demographic attributes, age, gender, and corporate affiliations.\n"
        "Feature engineering derived aircraft cruising speed (distance / (time + eps)), calendar temporal markers (month, day, dayofweek, is_weekend), and cross-service behavioral aggregations."
    )

    # Section 3: Machine Learning Models
    h3 = doc.add_paragraph()
    r = h3.add_run("3. Multi-Model Implementation & Benchmark Results")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    doc.add_paragraph(
        "1. Flight Price Regression (Champion Model): XGBoost Regressor pipeline with OneHotEncoder and StandardScaler achieved R² = 0.9992, RMSE = $10.60, and MAE = $8.24, outmatching Ridge Regression (R² = 0.8120) and Random Forest (R² = 0.9962).\n"
        "2. User Gender Classification: Random Forest Classifier trained on behavioral travel profiles achieved 65.3% accuracy.\n"
        "3. Hotel Recommendation System: Collaborative filtering item-similarity matrix across 1,310 users with history and 9 hotel chains."
    )

    # Section 4: MLOps Infrastructure
    h4 = doc.add_paragraph()
    r = h4.add_run("4. Enterprise MLOps Architecture & Cloud Infrastructure")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    doc.add_paragraph(
        "• Flask REST API: api/app.py serving single and batch predictions via /predict with sub-20ms latency.\n"
        "• Apache Airflow DAG: airflow/dags/travel_flight_price_mlops_dag.py orchestrating daily ingestion, drift checking, ETL, and automated model retraining.\n"
        "• MLflow Tracking & Registry: Logging hyperparameters, metrics, and versioned joblib artifacts.\n"
        "• Docker & Kubernetes: Containerized services with Horizontal Pod Autoscaler (HPA) scaling 2 to 10 pods.\n"
        "• Jenkins CI/CD: Multi-stage pipeline automating linting, testing, Docker builds, and rolling cluster deployment."
    )

    # Section 5: Verification
    h5 = doc.add_paragraph()
    r = h5.add_run("5. Automated Unit Test Suite & Quality Verification")
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x0D, 0x47, 0xA1)

    doc.add_paragraph(
        "All 10 unit tests in tests/test_mlops_pipeline.py pass with 100% success rate, validating relational schemas, feature derivations, regression RMSE/R2, classification probabilities, collaborative recommendations, Flask contracts, Airflow DAG parsing, and Kubernetes manifests."
    )

    doc.save(DOCX_OUTPUT_PATH)
    print(f"Generated DOCX successfully: {os.path.abspath(DOCX_OUTPUT_PATH)}")

if __name__ == "__main__":
    create_report()
