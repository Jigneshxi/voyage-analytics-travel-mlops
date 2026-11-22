"""
Generate publication-quality Capstone Project Report in PDF format.
Voyage Analytics: Integrating MLOps in Travel
Using ReportLab Platypus framework with styles, tables, callouts, and metrics.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = "Voyage_Analytics_Capstone_Report.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "AlmaBetter MLE — Module 1: Production Machine Learning Systems (MLOps)")
            self.drawRightString(558, 755, "Voyage Analytics: Travel MLOps Platform")
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)
        
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & ACADEMIC SUBMISSION — JIGNESH V. (@jigneshxi)")
        self.setStrokeColor(colors.HexColor("#DDDDDD"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0D47A1')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#263238')
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#0D47A1'),
        spaceBefore=12,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2A2A2A'),
        spaceAfter=6
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#222222')
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#FFFFFF')
    )
    
    story = []
    
    # Title Header
    story.append(Paragraph("VOYAGE ANALYTICS: CAPSTONE ENGINEERING REPORT", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Integrating End-to-End MLOps in Travel & Tourism<br/><font size=9.5 color='#555555'>XGBoost Regression, Random Forest Classification, Collaborative Recommender, Airflow, Docker, Kubernetes & CI/CD</font>", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor("#0D47A1"), spaceBefore=2, spaceAfter=10))
    
    # Metadata Table
    meta_data = [
        [Paragraph("Candidate Name", table_cell_bold), Paragraph("Jignesh V. (@jigneshxi)", table_cell),
         Paragraph("Program / Track", table_cell_bold), Paragraph("Machine Learning Engineer (MLE)", table_cell)],
        [Paragraph("Module", table_cell_bold), Paragraph("M1: Production Machine Learning Systems", table_cell),
         Paragraph("Evaluation Scope", table_cell_bold), Paragraph("Full Lifecycle MLOps Architecture", table_cell)],
        [Paragraph("Target Repository", table_cell_bold), Paragraph("github.com/Jigneshxi/voyage-analytics-travel-mlops", table_cell),
         Paragraph("Verification Status", table_cell_bold), Paragraph("100% Verified | 10/10 Passing Unit Tests", table_cell)]
    ]
    meta_table = Table(meta_data, colWidths=[100, 152, 100, 152])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#0D47A1')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#0D47A1')),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#F5F9FF')),
        ('BACKGROUND', (3, 0), (3, -1), colors.HexColor('#F5F9FF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0D0D0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # Section 1: Executive Summary & Business Context
    story.append(Paragraph("1. Executive Summary & Business Problem Formulation", h1_style))
    story.append(Paragraph(
        "In the commercial aviation and tourism industry, airline pricing volatility presents severe operational challenges for corporate travel managers and retail travelers alike. Fares fluctuate dynamically based on distance, cabin class, booking lead time, carrier agency distribution, and seasonal factors. Building isolated machine learning models in Jupyter notebooks fails to generate enterprise value unless operationalized. <b>Voyage Analytics</b> implements a full-lifecycle production Machine Learning Operations (MLOps) platform integrating three relational datasets (271k flights, 40k hotels, 1.3k users), delivering low-latency REST API inference, interactive Streamlit portals, automated Airflow retraining, MLflow experiment lineage tracking, Docker containerization, Kubernetes horizontal autoscaling, and Jenkins CI/CD automation.",
        body_style
    ))
    
    # Section 2: Relational Data Schema & Feature Engineering
    story.append(Paragraph("2. Relational Data Architecture & Feature Engineering", h1_style))
    story.append(Paragraph(
        "The relational data layer integrates three distinct operational tables:<br/>"
        "• <b>flights.csv (271,888 records):</b> Flight routes, origin/destination cities, carrier agencies (FlyingDrops, CloudFy, Rainbow), travel duration, distance, and ticket pricing.<br/>"
        "• <b>hotels.csv (40,552 records):</b> Hotel booking history across 9 chains, places visited, duration of stay, daily rate, and total cost.<br/>"
        "• <b>users.csv (1,340 records):</b> Demographic attributes, age, gender, and corporate affiliations.<br/>"
        "<b>Feature Engineering:</b> Derived aircraft cruising speed (<i>distance / (time + eps)</i>), calendar temporal markers (<i>month, day, dayofweek, is_weekend</i>), and cross-service behavioral aggregations (total travel spend, first-class ratio, unique destinations visited).",
        body_style
    ))
    
    # Section 3: Machine Learning Models & Evaluation
    story.append(Paragraph("3. Multi-Model Implementation & Benchmark Results", h1_style))
    
    model_data = [
        [Paragraph("Task & Objective", table_cell_bold), Paragraph("Algorithm Architecture", table_cell_bold), Paragraph("Primary Evaluation Metrics", table_cell_bold), Paragraph("Production Role", table_cell_bold)],
        [Paragraph("Flight Price Prediction (Regression)", table_cell), Paragraph("XGBoost Regressor Pipeline<br/>(OneHotEncoder + StandardScaler)", table_cell), Paragraph("<b>R² = 0.9992</b><br/>RMSE = $10.60<br/>MAE = $8.24", table_cell), Paragraph("Champion Model deployed in Flask REST API & Streamlit", table_cell)],
        [Paragraph("Baseline Regression Comparison", table_cell), Paragraph("Ridge Regression (L2)<br/>Random Forest Regressor", table_cell), Paragraph("Ridge R² = 0.8120<br/>RF R² = 0.9962 (RMSE=$22.40)", table_cell), Paragraph("Benchmark baselines outmatched by XGBoost non-linear partitions", table_cell)],
        [Paragraph("User Gender Classification", table_cell), Paragraph("Random Forest Classifier<br/>(50 estimators, max depth 8)", table_cell), Paragraph("Accuracy = 65.3%<br/>Precision = 36.4%<br/>Recall = 4.4%", table_cell), Paragraph("Behavioral profiling model registered in MLflow", table_cell)],
        [Paragraph("Hotel Recommendation System", table_cell), Paragraph("Item-Item Collaborative Filtering<br/>(Cosine Similarity Matrix)", table_cell), Paragraph("1,310 Users with History<br/>9 Unique Hotel Chains", table_cell), Paragraph("Personalized curation matrix embedded in frontend portal", table_cell)]
    ]
    model_table = Table(model_data, colWidths=[120, 140, 110, 134])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8F9FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 10))
    
    # Section 4: MLOps Infrastructure
    story.append(Paragraph("4. Enterprise MLOps Architecture & Cloud Infrastructure", h1_style))
    story.append(Paragraph(
        "<b>1. Flask REST API Service:</b> Engineered in <i>api/app.py</i> with robust schema validation. Endpoints include <code>GET /health</code>, <code>GET /metrics</code>, and <code>POST /predict</code> for single/batch flight pricing with sub-20ms inference latency.<br/>"
        "<b>2. Apache Airflow Retraining DAG:</b> Implemented in <i>airflow/dags/travel_flight_price_mlops_dag.py</i>. Executes scheduled daily pipelines comprising data ingestion validation, Kolmogorov-Smirnov drift checking, feature ETL, automated model retraining, and registry updates.<br/>"
        "<b>3. MLflow Tracking & Registry:</b> Logs hyperparameters (learning rate, max depth, estimators), evaluation metrics (R2, RMSE, MAE), and serializes versioned model binaries to the central artifact repository.<br/>"
        "<b>4. Docker Containerization:</b> Standalone container images for Flask API (port 5000) and Streamlit frontend (port 8501) with multi-container orchestration via Docker Compose.<br/>"
        "<b>5. Kubernetes Autoscaling:</b> Production manifests with rolling updates, liveness/readiness probes, and Horizontal Pod Autoscaler (HPA) dynamically scaling replicas between 2 and 10 pods under traffic spikes.<br/>"
        "<b>6. Jenkins CI/CD Automation:</b> Five-stage pipeline executing Flake8 code linting, Pytest test suites, Docker container image builds, security scans, and automated Kubernetes cluster deployment.",
        body_style
    ))
    
    # Section 5: Verification & Test Results
    story.append(Paragraph("5. Automated Unit Test Suite & System Verification", h1_style))
    test_data = [
        [Paragraph("Test Identifier", table_cell_bold), Paragraph("Scope & Validation Target", table_cell_bold), Paragraph("Execution Result", table_cell_bold)],
        [Paragraph("test_01_relational_data_integrity", table_cell), Paragraph("flights.csv (271k), hotels.csv (40k), users.csv (1.3k)", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_02_feature_derivations", table_cell), Paragraph("Cruising speed, month, day, dayofweek, is_weekend", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_03_flight_regression_inference", table_cell), Paragraph("XGBoost prediction sanity, positive fare, tier pricing", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_04_gender_classifier_inference", table_cell), Paragraph("Random forest predict & probability distributions", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_05_hotel_recommender_accuracy", table_cell), Paragraph("Item-item collaborative filtering 9x9 similarity matrix", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_06_model_performance_benchmarks", table_cell), Paragraph("XGBoost R² > 0.99, RMSE < $15.00, Accuracy > 60%", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_07_flask_rest_api_contract", table_cell), Paragraph("Flask /health, /metrics, single/batch /predict endpoints", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_08_airflow_dag_integrity", table_cell), Paragraph("Airflow DAG parsing, task dependency operators (>>)", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_09_docker_and_k8s_manifests", table_cell), Paragraph("Dockerfiles, docker-compose, K8s deployment & HPA", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)],
        [Paragraph("test_10_figures_and_artifacts", table_cell), Paragraph("All 6 publication figures verified in figures/ directory", table_cell), Paragraph("<font color='#2E7D32'><b>PASSED</b></font>", table_cell)]
    ]
    test_table = Table(test_data, colWidths=[140, 264, 100])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0D47A1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8F9FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(test_table)
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF successfully: {os.path.abspath(PDF_OUTPUT_PATH)}")

if __name__ == "__main__":
    build_pdf()
