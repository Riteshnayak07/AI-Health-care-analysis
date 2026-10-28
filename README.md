# PulseAI – Intelligent Clinical Healthcare & Diagnostic Assistant
**MCA Project | Multi-Modal AI Healthcare Analysis System**

PulseAI is an advanced healthcare decision-support and triage prototype built with **Python, Flask, NLP (TF-IDF & Scikit-Learn), Computer Vision (OpenCV), SQLite, and Chart.js**. Designed for clinical simulation and academic demonstrations, it bridges multi-modal artificial intelligence with an intuitive glassmorphic clinical interface.

---

## 🌟 Key Features

### 1. 🩺 Intelligent NLP Symptom Checker & Clinical Triage
- **Natural Language Parsing**: Accepts everyday spoken or written descriptions of ailments.
- **Voice-to-Text Input**: Integrated Web Speech API microphone dictation for hands-free symptom recording.
- **Interactive Quick-Add Symptom Cloud**: Anatomically organized chips (*General*, *Head & ENT*, *Respiratory*, *Digestive*, *Skin & Urinary*).
- **Ensemble Inference**: Combines TF-IDF Vectorization and Logistic Regression ML classification with emergency heuristic safety overrides.
- **Triage Priority Stratification**: Categorizes urgency into **Urgent Emergency (Red)**, **Moderate Clinical (Amber)**, or **Mild Home Care (Green)** based on duration and severity.
- **Actionable Guidance**: Details clinical explanations, recommended medical specialists (e.g. Pulmonologist, Cardiologist, Neurologist), home-care checklists, and clinical red flags.

### 2. 🔬 Medical Computer Vision Scan & Lesion Analyzer
- **Diagnostic Image Quality Assessment**: Evaluates Laplacian sharpness variance, contrast standard deviation, and brightness uniformity.
- **Erythema / Redness Ratio**: Analyzes normalized chromatic red dominance to flag cutaneous inflammation and skin rash conditions.
- **Feature & Edge Heatmap Overlay**: Employs OpenCV Canny filters and JET colormap blends to output diagnostic edge previews.

### 3. 🧮 Health Vitals & Risk Calculators Suite
- **BMI & BMR Calculator**: Computes Body Mass Index, ideal weight range, and Basal Metabolic Rate (Mifflin-St Jeor).
- **Cardiovascular Lifestyle Risk Screener**: Multi-factor scoring analyzing age, resting heart rate, blood pressure, smoking, and physical activity.
- **Daily Hydration Planner**: Tailors optimal daily fluid intake based on body mass, exercise duration, and ambient temperature.

### 4. 📊 Clinical Analytics & Epidemiological Dashboard
- **Interactive Visualizations (Chart.js)**:
  - *Condition Prevalence Distribution* (Doughnut chart)
  - *Triage Urgency Breakdown* (Bar chart)
  - *Most Frequent Clinical Symptoms* (Horizontal bar ranking)
- **Consultation Registry**: Live searchable, filterable registry of all past symptom analyses and computer vision logs.
- **Data Portability**: Instant one-click CSV export and sample demo data loader.

### 5. 🚑 Emergency Directory & First-Aid Knowledge Base
- **One-Touch Emergency Calling**: Direct phone links for 112, 108/102 (Ambulance), and Poison Control.
- **Step-by-Step Life Support Guides**: Rapid protocols for Hands-Only CPR, Heimlich Maneuver, Bleeding Control, and Burns.
- **Specialist Referral Matrix**: Maps symptoms to corresponding medical departments.

### 6. 🖨️ Standalone Printable Patient Summary Sheets
- Generates clean, printer-friendly consultation reports suitable for patient records or referral slips.

---

## 📂 Project Architecture

```
AI_Healthcare_Analysis/
├── app.py                     # Main Flask web application controller
├── train_model.py             # Machine learning training script (TF-IDF + Logistic Regression)
├── requirements.txt           # Python package dependencies
├── healthcare.db              # SQLite persistent database
│
├── dataset/
│   └── symptoms_disease.csv   # Comprehensive labeled symptom-disease training dataset
│
├── model/ & models/
│   ├── healthcare_model.pkl   # Serialized end-to-end ML pipeline
│   ├── symptom_model.pkl      # Trained classifier
│   └── tfidf_vectorizer.pkl   # Fitted vocabulary vectorizer
│
├── nlp/
│   └── symptom_prediction.py  # Clinical taxonomy, typo-tolerant NLP, and triage logic
│
├── computer_vision/
│   └── image_analysis.py      # OpenCV sharpness, contrast, erythema & edge heatmap
│
├── database/
│   └── database.py            # SQLite schema migrations, analytics aggregations, and CRUD
│
├── utils/
│   └── report_generator.py    # CSV export and printable HTML summary generator
│
├── static/
│   ├── style.css              # Glassmorphic responsive design system with Dark/Light mode
│   └── app.js                 # Web Speech API, dynamic chips, live AJAX preview, Chart.js
│
└── templates/
    ├── base.html              # Core layout, theme switcher, navigation, disclaimer
    ├── index.html             # Symptom checker with voice input and chip selector
    ├── result.html            # Diagnostic assessment with confidence meter and checklist
    ├── vision.html            # Computer Vision upload, metrics, and heatmap viewer
    ├── vitals.html            # Health vitals and cardiovascular risk screener
    ├── dashboard.html         # Clinical analytics charts, search, and CSV export
    └── emergency.html         # Emergency hotline cards and first-aid protocols
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Installation
```bash
# Clone or navigate to the directory
cd AI_Healthcare_Analysis

# Create and activate virtual environment (recommended)
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Retrain Machine Learning Model
```bash
python train_model.py
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:5000`**
