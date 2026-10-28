"""
PulseAI - Advanced AI Healthcare & Clinical Diagnostic Assistant
Features:
- NLP Symptom Analysis & Triage Urgency Assessment (Rule + ML Ensemble)
- Computer Vision Medical Image Quality & Erythema/Rash Screener (OpenCV)
- Health Vitals & Risk Calculators (BMI, Cardiac Risk, Hydration Planner)
- Clinical Analytics Dashboard with Chart.js Visualizations
- Emergency Directory & First-Aid Guide
- Exportable Clinical Reports & CSV Data
"""
import os
import io
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, Response

from nlp.symptom_prediction import extract_symptoms, predict_condition, SYMPTOM_LEXICON
from computer_vision.image_analysis import analyze_image
from database.database import (
    init_db, save_analysis, get_history, get_analytics_stats,
    delete_analysis, clear_all, seed_demo_data
)
from utils.report_generator import export_history_to_csv, generate_patient_summary_html

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize database schema on startup
init_db()

@app.route("/")
def index():
    # Organize quick symptoms by anatomical / functional category for chip selector
    symptom_categories = {
        "Common / General": ["fever", "chills", "fatigue", "body pain", "joint pain", "sweating"],
        "Head, Eyes & ENT": ["headache", "dizziness", "sore throat", "runny nose", "nasal congestion", "sneezing", "light sensitivity"],
        "Chest & Respiratory": ["cough", "shortness of breath", "chest pain", "wheezing", "chest congestion", "palpitations"],
        "Digestive / Abdominal": ["nausea", "vomiting", "diarrhea", "stomach pain", "heartburn"],
        "Skin & Urinary": ["rash", "itching", "skin redness", "burning urination", "frequent urination"]
    }
    return render_template("index.html", categories=symptom_categories)

@app.route("/analyze", methods=["POST"])
def analyze_route():
    text = request.form.get("symptoms", "").strip()
    severity = request.form.get("severity", "moderate").lower()
    try:
        duration = int(request.form.get("duration", 2))
    except (ValueError, TypeError):
        duration = 2

    detected = extract_symptoms(text)
    res = predict_condition(text, detected, severity=severity, duration_days=duration)
    
    # Save into SQLite history
    record_id = save_analysis(
        analysis_type="symptom",
        text=text,
        symptoms=", ".join(detected),
        prediction=res["prediction"],
        confidence=res["confidence"],
        urgency=res["urgency"]
    )
    
    return render_template(
        "result.html",
        record_id=record_id,
        text=text,
        symptoms=detected,
        prediction=res["prediction"],
        confidence=res["confidence"],
        urgency=res["urgency"],
        specialist=res["specialist"],
        description=res["description"],
        home_care=res["home_care"],
        red_flags=res["red_flags"],
        secondary=res["secondary_matches"],
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("symptoms", "")).strip()
    severity = str(data.get("severity", "moderate")).lower()
    try:
        duration = int(data.get("duration", 2))
    except (ValueError, TypeError):
        duration = 2

    if not text:
        return jsonify({"error": "No symptoms provided."}), 400

    detected = extract_symptoms(text)
    res = predict_condition(text, detected, severity=severity, duration_days=duration)

    record_id = save_analysis(
        analysis_type="symptom",
        text=text,
        symptoms=", ".join(detected),
        prediction=res["prediction"],
        confidence=res["confidence"],
        urgency=res["urgency"]
    )

    return jsonify({
        "id": record_id,
        "input_text": text,
        "symptoms": detected,
        "prediction": res["prediction"],
        "confidence": res["confidence"],
        "urgency": res["urgency"],
        "specialist": res["specialist"],
        "description": res["description"],
        "home_care": res["home_care"],
        "red_flags": res["red_flags"],
        "secondary_matches": res["secondary_matches"]
    })

@app.route("/vision", methods=["GET", "POST"])
def vision_route():
    result = None
    confidence = None
    details = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please select a valid image file to upload."
        else:
            try:
                img_bytes = file.read()
                result, confidence, details = analyze_image(img_bytes)
                
                # Log scan in database
                save_analysis(
                    analysis_type="computer_vision",
                    text=f"Medical Scan Image: {file.filename} ({details['resolution']})",
                    symptoms=f"Erythema: {details['redness_index']}, Sharpness: {details['sharpness_laplacian']}",
                    prediction=details["quality_assessment"],
                    confidence=confidence,
                    urgency="Mild" if confidence >= 60 else "Moderate"
                )
            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render_template("vision.html", result=result, confidence=confidence, details=details, error=error)

@app.route("/vitals", methods=["GET", "POST"])
def vitals_route():
    bmi_data = None
    cardiac_data = None
    water_data = None

    if request.method == "POST":
        calc_type = request.form.get("calc_type")
        
        # 1. BMI & BMR Calculator
        if calc_type == "bmi":
            try:
                weight = float(request.form.get("weight", 0))
                height = float(request.form.get("height", 0)) / 100.0  # cm to m
                age = int(request.form.get("age", 25))
                gender = request.form.get("gender", "male")

                if height > 0 and weight > 0:
                    bmi = round(weight / (height * height), 1)
                    if bmi < 18.5:
                        category = "Underweight"
                        status_class = "warning"
                        advice = "Consider nutrient-dense meals and strength training to build lean muscle mass."
                    elif bmi < 24.9:
                        category = "Normal Weight"
                        status_class = "success"
                        advice = "Great job! Maintain your current balanced diet and active lifestyle."
                    elif bmi < 29.9:
                        category = "Overweight"
                        status_class = "warning"
                        advice = "Aim for 150 minutes of moderate cardiovascular activity weekly and a calorie-controlled diet."
                    else:
                        category = "Obese"
                        status_class = "danger"
                        advice = "Consult a healthcare provider or nutritionist for a structured weight management roadmap."

                    # Ideal weight range (BMI 18.5 - 24.9)
                    ideal_min = round(18.5 * (height * height), 1)
                    ideal_max = round(24.9 * (height * height), 1)

                    # Basal Metabolic Rate (Mifflin-St Jeor)
                    if gender == "male":
                        bmr = round(10 * weight + 6.25 * (height * 100) - 5 * age + 5)
                    else:
                        bmr = round(10 * weight + 6.25 * (height * 100) - 5 * age - 161)

                    bmi_data = {
                        "bmi": bmi,
                        "category": category,
                        "status_class": status_class,
                        "ideal_range": f"{ideal_min} - {ideal_max} kg",
                        "bmr": bmr,
                        "advice": advice
                    }
            except Exception:
                pass

        # 2. Cardiovascular Risk Screener
        elif calc_type == "cardiac":
            try:
                age = int(request.form.get("cardiac_age", 40))
                systolic = int(request.form.get("systolic_bp", 120))
                heart_rate = int(request.form.get("heart_rate", 72))
                smoker = request.form.get("smoker") == "yes"
                active = request.form.get("active") == "yes"

                risk_points = 0
                if age > 45: risk_points += 2
                if age > 60: risk_points += 2
                if systolic > 130: risk_points += 2
                if systolic > 140: risk_points += 3
                if heart_rate > 85: risk_points += 1
                if smoker: risk_points += 3
                if not active: risk_points += 2

                if risk_points <= 2:
                    level = "Low Risk"
                    color = "success"
                    note = "Your cardiovascular profile appears healthy. Continue regular cardio exercise and balanced nutrition."
                elif risk_points <= 5:
                    level = "Moderate Risk"
                    color = "warning"
                    note = "Moderate risk factors noted. Monitor blood pressure periodically and incorporate daily aerobic walking."
                else:
                    level = "Elevated Risk"
                    color = "danger"
                    note = "Multiple cardiac risk factors identified. Schedule a routine cardiovascular checkup with a physician."

                cardiac_data = {
                    "level": level,
                    "color": color,
                    "score": risk_points,
                    "max_score": 12,
                    "note": note
                }
            except Exception:
                pass

        # 3. Water Intake Planner
        elif calc_type == "hydration":
            try:
                weight_kg = float(request.form.get("water_weight", 65))
                activity_mins = int(request.form.get("water_activity", 30))
                climate = request.form.get("water_climate", "moderate")

                # Baseline: 35ml per kg
                intake_liters = (weight_kg * 0.035) + (activity_mins / 30.0 * 0.35)
                if climate == "hot":
                    intake_liters += 0.5
                
                glasses = round(intake_liters / 0.25)  # 250ml glass
                water_data = {
                    "liters": round(intake_liters, 2),
                    "glasses": glasses,
                    "tip": "Spread your intake evenly throughout the day rather than drinking large quantities at once."
                }
            except Exception:
                pass

    return render_template("vitals.html", bmi=bmi_data, cardiac=cardiac_data, water=water_data)

@app.route("/dashboard")
def dashboard():
    stats = get_analytics_stats()
    records = get_history(limit=50)
    return render_template("dashboard.html", stats=stats, records=records)

@app.route("/emergency")
def emergency_route():
    return render_template("emergency.html")

@app.route("/report/<int:record_id>")
def report_route(record_id):
    records = get_history(limit=200)
    matched = next((r for r in records if r["id"] == record_id), None)
    if not matched:
        return "Record not found", 404
    html = generate_patient_summary_html(matched)
    return Response(html, mimetype="text/html")

@app.route("/api/analytics")
def api_analytics():
    return jsonify(get_analytics_stats())

@app.route("/api/export-csv")
def api_export_csv():
    records = get_history(limit=500)
    csv_path = export_history_to_csv(records)
    return send_file(csv_path, as_attachment=True, download_name="pulseai_consultations.csv")

@app.route("/api/seed-demo", methods=["POST"])
def api_seed_demo():
    seed_demo_data()
    return redirect(url_for("dashboard"))

@app.route("/api/delete/<int:record_id>", methods=["POST"])
def api_delete(record_id):
    delete_analysis(record_id)
    return redirect(url_for("dashboard"))

@app.route("/api/clear-all", methods=["POST"])
def api_clear_all():
    clear_all()
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
