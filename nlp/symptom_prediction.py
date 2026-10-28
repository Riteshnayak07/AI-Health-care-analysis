"""
PulseAI - NLP & Clinical Symptom Engine
Provides medical taxonomy extraction, typo-tolerant symptom matching,
TF-IDF + ML inference, triage urgency evaluation, and specialist recommendations.
"""
import os
import re
import pickle
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "model" / "healthcare_model.pkl"

# Comprehensive symptom dictionary with medical synonyms and common misspellings
SYMPTOM_LEXICON = {
    # General / Systemic
    "fever": ["fever", "high temperature", "temperature", "feverish", "fewer", "pyrexia", "hot body", "burning up"],
    "chills": ["chills", "shivering", "shivers", "feeling cold"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhaustion", "weakness", "lethargy", "low energy", "fatigued"],
    "body pain": ["body pain", "body ache", "muscle pain", "muscle ache", "generalized ache", "myalgia"],
    "joint pain": ["joint pain", "joint ache", "pain in joints", "arthritis", "stiff joints"],
    "sweating": ["sweating", "cold sweats", "night sweats", "excessive sweat"],

    # Head & Neurological
    "headache": ["headache", "head ache", "head pain", "migraine", "head throbbing", "temple pain", "cephalea"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "lightheadedness", "vertigo", "fainting feeling", "spinning"],
    "light sensitivity": ["light sensitivity", "photophobia", "sensitive to light"],

    # Respiratory & ENT
    "cough": ["cough", "coughing", "dry cough", "wet cough", "hacking cough", "khansi"],
    "cold": ["cold", "common cold", "head cold"],
    "runny nose": ["runny nose", "running nose", "watery nose", "rhinorrhea"],
    "nasal congestion": ["nasal congestion", "blocked nose", "stuffy nose", "nose block"],
    "sneezing": ["sneezing", "sneeze", "continuous sneezing"],
    "sore throat": ["sore throat", "throat pain", "scratchy throat", "irritated throat", "pharyngitis"],
    "shortness of breath": ["shortness of breath", "breathlessness", "difficulty breathing", "dyspnea", "breathing trouble", "breathless"],
    "wheezing": ["wheezing", "whistling breath", "raspy breathing"],
    "chest congestion": ["chest congestion", "mucus", "phlegm", "sputum", "lung congestion"],

    # Cardiovascular & Urgent
    "chest pain": ["chest pain", "pain in chest", "chest tightness", "chest pressure", "heavy chest", "angina"],
    "palpitations": ["palpitations", "rapid heartbeat", "racing heart", "pounding heart", "irregular pulse"],

    # Gastrointestinal
    "nausea": ["nausea", "nauseous", "feeling sick", "queasy"],
    "vomiting": ["vomiting", "vomit", "throwing up", "emesis", "puking"],
    "diarrhea": ["diarrhea", "loose motion", "loose motions", "watery stool", "frequent motions"],
    "stomach pain": ["stomach pain", "abdominal pain", "tummy ache", "belly pain", "stomach cramps", "cramping"],
    "heartburn": ["heartburn", "acid reflux", "chest burn", "sour burps", "indigestion", "gerd", "acidity"],

    # Dermatological
    "rash": ["rash", "skin rash", "red spots", "red patches", "hives", "skin breakout"],
    "itching": ["itching", "itchy", "itchy skin", "pruritus", "scratching"],
    "skin redness": ["skin redness", "erythema", "inflamed skin"],

    # Urinary
    "burning urination": ["burning urination", "painful urination", "dysuria", "stinging when peeing", "burn when peeing"],
    "frequent urination": ["frequent urination", "frequent pee", "peeing often", "constant urge to urinate"]
}

# Clinical metadata, triage protocols, and specialist recommendations
CLINICAL_KNOWLEDGE = {
    "Influenza (Flu)": {
        "description": "A contagious respiratory viral infection affecting the nose, throat, and lungs.",
        "urgency": "Moderate",
        "specialist": "General Physician / Pulmonologist",
        "home_care": [
            "Rest thoroughly and drink warm fluids (soups, herbal teas, water).",
            "Use steam inhalation to soothe airway congestion.",
            "Monitor body temperature at regular 4-6 hour intervals."
        ],
        "red_flags": "Difficulty breathing, persistent chest pain, bluish lips, or confusion."
    },
    "Common Cold": {
        "description": "A benign, self-limiting viral upper-respiratory tract infection.",
        "urgency": "Mild",
        "specialist": "General Practitioner / Primary Care",
        "home_care": [
            "Maintain high hydration with saline gargles for throat soothing.",
            "Use nasal saline drops or a humidifier for nasal clearance.",
            "Allow ample sleep and light nutritious meals."
        ],
        "red_flags": "Symptoms lasting over 10 days, ear pain, or sudden high fever spikes."
    },
    "Viral Fever": {
        "description": "Systemic viral illness characterized by elevated core temperature and myalgia.",
        "urgency": "Moderate",
        "specialist": "Internal Medicine / General Physician",
        "home_care": [
            "Stay well hydrated with electrolyte solutions or coconut water.",
            "Use cold compress on forehead if temperature exceeds 101°F.",
            "Avoid strenuous physical exertion."
        ],
        "red_flags": "Fever persisting over 3 days, stiff neck, severe vomiting, or petechial rash."
    },
    "Respiratory Infection / Bronchitis": {
        "description": "Inflammation or infection of the bronchial tubes and lower respiratory pathways.",
        "urgency": "Moderate",
        "specialist": "Pulmonologist / Chest Physician",
        "home_care": [
            "Practice deep breathing exercises and maintain good room ventilation.",
            "Warm fluids and honey-lemon tea can ease severe coughing fits.",
            "Avoid exposure to tobacco smoke, cold air, and fine dust."
        ],
        "red_flags": "Gasping for breath, blood in cough/sputum, or severe wheezing."
    },
    "Possible Cardiac Concern (Urgent)": {
        "description": "Potentially critical cardiovascular episode requiring rapid clinical evaluation.",
        "urgency": "Urgent",
        "specialist": "Cardiologist / Emergency Medicine (Immediate)",
        "home_care": [
            "Stop all physical activity immediately and sit comfortably upright.",
            "Loosen tight clothing around chest, neck, and waist.",
            "Have someone call for an emergency ambulance (108 / 112 / 911) right away."
        ],
        "red_flags": "Crushing chest pressure, pain radiating to left arm/jaw, diaphoresis, or collapse."
    },
    "Gastroenteritis (Stomach Infection)": {
        "description": "Inflammation of digestive lining, commonly from food or waterborne pathogens.",
        "urgency": "Moderate",
        "specialist": "Gastroenterologist / General Physician",
        "home_care": [
            "Rehydrate continuously with Oral Rehydration Salts (ORS) in small sips.",
            "Follow the BRAT diet: Bananas, Rice, Applesauce, and Toast.",
            "Avoid dairy, greasy, caffeinated, and spicy foods."
        ],
        "red_flags": "Inability to keep liquids down for 24h, blood in stool, severe dehydration, or dark urine."
    },
    "Migraine / Tension Headache": {
        "description": "Neurovascular headache disorder often accompanied by sensory disturbances.",
        "urgency": "Moderate",
        "specialist": "Neurologist / General Physician",
        "home_care": [
            "Rest in a quiet, cool, dimly lit room with eyes closed.",
            "Apply a cool or lukewarm gel pack across forehead or nape of neck.",
            "Practice gentle neck stretches and maintain consistent hydration."
        ],
        "red_flags": "Sudden 'thunderclap' explosive headache, vision loss, numbness, or difficulty speaking."
    },
    "Allergic Reaction / Dermatitis": {
        "description": "Immune hypersensitivity reaction producing cutaneous or mucosal inflammation.",
        "urgency": "Mild",
        "specialist": "Dermatologist / Allergist",
        "home_care": [
            "Apply cool damp compresses and hypoallergenic calamine lotion to soothe itch.",
            "Avoid scratching to prevent secondary bacterial skin infections.",
            "Identify and remove recent environmental, chemical, or dietary triggers."
        ],
        "red_flags": "Facial swelling, tongue swelling, or difficulty breathing (Anaphylaxis - Call Emergency)."
    },
    "Acid Reflux / GERD": {
        "description": "Retrograde flow of gastric acid into the esophagus causing mucosal irritation.",
        "urgency": "Mild",
        "specialist": "Gastroenterologist",
        "home_care": [
            "Eat smaller, more frequent meals; do not lie down within 2-3 hours after eating.",
            "Elevate head of bed by 6 inches while sleeping.",
            "Reduce intake of citrus, carbonated beverages, chocolate, and heavy fried foods."
        ],
        "red_flags": "Difficulty swallowing (dysphagia), vomiting blood, or unremitting chest burning."
    },
    "Hypertension / Stress": {
        "description": "Elevated vascular arterial pressure or autonomic stress-induced cardiovascular strain.",
        "urgency": "Moderate",
        "specialist": "Cardiologist / Internal Medicine",
        "home_care": [
            "Sit calmly, perform 4-7-8 rhythmic diaphragmatic breathing.",
            "Hydrate with plain water and reduce dietary sodium intake.",
            "Log blood pressure readings at rest in the morning and evening."
        ],
        "red_flags": "BP over 180/120 mmHg, sudden blurred vision, severe headache, or chest pain."
    },
    "Urinary Tract Infection": {
        "description": "Bacterial colonization of the urinary bladder or urethra.",
        "urgency": "Moderate",
        "specialist": "Urologist / General Physician",
        "home_care": [
            "Drink plenty of water to help flush bacteria out of the urinary tract.",
            "Urinate frequently without holding it in.",
            "Avoid caffeine, alcohol, and carbonated beverages."
        ],
        "red_flags": "High fever with chills, severe lower back/flank pain, or blood in urine (hematuria)."
    }
}

_cached_model = None

def get_ml_model():
    global _cached_model
    if _cached_model is None and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _cached_model = pickle.load(f)
        except Exception:
            _cached_model = None
    return _cached_model

def extract_symptoms(text: str):
    """
    Extracts canonical symptoms from free-form text using phrase dictionary
    and multi-word priority matching.
    """
    clean = text.lower()
    found = []
    
    # Sort phrases by descending length so multi-word phrases match first
    for canonical, variations in SYMPTOM_LEXICON.items():
        matched = False
        for v in sorted(variations, key=lambda x: -len(x)):
            # Word boundary check or substring for phrases
            pattern = r'\b' + re.escape(v) + r'\b' if len(v) <= 5 else re.escape(v)
            if re.search(pattern, clean):
                found.append(canonical)
                matched = True
                break
    return list(dict.fromkeys(found))

def predict_condition(text: str, detected_symptoms: list = None, severity: str = "moderate", duration_days: int = 2):
    """
    Ensemble diagnostic inference combining NLP rule safety checks
    with TF-IDF ML model prediction probabilities.
    """
    if detected_symptoms is None:
        detected_symptoms = extract_symptoms(text)
    
    s_set = set(detected_symptoms)
    
    # 1. Critical Emergency Override: Cardiac / Acute respiratory distress
    if "chest pain" in s_set or ("shortness of breath" in s_set and ("chest pain" in s_set or "dizziness" in s_set)):
        info = CLINICAL_KNOWLEDGE["Possible Cardiac Concern (Urgent)"]
        return {
            "prediction": "Possible Cardiac Concern (Urgent)",
            "confidence": 96.5,
            "urgency": "Urgent",
            "specialist": info["specialist"],
            "description": info["description"],
            "home_care": info["home_care"],
            "red_flags": info["red_flags"],
            "secondary_matches": ["Respiratory Infection / Bronchitis", "Hypertension / Stress"]
        }

    # 2. ML Model Inference
    model = get_ml_model()
    ml_pred = None
    ml_confidence = 0.0
    secondary = []
    
    if model and text.strip():
        try:
            probs = model.predict_proba([text])[0]
            classes = model.classes_
            sorted_indices = probs.argsort()[::-1]
            top_idx = sorted_indices[0]
            ml_pred = classes[top_idx]
            ml_confidence = float(probs[top_idx]) * 100.0
            
            # Secondary differential matches (next top 2 classes)
            for idx in sorted_indices[1:3]:
                if probs[idx] > 0.08:
                    secondary.append(classes[idx])
        except Exception:
            ml_pred = None

    # 3. Rule-based heuristic scoring fallback / corroboration
    scores = {
        "Common Cold": len(s_set & {"cold", "runny nose", "cough", "sneezing", "nasal congestion", "sore throat"}),
        "Influenza (Flu)": len(s_set & {"fever", "cough", "body pain", "fatigue", "chills", "headache", "sore throat"}),
        "Viral Fever": len(s_set & {"fever", "headache", "body pain", "fatigue", "sweating", "chills"}),
        "Gastroenteritis (Stomach Infection)": len(s_set & {"vomiting", "nausea", "diarrhea", "stomach pain"}),
        "Migraine / Tension Headache": len(s_set & {"headache", "dizziness", "nausea", "light sensitivity"}),
        "Allergic Reaction / Dermatitis": len(s_set & {"rash", "itching", "skin redness", "runny nose", "sneezing"}),
        "Respiratory Infection / Bronchitis": len(s_set & {"cough", "sore throat", "wheezing", "chest congestion", "shortness of breath"}),
        "Acid Reflux / GERD": len(s_set & {"heartburn", "stomach pain", "nausea"}),
        "Hypertension / Stress": len(s_set & {"headache", "dizziness", "palpitations"}),
        "Urinary Tract Infection": len(s_set & {"burning urination", "frequent urination", "stomach pain", "fever"})
    }

    best_rule = max(scores, key=scores.get)
    best_rule_score = scores[best_rule]

    if best_rule_score == 0 and not ml_pred:
        return {
            "prediction": "Insufficient Information",
            "confidence": 0.0,
            "urgency": "Mild",
            "specialist": "General Physician",
            "description": "The symptoms provided do not match recognized patterns. Please describe your symptoms in more detail or consult a doctor.",
            "home_care": ["Rest well and monitor if new symptoms develop.", "Stay hydrated."],
            "red_flags": "Seek clinical evaluation if you experience high fever, severe pain, or difficulty breathing.",
            "secondary_matches": []
        }

    # Decide between ML prediction and Rule-based prediction
    if ml_pred and ml_confidence >= 35.0:
        prediction = ml_pred
        confidence = min(96.0, max(52.0, ml_confidence))
    else:
        prediction = best_rule
        confidence = min(92.0, 50.0 + (best_rule_score * 12.0))

    info = CLINICAL_KNOWLEDGE.get(prediction, CLINICAL_KNOWLEDGE["Common Cold"])
    urgency = info["urgency"]

    # Adjust urgency based on severity slider and duration
    if severity == "severe":
        if urgency == "Moderate":
            urgency = "Urgent"
        elif urgency == "Mild":
            urgency = "Moderate"
    elif duration_days >= 7 and urgency == "Mild":
        urgency = "Moderate"

    if not secondary:
        secondary = [k for k, v in sorted(scores.items(), key=lambda x: -x[1]) if k != prediction and v > 0][:2]

    return {
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "urgency": urgency,
        "specialist": info["specialist"],
        "description": info["description"],
        "home_care": info["home_care"],
        "red_flags": info["red_flags"],
        "secondary_matches": secondary
    }
