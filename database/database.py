"""
PulseAI - Database Access & Analytics Module
Provides SQLite storage, schema migration, and aggregation metrics for charts.
"""
import sqlite3
import os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "healthcare.db")

def get_db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with get_db() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                symptoms TEXT,
                prediction TEXT,
                confidence REAL,
                urgency TEXT DEFAULT 'Mild',
                analysis_type TEXT DEFAULT 'symptom',
                created_at TEXT
            )
        """)
        # Ensure schema migrations for existing databases
        cols = [c[1] for c in cur.execute("PRAGMA table_info(analyses)").fetchall()]
        if "urgency" not in cols:
            cur.execute("ALTER TABLE analyses ADD COLUMN urgency TEXT DEFAULT 'Mild'")
        if "analysis_type" not in cols:
            cur.execute("ALTER TABLE analyses ADD COLUMN analysis_type TEXT DEFAULT 'symptom'")
        con.commit()

def save_analysis(analysis_type, text, symptoms, prediction, confidence, urgency="Mild"):
    init_db()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO analyses (text, symptoms, prediction, confidence, urgency, analysis_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (text, symptoms, prediction, float(confidence), urgency, analysis_type, now_str))
        con.commit()
        return cur.lastrowid

def get_history(limit=50, analysis_type=None):
    init_db()
    with get_db() as con:
        if analysis_type:
            rows = con.execute(
                "SELECT * FROM analyses WHERE analysis_type = ? ORDER BY id DESC LIMIT ?",
                (analysis_type, limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

def get_analytics_stats():
    init_db()
    with get_db() as con:
        total = con.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        if total == 0:
            return {
                "total": 0,
                "avg_confidence": 0,
                "urgent_count": 0,
                "top_condition": "N/A",
                "conditions": {},
                "urgencies": {"Urgent": 0, "Moderate": 0, "Mild": 0},
                "symptoms_freq": {}
            }

        avg_conf = con.execute("SELECT AVG(confidence) FROM analyses WHERE confidence > 0").fetchone()[0] or 0
        urgent_cnt = con.execute("SELECT COUNT(*) FROM analyses WHERE urgency = 'Urgent'").fetchone()[0]

        # Conditions breakdown
        cond_rows = con.execute("""
            SELECT prediction, COUNT(*) as cnt 
            FROM analyses 
            WHERE prediction != 'Insufficient Information'
            GROUP BY prediction 
            ORDER BY cnt DESC
        """).fetchall()
        conditions = {r["prediction"]: r["cnt"] for r in cond_rows}
        top_condition = cond_rows[0]["prediction"] if cond_rows else "General Assessment"

        # Urgency breakdown
        urg_rows = con.execute("SELECT urgency, COUNT(*) as cnt FROM analyses GROUP BY urgency").fetchall()
        urgencies = {"Urgent": 0, "Moderate": 0, "Mild": 0}
        for r in urg_rows:
            urgencies[r["urgency"] or "Mild"] = r["cnt"]

        # Symptom frequency
        sym_rows = con.execute("SELECT symptoms FROM analyses WHERE symptoms IS NOT NULL AND symptoms != ''").fetchall()
        sym_freq = {}
        for r in sym_rows:
            tokens = [s.strip().lower() for s in r["symptoms"].split(",") if s.strip()]
            for t in tokens:
                sym_freq[t] = sym_freq.get(t, 0) + 1

        sorted_syms = dict(sorted(sym_freq.items(), key=lambda x: -x[1])[:8])

        return {
            "total": total,
            "avg_confidence": round(avg_conf, 1),
            "urgent_count": urgent_cnt,
            "top_condition": top_condition,
            "conditions": conditions,
            "urgencies": urgencies,
            "symptoms_freq": sorted_syms
        }

def delete_analysis(record_id):
    with get_db() as con:
        con.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
        con.commit()

def clear_all():
    with get_db() as con:
        con.execute("DELETE FROM analyses")
        con.commit()

def seed_demo_data():
    samples = [
        ("fever, cough, body ache, tiredness", "fever, cough, body pain, fatigue", "Influenza (Flu)", 88.5, "Moderate", "symptom", "2026-10-27 09:30:15"),
        ("severe chest pain and breathlessness", "chest pain, shortness of breath", "Possible Cardiac Concern (Urgent)", 95.0, "Urgent", "symptom", "2026-10-27 11:15:20"),
        ("runny nose, continuous sneezing, mild headache", "runny nose, sneezing, cough", "Common Cold", 82.0, "Mild", "symptom", "2026-10-27 14:40:02"),
        ("vomiting, loose stools and abdominal cramps", "vomiting, diarrhea, stomach pain", "Gastroenteritis (Stomach Infection)", 90.0, "Moderate", "symptom", "2026-10-28 08:20:10"),
        ("one sided throbbing headache with nausea", "headache, nausea, dizziness", "Migraine / Tension Headache", 85.0, "Moderate", "symptom", "2026-10-28 10:05:44"),
        ("red itchy rash on forearms and neck", "rash, itching", "Allergic Reaction / Dermatitis", 79.5, "Mild", "symptom", "2026-10-28 13:12:30"),
        ("high fever, chills, severe muscle aches", "fever, fatigue, body pain", "Viral Fever", 84.0, "Moderate", "symptom", "2026-10-28 16:30:00"),
        ("burning sensation after food, acid taste", "heartburn, acid reflux", "Acid Reflux / GERD", 86.0, "Mild", "symptom", "2026-10-28 18:22:15"),
    ]
    with get_db() as con:
        for s in samples:
            con.execute("""
                INSERT INTO analyses (text, symptoms, prediction, confidence, urgency, analysis_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, s)
        con.commit()
