"""
PulseAI - Medical Report Generator
Exports diagnostic consultation records to formatted printable reports or CSV.
Gracefully handles environments with or without ReportLab.
"""
import os
import csv
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def export_history_to_csv(history, filepath=None):
    if not filepath:
        filepath = BASE / "pulseai_consultation_history.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp", "Analysis Type", "Input Symptoms", "Detected Symptoms", "Prediction", "Confidence %", "Triage Urgency"])
        for r in history:
            writer.writerow([
                r.get("id", ""),
                r.get("created_at", ""),
                r.get("analysis_type", "symptom"),
                r.get("text", ""),
                r.get("symptoms", ""),
                r.get("prediction", ""),
                f"{r.get('confidence', 0):.1f}",
                r.get("urgency", "Mild")
            ])
    return str(filepath)

def generate_patient_summary_html(record):
    """
    Generates a standalone, print-friendly clinical summary HTML sheet.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Clinical Consultation Summary - PulseAI</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 40px; color: #1e293b; line-height: 1.5; }}
  .header {{ border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }}
  .title {{ font-size: 24px; font-weight: bold; color: #0f172a; margin: 0; }}
  .subtitle {{ color: #64748b; font-size: 13px; margin: 4px 0 0 0; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }}
  .badge-Urgent {{ background: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }}
  .badge-Moderate {{ background: #fef3c7; color: #b45309; border: 1px solid #fcd34d; }}
  .badge-Mild {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
  .box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .box-title {{ font-weight: bold; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; margin-bottom: 8px; }}
  .prediction {{ font-size: 20px; font-weight: bold; color: #0369a1; }}
  .footer {{ margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 12px; font-size: 11px; color: #94a3b8; text-align: center; }}
  @media print {{
    .no-print {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1 class="title">PulseAI Health Summary</h1>
    <p class="subtitle">AI-Assisted Diagnostic Triage Report &bull; Reference #{record.get('id', 'N/A')}</p>
  </div>
  <div style="text-align: right;">
    <div style="font-size: 12px; color: #64748b;">Generated Date</div>
    <div style="font-weight: 600;">{record.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</div>
  </div>
</div>

<div class="box">
  <div class="box-title">Patient Stated Symptoms</div>
  <p style="margin: 0; font-size: 15px;">&ldquo;{record.get('text', 'No text entered')}&rdquo;</p>
  <div style="margin-top: 8px; font-size: 13px; color: #475569;">
    <b>Identified Clinical Symptoms:</b> {record.get('symptoms', 'None detected')}
  </div>
</div>

<div class="box" style="display: flex; justify-content: space-between; align-items: center;">
  <div>
    <div class="box-title">Primary Clinical Assessment</div>
    <div class="prediction">{record.get('prediction', 'Assessment Inconclusive')}</div>
  </div>
  <div style="text-align: right;">
    <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Confidence Metric</div>
    <div style="font-size: 20px; font-weight: bold; color: #0284c7;">{record.get('confidence', 0):.1f}%</div>
    <span class="badge badge-{record.get('urgency', 'Mild')}">{record.get('urgency', 'Mild')} Priority</span>
  </div>
</div>

<div class="box">
  <div class="box-title">Recommended Specialist</div>
  <div style="font-weight: 600; color: #1e293b;">{record.get('specialist', 'General Physician')}</div>
  <p style="margin: 4px 0 0; font-size: 13px; color: #64748b;">{record.get('description', '')}</p>
</div>

<div class="footer">
  <strong>Educational Prototype Disclaimer:</strong> PulseAI is an educational software system and does not replace certified professional medical advice, clinical diagnosis, or hospital emergency triage.
</div>
</body>
</html>"""
    return html
