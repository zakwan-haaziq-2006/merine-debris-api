import os

# Keep thread counts low for constrained CPU environments (e.g. free-tier hosting)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_AUTOUPDATE"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

import threading
import io
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image


def _load_env_file():
    for candidate in [".env", "app/.env", os.path.join(os.path.dirname(__file__), ".env")]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


_load_env_file()

ONNX_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
PT_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
MODEL_PATH = ONNX_PATH if os.path.exists(ONNX_PATH) else (PT_PATH if os.path.exists(PT_PATH) else "best.pt")

INFERENCE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CRITICAL_ANOMALIES = {"human", "aircraft", "ship"}
HARDWARE_AND_RIGGING = {"Chain", "Hook", "Propeller", "Valve"}
CONSUMER_PLASTICS_WASTE = {"Bottle", "Can", "Drink-carton", "Shampoo-bottle", "Standing-bottle", "Tire"}

app = FastAPI(
    title="Marine Debris & Sonar Anomaly Detection API",
    description="Acoustic sonar object detection (YOLOv8) and AI-powered survey analysis (Gemini)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(MODEL_PATH)

import torch
torch.set_num_threads(1)

langchain_llm = None
if GEMINI_API_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY,
            max_output_tokens=800,
            max_retries=0,
            timeout=10
        )
    except Exception as e:
        print(f"[WARN] Gemini setup failed: {e}")
        langchain_llm = None


@app.on_event("startup")
def warmup():
    def _run_warmup():
        try:
            import numpy as np
            dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
            model.predict(dummy, imgsz=INFERENCE_SIZE, device="cpu", verbose=False)
        except Exception:
            pass
    threading.Thread(target=_run_warmup, daemon=True).start()


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Marine Debris & Sonar Anomaly Detection API",
        "version": "2.1.0",
        "model_format": "ONNX" if MODEL_PATH.endswith(".onnx") else "PyTorch",
        "inference_size": INFERENCE_SIZE,
        "gemini_configured": langchain_llm is not None,
        "docs_url": "/docs"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_SIZE, device="cpu", verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        box_area = (x2 - x1) * (y2 - y1)
        total_area = image.width * image.height
        area_pct = round((box_area / total_area) * 100, 2) if total_area > 0 else 0

        detections.append({
            "class": cls_name,
            "confidence": round(confidence, 3),
            "box": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            },
            "area_percentage": area_pct
        })

    return {"detections": detections}


class SurveyMetadata(BaseModel):
    survey_id: str | None = None
    water_depth_m: float | None = None
    sensor_type: str | None = "Side-Scan / Forward-Looking Sonar"
    coordinates: str | None = None


class ReportRequest(BaseModel):
    detections: list
    location_note: str | None = None
    metadata: SurveyMetadata | None = None


def _aggregate_survey_telemetry(detections: list):
    total = len(detections)
    if total == 0:
        return None

    class_counts = {}
    confidences = []
    anomalies_count = 0
    hardware_count = 0
    plastics_count = 0

    for d in detections:
        cls = d.get("class", "Unknown")
        conf = float(d.get("confidence", 0.0))
        class_counts[cls] = class_counts.get(cls, 0) + 1
        confidences.append(conf)

        if cls in CRITICAL_ANOMALIES:
            anomalies_count += 1
        elif cls in HARDWARE_AND_RIGGING:
            hardware_count += 1
        elif cls in CONSUMER_PLASTICS_WASTE:
            plastics_count += 1

    avg_conf = round(sum(confidences) / total, 3) if confidences else 0.0

    if "human" in class_counts:
        risk_level = "CRITICAL"
        primary_hazard = "Human / Diver in Distress (Immediate SAR Protocol)"
    elif any(c in class_counts for c in ["aircraft", "ship"]):
        risk_level = "CRITICAL"
        primary_hazard = "Submerged Vessel/Aviation Wreckage & Navigational Obstruction"
    elif hardware_count > 0:
        risk_level = "HIGH" if hardware_count >= 2 else "MEDIUM"
        primary_hazard = "Subsea Rigging / Heavy Hardware Entanglement & Vessel Snag Hazard"
    elif plastics_count > 4:
        risk_level = "MEDIUM"
        primary_hazard = "High-Density Anthropogenic Debris Field & Benthic Plastic Smothering"
    else:
        risk_level = "LOW"
        primary_hazard = "Isolated Anthropogenic Waste"

    return {
        "total_detections": total,
        "class_counts": class_counts,
        "avg_confidence": avg_conf,
        "risk_level": risk_level,
        "primary_hazard": primary_hazard,
        "categories": {
            "critical_anomalies": anomalies_count,
            "subsea_hardware": hardware_count,
            "plastics_and_debris": plastics_count
        }
    }


@app.post("/report")
async def generate_report(request: ReportRequest):
    telemetry = _aggregate_survey_telemetry(request.detections)
    location_text = request.location_note or "General Coastal Survey Zone"
    meta_info = request.metadata or SurveyMetadata()

    if not telemetry:
        clean_report = f"""# Acoustic Sonar Survey Report
**Location**: {location_text}
**Status**: CLEAR / NO TARGETS DETECTED
**Threat Level**: LOW

### 1. Survey Overview
Acoustic inspection of the specified survey sector returned zero high-confidence debris targets or seabed anomalies.

### 2. Conclusion & Recommendation
No navigational or environmental hazards detected. Normal maritime transit may continue without intervention."""
        return {
            "report": clean_report,
            "risk_level": "LOW",
            "summary": "Survey clear: No marine debris or acoustic anomalies detected.",
            "primary_hazard": "None",
            "statistics": {
                "total_detections": 0,
                "avg_confidence": 1.0,
                "categories": {"critical_anomalies": 0, "subsea_hardware": 0, "plastics_and_debris": 0}
            },
            "priority_actions": ["Log sector as clear in maritime registry", "Proceed with routine monitoring schedule"]
        }

    item_lines = [f"  - {count}x '{cls}'" for cls, count in telemetry["class_counts"].items()]
    detections_summary = "\n".join(item_lines)

    prompt = f"""You are a Lead Marine Acoustic Surveyor conducting subsea survey evaluations using sonar imagery.

Provide a technically accurate, professional Marine Survey & Hazard Assessment Report based strictly on the verified acoustic detections below.

Location/Sector: {location_text}
Sensor Type: {meta_info.sensor_type}
Total Detected Objects: {telemetry["total_detections"]}
Average Acoustic Confidence: {int(telemetry["avg_confidence"] * 100)}%
Target Breakdown:
{detections_summary}
Preliminary Baseline Risk Rating: {telemetry["risk_level"]} ({telemetry["primary_hazard"]})

Write a structured markdown report with these sections:
# Marine Sonar Survey & Environmental Hazard Assessment
## 1. Executive Summary
## 2. Acoustic Target Inventory & Classification
## 3. Threat Assessment & Navigational Impact
## 4. Operational Remediation & Action Protocol

Keep it rigorous, technical, and actionable (max 350 words). No conversational filler."""

    report_markdown = None
    if langchain_llm is not None:
        try:
            res = await langchain_llm.ainvoke(prompt)
            text = res.content if hasattr(res, "content") else str(res)
            if text and text.strip():
                report_markdown = text.strip()
        except Exception as e:
            print(f"[WARN] Gemini report generation failed ({e}). Using fallback report.")

    if not report_markdown:
        report_markdown = f"""# Marine Sonar Survey & Environmental Hazard Assessment
**Sector**: {location_text} | **Sensor**: {meta_info.sensor_type}
**Assessment Risk Level**: [{telemetry["risk_level"]}]

## 1. Executive Summary
Acoustic sonar survey across {location_text} identified {telemetry["total_detections"]} target return(s) with average confidence {int(telemetry["avg_confidence"] * 100)}%. Preliminary hazard rating: {telemetry["risk_level"]} due to {telemetry["primary_hazard"]}.

## 2. Acoustic Target Inventory & Classification
{detections_summary}

## 3. Threat Assessment & Navigational Impact
Primary Hazard: {telemetry["primary_hazard"]}. Subsea debris poses entanglement and snag risk to vessels and equipment.

## 4. Operational Remediation & Action Protocol
1. Transmit hazard bulletin to harbor master and coastal patrol ({location_text}).
2. Deploy ROV for visual confirmation.
3. Schedule mechanical recovery for identified heavy targets."""

    summary_match = re.search(r"## 1\. Executive Summary\s+([^\n#]+)", report_markdown)
    exec_summary = summary_match.group(1).strip() if summary_match else f"Identified {telemetry['total_detections']} objects in {location_text}, risk: {telemetry['risk_level']}."

    actions = []
    actions_section = re.search(r"## 4\..*?\n([\s\S]*?)(?:$|#)", report_markdown)
    if actions_section:
        raw_actions = re.findall(r"(?:^|\n)\s*(?:\d+\.|\-|\*)\s*(.+)", actions_section.group(1))
        actions = [a.strip() for a in raw_actions[:5] if a.strip()]
    if not actions:
        actions = [
            f"Transmit hazard bulletin to harbor master ({location_text})",
            "Deploy ROV for visual confirmation",
            "Schedule mechanical recovery for heavy targets"
        ]

    return {
        "report": report_markdown,
        "risk_level": telemetry["risk_level"],
        "summary": exec_summary,
        "primary_hazard": telemetry["primary_hazard"],
        "statistics": {
            "total_detections": telemetry["total_detections"],
            "avg_confidence": telemetry["avg_confidence"],
            "class_counts": telemetry["class_counts"],
            "categories": telemetry["categories"]
        },
        "priority_actions": actions
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_interface():
    html_file = os.path.join(os.path.dirname(__file__), "demo.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>No demo.html found. /docs is available for testing the API directly.</h1>"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)