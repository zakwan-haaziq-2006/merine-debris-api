import os
# Force 1 thread for all C++/OpenMP/NumPy/ONNX runtimes to eliminate thread-contention on 0.1 CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Prevent runtime AutoUpdate which blocked container startup for 278 seconds
os.environ["YOLO_AUTOUPDATE"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

import threading

import io
import base64
import json
import re

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai

# ---- AUTO-LOAD .ENV IF PRESENT ----
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

# Prefer ultra-fast compiled ONNX model over PyTorch weights
ONNX_PATH = os.path.join(os.path.dirname(__file__), "best.onnx")
PT_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
MODEL_PATH = ONNX_PATH if os.path.exists(ONNX_PATH) else (PT_PATH if os.path.exists(PT_PATH) else "best.onnx")

INFERENCE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Domain classification categories for the 13 sonar classes
CRITICAL_ANOMALIES = {"human", "aircraft", "ship"}
HARDWARE_AND_RIGGING = {"Chain", "Hook", "Propeller", "Valve"}
CONSUMER_PLASTICS_WASTE = {"Bottle", "Can", "Drink-carton", "Shampoo-bottle", "Standing-bottle", "Tire"}

# ---- SETUP ----
app = FastAPI(
    title="Marine Debris & Sonar Anomaly Detection API",
    description="Acoustic sonar object detection (YOLOv8 ONNX) and AI-powered survey analysis (Gemini)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (supports both .onnx and .pt)
model = YOLO(MODEL_PATH)

import torch
# Render free tier has 0.1 CPU core. Limiting to 1 thread eliminates massive thread contention.
torch.set_num_threads(1)

# LangChain Gemini Setup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

if GEMINI_API_KEY:
    langchain_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=800,
        max_retries=0,
        timeout=10
    )
else:
    langchain_llm = None


@app.on_event("startup")
def warmup():
    """Warms up ONNX in a background daemon thread so Uvicorn binds immediately and passes health checks."""
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
        "docs_url": "/docs",
        "demo_url": "/demo"
    }


# ---------------------------------------------------------------------------
# ENDPOINT 1: /detect
# ---------------------------------------------------------------------------
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Accepts an uploaded sonar image, runs YOLOv8 detection at 416px,
    and returns only the detected objects and bounding box telemetry.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Fast inference at 416px on CPU (ONNX optimized)
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_SIZE, device="cpu", verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        
        # Calculate relative bounding box area percentage
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

    return {
        "detections": detections
    }


# ---------------------------------------------------------------------------
# ENDPOINT 2: /report
# ---------------------------------------------------------------------------
class SurveyMetadata(BaseModel):
    survey_id: str | None = None
    water_depth_m: float | None = None
    sensor_type: str | None = "Forward-Looking Sonar (FLS)"
    coordinates: str | None = None


class ReportRequest(BaseModel):
    detections: list
    location_note: str | None = None
    metadata: SurveyMetadata | None = None


def _aggregate_survey_telemetry(detections: list):
    """Computes statistical and domain distribution metrics from raw detections."""
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

    # Determine baseline risk classification
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
    """
    Takes detection telemetry from /detect and generates an elaborated,
    highly accurate, professional maritime survey report using Gemini / LangChain.
    """
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

### 2. Acoustic Analysis
The seabed profile shows continuous natural acoustic backscatter without man-made acoustic shadows, metallic reflections, or synthetic debris silhouettes.

### 3. Conclusion & Recommendation
No navigational or environmental hazards detected. Normal maritime transit and benthic habitat operations may continue without intervention."""
        return {
            "report": clean_report,
            "report_markdown": clean_report,
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

    # Format telemetry breakdown for Gemini prompt
    item_lines = [
        f"  - {count}x '{cls}'"
        for cls, count in telemetry["class_counts"].items()
    ]
    detections_summary = "\n".join(item_lines)

    # Elaborated domain prompt
    prompt = f"""You are a Lead Marine Acoustic Surveyor and Oceanographic Environmental Officer conducting subsea survey evaluations using Forward-Looking Sonar (FLS) imagery.

Provide an elaborated, technically accurate, and highly professional Marine Survey & Hazard Assessment Report based strictly on the verified acoustic detections below.

=== MISSION & SENSOR TELEMETRY ===
- Location/Sector: {location_text}
- Sensor Type: {meta_info.sensor_type}
- Total Detected Objects: {telemetry["total_detections"]}
- Average Acoustic Confidence: {int(telemetry["avg_confidence"] * 100)}%
- Target Breakdown:
{detections_summary}
- Categorical Distribution:
  * High-Consequence Anomalies: {telemetry["categories"]["critical_anomalies"]}
  * Subsea Rigging / Hardware: {telemetry["categories"]["subsea_hardware"]}
  * Synthetic Polymers & General Waste: {telemetry["categories"]["plastics_and_debris"]}
- Preliminary Baseline Risk Rating: {telemetry["risk_level"]} ({telemetry["primary_hazard"]})

=== REPORT REQUIREMENTS ===
Generate a comprehensive, structured technical survey document in professional GitHub-Flavored Markdown. 
Structure the report into the following exact sections:

# Marine Sonar Survey & Environmental Hazard Assessment
**Sector**: {location_text} | **Sensor**: {meta_info.sensor_type}  
**Assessment Risk Level**: [{telemetry["risk_level"]}]

## 1. Executive Summary
Provide an authoritative 2-3 paragraph operational synthesis. Summarize the acoustic scan, describe the concentration and nature of target returns, and state the immediate operational posture required.

## 2. Acoustic Target Inventory & Classification
Break down the detections by domain category:
- **Maritime Anomalies & Structural Wreckage** (e.g. aircraft, ship, human): Discuss dimensions, structural integrity, potential historical/salvage significance, or urgent SAR (Search and Rescue) considerations.
- **Subsea Machinery & Heavy Hardware** (e.g. chain, propeller, hook, valve): Discuss snag/entanglement hazards to submarine cables, commercial bottom-trawling nets, and vessel propulsion systems.
- **Anthropogenic Debris & Polymers** (e.g. tires, bottles, cartons, cans): Assess benthic smothering, chemical leaching, microplastic degradation timelines, and marine fauna toxicity.

## 3. Threat Assessment & Navigational Impact
- **Hydrodynamic & Navigational Risks**: Evaluate water column clearance, shallow hazards to shallow-draft vessels, diver safety, and ROV navigation.
- **Ecosystem & Benthic Toxicity**: Detail the ecological fallout if targets remain unrecovered.

## 4. Operational Remediation & Action Protocol
Provide a numbered, prioritized action protocol for port authorities, coast guard, or salvage teams (e.g. Priority Level 1 immediate notices to mariners, ROV deployment, diver dispatch, targeted recovery crane operations).

Keep the tone rigorous, technical, concise, and actionable (maximum 350-400 words total). Do NOT generate conversational filler or introductory greetings."""

    report_markdown = None

    # Attempt LangChain Gemini generation if configured
    if langchain_llm is not None:
        try:
            prompt_template = PromptTemplate.from_template("{prompt_text}")
            chain = prompt_template | langchain_llm | StrOutputParser()
            res = await chain.ainvoke({"prompt_text": prompt})
            if res and res.strip():
                report_markdown = res.strip()
        except Exception as e:
            # When Gemini hits quota (429) or is unreachable, gracefully fall back to telemetry report
            print(f"[WARN] Gemini report generation failed ({e}). Using telemetry-based acoustic survey report.")

    # High-reliability fallback report (ensures /report NEVER crashes with 500 error)
    if not report_markdown:
        report_markdown = f"""# Marine Sonar Survey & Environmental Hazard Assessment
**Sector**: {location_text} | **Sensor**: {meta_info.sensor_type}  
**Assessment Risk Level**: [{telemetry["risk_level"]}]

## 1. Executive Summary
Acoustic sonar survey across {location_text} identified a total of {telemetry["total_detections"]} target return(s) with an average acoustic confidence of {int(telemetry["avg_confidence"] * 100)}%. Preliminary hazard rating is classified as {telemetry["risk_level"]} due to {telemetry["primary_hazard"]}.

## 2. Acoustic Target Inventory & Classification
Target breakdown:
{detections_summary}

- **Critical Maritime Anomalies**: {telemetry["categories"]["critical_anomalies"]} target(s) logged.
- **Subsea Rigging & Hardware**: {telemetry["categories"]["subsea_hardware"]} target(s) logged.
- **Anthropogenic Waste & Polymers**: {telemetry["categories"]["plastics_and_debris"]} target(s) logged.

## 3. Threat Assessment & Navigational Impact
- **Primary Hazard**: {telemetry["primary_hazard"]}
- **Navigational Impact**: Subsea debris poses entanglement and snag risks to vessel propulsion systems, commercial nets, and underwater infrastructure. Immediate caution advised for local maritime traffic.

## 4. Operational Remediation & Action Protocol
1. Transmit acoustic hazard bulletin to harbor master and coastal patrol ({location_text}).
2. Deploy Remotely Operated Vehicle (ROV) for optical inspection and target coordinates confirmation.
3. Schedule targeted mechanical recovery operations for identified heavy subsea targets."""

    # Extract 2-line executive summary from the markdown text
    summary_match = re.search(r"## 1\. Executive Summary\s+([^\n#]+)", report_markdown)
    exec_summary = summary_match.group(1).strip() if summary_match else f"Identified {telemetry['total_detections']} objects across survey sector {location_text} with baseline risk rating {telemetry['risk_level']}."

    # Extract priority actions if available
    actions = []
    actions_section = re.search(r"## 4\. Operational Remediation & Action Protocol\s+([\s\S]*?)(?:$|#)", report_markdown)
    if actions_section:
        raw_actions = re.findall(r"(?:^|\n)\s*(?:\d+\.|\-|\*)\s*(.+)", actions_section.group(1))
        actions = [a.strip() for a in raw_actions[:5] if a.strip()]
    if not actions:
        actions = [
            f"Transmit hazard bulletin to local harbor master and coastal patrol ({location_text})",
            "Deploy Remotely Operated Vehicle (ROV) for high-resolution visual confirmation",
            "Schedule mechanical salvage operation for heavy entanglement risks"
        ]

    return {
        # Backward compatibility field:
        "report": report_markdown,
        
        # Enriched structured fields:
        "report_markdown": report_markdown,
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


# ---------------------------------------------------------------------------
# INTERACTIVE DEMO / TEST UI (for frontend preview and verification)
# ---------------------------------------------------------------------------
@app.get("/demo", response_class=HTMLResponse)
def demo_interface():
    """Provides a built-in modern dashboard to test /detect and /report interactively."""
    html_file = os.path.join(os.path.dirname(__file__), "demo.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Demo dashboard template not found</h1>"