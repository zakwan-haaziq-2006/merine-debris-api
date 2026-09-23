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
import base64
import sqlite3
import datetime
import bcrypt
import jwt

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "deepsea_marine_sonar_sec_key_2026_9981")
JWT_ALGORITHM = "HS256"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(email: str, full_name: str) -> str:
    payload = {
        "sub": email,
        "name": full_name,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication token required")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token identity")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, created_at FROM users WHERE email = ?", (email.lower().strip(),))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=401, detail="User account not found")

        return {
            "id": row[0],
            "email": row[1],
            "full_name": row[2],
            "created_at": row[3]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please sign in again")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authorization token")


class UserSignupRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = "Marine Surveyor"


class UserLoginRequest(BaseModel):
    email: str
    password: str



def _crop_and_enhance_target(image: Image.Image, box: dict, scale: float = 3.0) -> dict:
    """
    Crops the bounding box from the sonar image, scales up by 3x,
    and applies classical contrast enhancement, noise reduction, and sharpening.
    Returns Base64 data URIs for raw_crop and enhanced_crop.
    """
    try:
        w, h = image.width, image.height
        x1 = max(0, min(w, int(box["x1"])))
        y1 = max(0, min(h, int(box["y1"])))
        x2 = max(0, min(w, int(box["x2"])))
        y2 = max(0, min(h, int(box["y2"])))

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return {"raw_crop": None, "enhanced_crop": None}

        crop = image.crop((x1, y1, x2, y2))

        # Base64 Raw Crop
        buf_raw = io.BytesIO()
        crop.save(buf_raw, format="PNG")
        raw_b64 = "data:image/png;base64," + base64.b64encode(buf_raw.getvalue()).decode("utf-8")

        # 3x Lanczos Upscale
        new_w = max(30, int(crop.width * scale))
        new_h = max(30, int(crop.height * scale))
        upscaled = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Classical Speckle Denoise (Median filter) + Contrast & Sharpness Boost
        denoised = upscaled.filter(ImageFilter.MedianFilter(size=3))
        enhanced = ImageEnhance.Contrast(denoised).enhance(1.65)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.8)

        buf_enh = io.BytesIO()
        enhanced.save(buf_enh, format="PNG")
        enh_b64 = "data:image/png;base64," + base64.b64encode(buf_enh.getvalue()).decode("utf-8")

        return {"raw_crop": raw_b64, "enhanced_crop": enh_b64}
    except Exception as e:
        print(f"[WARN] Failed to crop and enhance box {box}: {e}")
        return {"raw_crop": None, "enhanced_crop": None}



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
MODEL_PATH = ONNX_PATH if os.path.exists(ONNX_PATH) else PT_PATH

INFERENCE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.5
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CRITICAL_ANOMALIES = {"human", "aircraft", "ship"}
HARDWARE_AND_RIGGING = {"Chain", "Hook", "Propeller", "Valve"}
CONSUMER_PLASTICS_WASTE = {"Bottle", "Can", "Drink-carton", "Shampoo-bottle", "Standing-bottle", "Tire"}


def calculate_priority_score(detection: dict):
    """
    Calculates a numeric priority score (0-100) and priority label for a single detection.
    Factors & Weights (Total max = 100):
      1. Category weight (max 50 pts): Critical anomalies (50) > Hardware/Rigging (30) > Plastics/Waste (15) > Other (10)
      2. Confidence weight (max 30 pts): confidence * 30 (higher confidence = more trust in detection)
      3. Size weight (max 20 pts): min(area_percentage * 2, 20) (larger objects = greater potential hazard)
    
    Priority Labels:
      >= 75.0: URGENT
      >= 50.0: HIGH
      >= 30.0: MEDIUM
      < 30.0 : LOW
    """
    cls = detection.get("class", "")
    confidence = float(detection.get("confidence", 0.0))
    area_pct = float(detection.get("area_percentage", 0.0))

    if cls in CRITICAL_ANOMALIES:
        cat_score = 50.0
    elif cls in HARDWARE_AND_RIGGING:
        cat_score = 30.0
    elif cls in CONSUMER_PLASTICS_WASTE:
        cat_score = 15.0
    else:
        cat_score = 10.0

    conf_score = min(max(confidence, 0.0), 1.0) * 30.0
    size_score = min(max(area_pct, 0.0) * 2.0, 20.0)

    score = round(cat_score + conf_score + size_score, 1)

    if score >= 75.0:
        label = "URGENT"
    elif score >= 50.0:
        label = "HIGH"
    elif score >= 30.0:
        label = "MEDIUM"
    else:
        label = "LOW"

    return score, label

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

model = None
try:
    model = YOLO(PT_PATH)
except Exception as e:
    print(f"[WARN] Failed to load primary model path '{MODEL_PATH}': {e}")
    if MODEL_PATH != PT_PATH and os.path.exists(PT_PATH):
        print(f"[INFO] Attempting fallback to PyTorch model weights '{PT_PATH}'...")
        try:
            model = YOLO(PT_PATH)
        except Exception as pt_err:
            raise RuntimeError(
                f"Failed to load trained sonar model from both '{MODEL_PATH}' and '{PT_PATH}'. "
                f"Error: {pt_err}"
            ) from pt_err
    else:
        raise RuntimeError(
            f"Failed to load trained sonar model from '{MODEL_PATH}'. "
            f"Ensure 'best.onnx' or 'best.pt' exists and is valid. Error: {e}"
        ) from e

import torch
torch.set_num_threads(1)

langchain_llm = None
if GEMINI_API_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
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

        det = {
            "class": cls_name,
            "confidence": round(confidence, 3),
            "box": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            },
            "area_percentage": area_pct
        }
        score, label = calculate_priority_score(det)
        det["priority_score"] = score
        det["priority_label"] = label

        crops = _crop_and_enhance_target(image, det["box"])
        det["raw_crop"] = crops.get("raw_crop")
        det["enhanced_crop"] = crops.get("enhanced_crop")

        detections.append(det)

    return {
        "detections": detections,
        "image_width": image.width,
        "image_height": image.height,
        "total_detected": len(detections)
    }


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
    # Calculate priority score and label for all detections if not present, then sort descending
    processed_detections = []
    for d in request.detections:
        det = dict(d)
        if "priority_score" not in det or "priority_label" not in det:
            score, label = calculate_priority_score(det)
            det["priority_score"] = score
            det["priority_label"] = label
        processed_detections.append(det)

    sorted_detections = sorted(
        processed_detections,
        key=lambda d: d.get("priority_score", 0.0),
        reverse=True
    )

    top_priority_items = [
        {
            "class": d.get("class", "Unknown"),
            "confidence": float(d.get("confidence", 0.0)),
            "priority_score": float(d.get("priority_score", 0.0)),
            "priority_label": d.get("priority_label", "LOW")
        }
        for d in sorted_detections[:3]
    ]

    telemetry = _aggregate_survey_telemetry(sorted_detections)
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
            "top_priority_items": [],
            "statistics": {
                "total_detections": 0,
                "avg_confidence": 1.0,
                "categories": {"critical_anomalies": 0, "subsea_hardware": 0, "plastics_and_debris": 0}
            },
            "priority_actions": ["Log sector as clear in maritime registry", "Proceed with routine monitoring schedule"]
        }

    item_lines = [
        f"  - [{d.get('priority_label', 'LOW')} | Priority Score: {d.get('priority_score', 0.0)}] '{d.get('class', 'Unknown')}' "
        f"(Confidence: {int(float(d.get('confidence', 0.0)) * 100)}%, Area: {d.get('area_percentage', 0.0)}%)"
        for d in sorted_detections
    ]
    detections_summary = "\n".join(item_lines)

    prompt = f"""You are a Lead Marine Acoustic Surveyor conducting subsea survey evaluations using sonar imagery.

Provide a technically accurate, professional Marine Survey & Hazard Assessment Report based strictly on the verified acoustic detections below (listed in order of priority score descending).

Location/Sector: {location_text}
Sensor Type: {meta_info.sensor_type}
Total Detected Objects: {telemetry["total_detections"]}
Average Acoustic Confidence: {int(telemetry["avg_confidence"] * 100)}%
Target Breakdown (Priority Descending):
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
        "top_priority_items": top_priority_items,
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