import asyncio
import logging
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from predict_video import predict_on_video
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR_REL = os.path.join(BASE_DIR, 'model')
sys.path.append(MODEL_DIR_REL)

from predict import load_model_with_custom_objects, add_temporal_features_realtime

# --- FastAPI App Initialization ---
app = FastAPI(title="Ultra-Fast Sign Language Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# This configuration is validated to match the new frontend extraction logic.
MODEL_DIR = os.path.join(MODEL_DIR_REL, 'sign_model_focused_enhanced_attention_v2_0.9880_prior1')
MODEL_PATH = os.path.join(MODEL_DIR, 'corrected_enhanced_focused_attention_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SEQUENCE_LENGTH = 32
# Expected count is (7*4 for pose) + (21*3 for left hand) + (21*3 for right hand)
EXPECTED_LANDMARK_COUNT = 154
CONFIDENCE_THRESHOLD = 0.70  # High confidence threshold

# --- Global State ---
model, scaler, label_encoder, actions_map = None, None, None, {}
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

@app.on_event("startup")
def startup_event():
    global model, scaler, label_encoder, actions_map
    logger.info("Loading prediction model and scaler...")
    model = load_model_with_custom_objects(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    logger.info("Resources loaded successfully.")

def predict_from_landmarks(landmarks: List[float], sequence_data: deque) -> dict:
    """Receives landmarks, preprocesses, and predicts, providing continuous feedback."""
    sequence_data.append(landmarks)

    if len(sequence_data) < SEQUENCE_LENGTH:
        return {
            "status": "buffering",
            "progress": len(sequence_data) / SEQUENCE_LENGTH
        }

    try:
        X_seq_raw = np.array(list(sequence_data))
        # The remainder of the prediction logic remains the same
        X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_scaled_reshaped = X_scaled.reshape(X_seq_raw.shape)
        X_enhanced = add_temporal_features_realtime(X_scaled_reshaped)
        X_input = np.expand_dims(X_enhanced, axis=0)
        
        prediction_probs = model.predict_on_batch(X_input)[0]
        predicted_index = np.argmax(prediction_probs)
        confidence = prediction_probs[predicted_index]

        if confidence > CONFIDENCE_THRESHOLD:
            return {
                "status": "prediction",
                "prediction": actions_map.get(predicted_index, "Unknown"),
                "confidence": float(confidence * 100)
            }
        else:
            return {"status": "low_confidence"}
            
    except Exception as e:
        logger.error(f"Error during prediction pipeline: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    loop = asyncio.get_running_loop()

    try:
        while True:
            landmarks_data = await websocket.receive_json()
            # **IMPROVEMENT**: Added logging for received data length to help debug future mismatches.
            if not isinstance(landmarks_data, list) or len(landmarks_data) != EXPECTED_LANDMARK_COUNT:
                logger.warning(f"Received invalid data. Expected list of {EXPECTED_LANDMARK_COUNT}, got {len(landmarks_data)}")
                continue

            prediction_result = await loop.run_in_executor(
                executor, predict_from_landmarks, landmarks_data, sequence_data
            )
            
            # Always send the result back to the client.
            await websocket.send_json(prediction_result)
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket handler: {e}", exc_info=True)




@app.post("/api/video-predict")
async def video_predict(file: UploadFile = File(...)):
    input_path = f"uploads/{file.filename}"
    output_path = f"outputs/annotated_{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    results = predict_on_video(input_path, output_path)
    return JSONResponse({
        "predictions": results,
        "annotated_video_url": f"/api/download/{os.path.basename(output_path)}"
    })

@app.get("/api/download/{filename}")
async def download_annotated_video(filename: str):
    file_path = f"outputs/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)