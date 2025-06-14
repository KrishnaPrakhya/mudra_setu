from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import base64
import sys
import os
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import joblib
import tensorflow as tf
import mediapipe as mp

# Add the model directory to the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR_REL = os.path.join(BASE_DIR, 'model')
sys.path.append(MODEL_DIR_REL)

# Import the prediction module
from predict import load_model_with_custom_objects, extract_focused_keypoints_realtime, add_temporal_features_realtime

app = FastAPI()

# Add a thread pool executor for running blocking ML code
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Add CORS middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # 
    allow_headers=["*"],  # 
)

# --- Configuration ---
MODEL_DIR = os.path.join(MODEL_DIR_REL, 'sign_model_focused_enhanced_attention_v2_0.9880_prior1')
MODEL_PATH = os.path.join(MODEL_DIR, 'corrected_enhanced_focused_attention_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
SEQUENCE_LENGTH = 32
PREDICTION_THRESHOLD = 0.60
PREDICTION_BUFFER_SIZE = 10

# --- Global State ---
model = None
scaler = None
label_encoder = None
holistic = None
sequence_data_raw = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
current_prediction = "..."
current_confidence = 0.0
actions_map = {}

# --- Initialization ---
def initialize():
    global model, scaler, label_encoder, holistic, actions_map
    
    print(f"Loading resources from: {MODEL_DIR}")
    model = load_model_with_custom_objects(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    if not all([model, scaler, label_encoder]):  # 
        print("Fatal: Failed to load one or more ML resources.")
        # In a real application, you might want to exit or prevent the app from starting
        return
    
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    print(f"Actions loaded: {actions_map}")
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("MediaPipe Holistic initialized successfully.")

# Initialize on startup
initialize()

# --- Pydantic Models ---
class FrameData(BaseModel):
    image: str  # Base64 encoded image 

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    landmarks: Optional[Dict[str, Any]] = None

# --- Synchronous Worker Function ---
def process_frame_sync(frame_data: FrameData) -> Dict[str, Any]:
    """
    This function runs the synchronous, blocking ML/CV code.
    It's designed to be called from the main async endpoint via run_in_executor.
    """
    global sequence_data_raw, prediction_buffer, current_prediction, current_confidence

    try:
        # Decode base64 image
        image_data_str = frame_data.image.split(',')[1] if ',' in frame_data.image else frame_data.image
        img_data = base64.b64decode(image_data_str)
        
        # Convert to OpenCV image
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image from base64 string")  # 
            
    except Exception as e:
        # Raise a specific error for invalid image data to be caught by the endpoint
        raise ValueError(f"Invalid image data: {str(e)}") from e # 

    # Pre-process frame for MediaPipe
    frame = cv2.resize(frame, (640, 480))
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = holistic.process(frame_rgb)  # 
    frame.flags.writeable = True  # 
    
    # Extract landmark data for frontend visualization
    landmarks_for_fe = {
        'pose': [{'x': p.x, 'y': p.y, 'z': p.z, 'visibility': p.visibility} 
                 for p in (results.pose_landmarks.landmark if results.pose_landmarks else [])],
        'leftHand': [{'x': p.x, 'y': p.y, 'z': p.z} 
                     for p in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])],  # 
        'rightHand': [{'x': p.x, 'y': p.y, 'z': p.z} 
                      for p in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])]
    }

    # If no pose is detected, we can't proceed with prediction
    if not results.pose_landmarks:
        return {
            "prediction": "No pose detected",
            "confidence": 0.0,  # 
            "landmarks": landmarks_for_fe
        }

    # Extract keypoints for the model
    try:
        keypoints = extract_focused_keypoints_realtime(results)
        sequence_data_raw.append(keypoints)
    except Exception as e:
        raise RuntimeError(f"Error extracting landmarks for model: {str(e)}") from e # 
    
    # Perform prediction only if the sequence is full
    if len(sequence_data_raw) == SEQUENCE_LENGTH:
        try:
            X_seq_raw = np.array(list(sequence_data_raw))
            X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
            X_scaled = scaler.transform(X_reshaped)  # 
            X_scaled_reshaped = X_scaled.reshape(X_seq_raw.shape)
            X_enhanced = add_temporal_features_realtime(X_scaled_reshaped)
            X_input = np.expand_dims(X_enhanced, axis=0)
            
            # Use predict_on_batch for slight performance gain
            prediction_probs = model.predict_on_batch(X_input)[0]  # 
            predicted_index = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_index]
            
            if confidence > PREDICTION_THRESHOLD:
                prediction_buffer.append(predicted_index)  # 
                
            if len(prediction_buffer) > 0:
                # Use a simple majority vote from the buffer for stability
                most_common_pred_index = max(set(prediction_buffer), key=list(prediction_buffer).count)  # 
                current_prediction = actions_map.get(most_common_pred_index, "...")
                # Update confidence to reflect the latest confident prediction
                if most_common_pred_index == predicted_index:
                    current_confidence = confidence  # 

        except Exception as e:
            raise RuntimeError(f"Error during prediction processing: {str(e)}") from e

    return {
        "prediction": current_prediction,
        "confidence": float(current_confidence * 100),  # 
        "landmarks": landmarks_for_fe
    }


# --- API Endpoints ---
@app.post("/api/predict", response_model=PredictionResponse)
async def predict_frame(frame_data: FrameData, request: Request):
    """
    Asynchronous endpoint that runs the blocking ML code in a thread pool
    to avoid blocking the server's event loop.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor, process_frame_sync, frame_data
        )
        return PredictionResponse(**result)
    except ValueError as e:
        # Catches bad image data errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catches all other errors from the prediction pipeline
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/api/reset")
async def reset_sequence():
    """Resets the prediction sequence and buffer."""
    global sequence_data_raw, prediction_buffer, current_prediction, current_confidence
    try:
        sequence_data_raw.clear()  # 
        prediction_buffer.clear()
        current_prediction = "..."
        current_confidence = 0.0
        return {"status": "sequence reset", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting sequence: {str(e)}")


@app.get("/api/health")
def health_check():
    """Provides a health check of the API and loaded models."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mediapipe_loaded": holistic is not None,  # 
        "actions_count": len(actions_map),
        "actions": actions_map
    }