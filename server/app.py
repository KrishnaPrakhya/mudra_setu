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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# holistic = None # Replaced with hand_landmarker
hand_landmarker = None # New HandLandmarker
sequence_data_raw = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
current_prediction = "..."
current_confidence = 0.0
actions_map = {}

# --- Initialization ---
def initialize():
    global model, scaler, label_encoder, hand_landmarker, actions_map # Updated holistic to hand_landmarker
    
    print(f"Loading resources from: {MODEL_DIR}")
    model = load_model_with_custom_objects(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    if not all([model, scaler, label_encoder]):
        print("Fatal: Failed to load one or more ML resources.")
        return
    
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    print(f"Actions loaded: {actions_map}")
    
    # Create HandLandmarker options
    # USER ACTION: You need to download the 'hand_landmarker.task' model file from MediaPipe
    # and place it in an accessible path. Update 'model_asset_path' accordingly.
    # Example: model_asset_path = os.path.join(BASE_DIR, 'models', 'hand_landmarker.task')
    model_asset_path = 'hand_landmarker.task' # <<< --- USER: UPDATE THIS PATH
    if not os.path.exists(model_asset_path):
        print(f"ERROR: Hand landmarker model file not found at {model_asset_path}")
        print("Please download it from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#models")
        # Decide how to handle this - exit, or run without hand landmarking?
        # For now, we'll allow the app to run but hand landmarking will fail.
        # return # Or raise an exception

    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO, # Use VIDEO mode for processing video frames
        num_hands=2, # Max number of hands to detect
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    try:
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize MediaPipe HandLandmarker: {e}")
        # Decide how to handle this - exit, or run without hand landmarking?
        # return # Or raise an exception

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
    
    # Convert the BGR image to RGB, then to MediaPipe's Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Get current timestamp in milliseconds for HandLandmarker
    # This is a placeholder; in a real video stream, you'd use actual frame timestamps
    timestamp_ms = int(getattr(process_frame_sync, 'timestamp', 0))
    process_frame_sync.timestamp = timestamp_ms + 1 # Increment for next frame

    # Process with HandLandmarker
    # NOTE: hand_landmarker might not be initialized if the .task file was not found
    if hand_landmarker is None:
        raise RuntimeError("HandLandmarker not initialized. Check model file path.")

    try:
        # For VIDEO mode, use detect_for_video
        hand_landmarker_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as e:
        # This can happen if the model file is invalid or other MediaPipe issues
        raise RuntimeError(f"Error during hand_landmarker.detect_for_video: {str(e)}")

    frame.flags.writeable = True
    
    # Extract landmark data for frontend visualization
    # This needs to be adapted based on HandLandmarkerResult structure
    # HandLandmarkerResult has `hand_landmarks` (list of hands) and `handedness` (list of hands)
    # Each hand in `hand_landmarks` is a list of NormalizedLandmark objects
    landmarks_for_fe = {
        'pose': [], # Pose landmarks are NOT available from HandLandmarker
        'leftHand': [],
        'rightHand': []
    }
    if hand_landmarker_result.hand_landmarks:
        for i, hand_landmarks_proto in enumerate(hand_landmarker_result.hand_landmarks):
            hand_label = hand_landmarker_result.handedness[i][0].category_name # 'Left' or 'Right'
            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks_proto]
            if hand_label == 'Left':
                landmarks_for_fe['leftHand'] = landmarks
            elif hand_label == 'Right':
                landmarks_for_fe['rightHand'] = landmarks

    # If no hands are detected, we might not proceed or return a specific message.
    # The original code checked for pose_landmarks. Now we check for hand_landmarks.
    if not hand_landmarker_result.hand_landmarks:
        return {
            "prediction": "No hands detected", # Updated message
            "confidence": 0.0,
            "landmarks": landmarks_for_fe
        }

    # Extract keypoints for the model
    # IMPORTANT: `extract_focused_keypoints_realtime` was designed for Holistic output.
    # It will LIKELY FAIL or produce incorrect results with HandLandmarkerResult.
    # This function (in predict.py) needs to be updated to accept HandLandmarkerResult
    # and extract relevant keypoints. It previously might have used pose landmarks which are no longer available here.
    # For now, we pass the new result object, but expect this to be a point of failure/required update.
    try:
        # USER ACTION: Review and update `extract_focused_keypoints_realtime` in `predict.py`
        # to handle `HandLandmarkerResult` and the absence of pose landmarks from this specific task.
        # The model's expected input features might also need to change if it relied on pose.
        keypoints = extract_focused_keypoints_realtime(hand_landmarker_result) # <<< --- LIKELY NEEDS UPDATE
        sequence_data_raw.append(keypoints)
    except Exception as e:
        # More specific error message
        error_msg = f"Error extracting/processing landmarks for model with HandLandmarkerResult: {str(e)}. " \
                    f"Ensure 'extract_focused_keypoints_realtime' in 'predict.py' is updated for HandLandmarker output " \
                    f"and handles the absence of pose data from this task."
        raise RuntimeError(error_msg) from e 
    
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