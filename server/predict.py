# %% [markdown]
# ## 3) Inference (with Custom Attention Layer, UI, and Confidence)

# %%
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
import os
from collections import deque
import traceback

# --- Configuration (Adjust to match the NEW model's directory) ---
MODEL_DIR = 'sign_language_model_attention' # *** Ensure this matches your training output path ***
MODEL_PATH = os.path.join(MODEL_DIR, 'attention_sign_classifier_best.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Model/Data specific config
SEQUENCE_LENGTH = 32

# Real-time specific config
PREDICTION_THRESHOLD = 0.60 
PREDICTION_BUFFER_SIZE = 10 
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# --- Custom Attention Layer Definition (Must match the one in training script) ---
@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class AttentionWeightedAverage(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('A list of two tensors is expected. '
                             f'Got: {inputs}')
        gru_output, attention_probs = inputs
        weighted_sequence = gru_output * attention_probs
        context_vector = tf.reduce_sum(weighted_sequence, axis=1)
        return context_vector

    def get_config(self):
        base_config = super().get_config()
        return base_config

# --- Helper Functions ---

def load_model_with_custom_objects(model_path):
    """Loads a Keras model with custom objects and recompiles."""
    custom_objects = {
        'AttentionWeightedAverage': AttentionWeightedAverage
    }
    try:
        print(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects, # Pass the custom layer here
            compile=False # It's often safer to recompile
        )
        print("Model loaded successfully.")
        # Recompile with an optimizer (Adam is a common choice)
        # Match the learning rate if it was critical during training, or use a default.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Match training LR
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model recompiled.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def extract_keypoints(results):
    """Extracts keypoints (Pose, Face, LH, RH) - same as collection."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def add_temporal_features_realtime(X_scaled):
    """Adds temporal features (Must match training implementation)."""
    if X_scaled.shape[0] < 2: 
        X_diff1 = np.zeros_like(X_scaled)
        X_diff2 = np.zeros_like(X_scaled)
    else:
        X_diff1 = np.diff(X_scaled, axis=0, prepend=X_scaled[:1, :])
        X_diff2 = np.diff(X_diff1, axis=0, prepend=X_diff1[:1, :])

    velocity_mag = np.linalg.norm(X_diff1, axis=-1, keepdims=True)
    acceleration_mag = np.linalg.norm(X_diff2, axis=-1, keepdims=True)
    
    if X_scaled.ndim == 1: 
        velocity_mag = velocity_mag.reshape(-1,1)
        acceleration_mag = acceleration_mag.reshape(-1,1)

    return np.concatenate([X_scaled, X_diff1, X_diff2, velocity_mag, acceleration_mag], axis=-1)


def draw_landmarks_and_prediction(image, results, prediction_text, confidence_value):
    """Draws landmarks on the image and the prediction text with confidence."""
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    text_to_display = f"Prediction: {prediction_text.upper()} ({confidence_value:.2f})"
    cv2.rectangle(image, (0,0), (image.shape[1], 40), (245, 117, 16), -1) 
    cv2.putText(image, text_to_display, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


# --- Main Inference Function ---
def run_inference():
    print("Loading resources...")
    model = load_model_with_custom_objects(MODEL_PATH) # Use the updated loading function
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if not all([model, scaler, label_encoder]):
        print("Failed to load one or more resources. Exiting.")
        return

    print("Resources loaded successfully.")
    actions_map = {i: action for i, action in enumerate(label_encoder.classes_)}
    print(f"Actions: {actions_map}")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        holistic.close()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

    window_name = 'Sign Language Inference'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    sequence_data_raw = deque(maxlen=SEQUENCE_LENGTH) 
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    current_prediction_text = "..."
    current_confidence = 0.0

    print("Starting inference loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame or camera closed.")
            break
        
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        keypoints = extract_keypoints(results)
        sequence_data_raw.append(keypoints)

        if len(sequence_data_raw) == SEQUENCE_LENGTH:
            try:
                X_seq_raw = np.array(sequence_data_raw)

                original_shape = X_seq_raw.shape
                X_reshaped = X_seq_raw.reshape(-1, X_seq_raw.shape[-1])
                X_scaled = scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(original_shape) 

                X_enhanced = add_temporal_features_realtime(X_scaled) 

                X_input = np.expand_dims(X_enhanced, axis=0) 
                prediction_probabilities = model.predict(X_input)[0]

                predicted_index = np.argmax(prediction_probabilities)
                confidence = prediction_probabilities[predicted_index]

                if confidence > PREDICTION_THRESHOLD:
                    prediction_buffer.append(predicted_index)
                    current_confidence = confidence 
                
                if len(prediction_buffer) > 0:
                    most_common_pred_index = max(set(prediction_buffer), key=prediction_buffer.count)
                    current_prediction_text = actions_map.get(most_common_pred_index, "...")
                    if most_common_pred_index == predicted_index and confidence > PREDICTION_THRESHOLD:
                         current_confidence = confidence
                else:
                    current_prediction_text = "..."
                    current_confidence = 0.0

            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()
                current_prediction_text = "Error"
                current_confidence = 0.0
        
        display_image = frame.copy() 
        draw_landmarks_and_prediction(display_image, results, current_prediction_text, current_confidence)
        
        cv2.imshow(window_name, display_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Inference finished.")

if __name__ == "__main__":
    run_inference()
