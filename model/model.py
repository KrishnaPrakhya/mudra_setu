# %% [markdown]
# ## 2) Model Training (Corrected Enhanced Focused Data with Conv1D + BiGRU + Attention)

# %%
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
import joblib
import traceback

warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
CONFIG = {
    'actions': ['drink','eat','hello','thankyou','yes','close','open','sit','stand','wake up','walk','book','dance','listen','look','talk'], # User updated actions
    'sequence_length': 32,
    'data_path': 'Mudras_Focused/raw_data_focused', # Data from the focused collection
    'model_save_path': 'sign_model_focused_enhanced_attention_v3', # New path for this corrected model
    'batch_size': 32,
    'epochs': 150, 
    'learning_rate': 0.0005, 
    'dropout_rate': 0.4,        # General dropout for Dense layers
    'spatial_dropout_rate_conv1': 0.25, # Spatial Dropout for first Conv block
    'spatial_dropout_rate_conv2': 0.35, # Spatial Dropout for second Conv block
    'gru_dropout': 0.3,         # Dropout for GRU inputs/recurrent state
    'validation_split': 0.2,
    'test_split': 0.15,
    'early_stopping_patience': 20, 
    'reduce_lr_patience': 10,
    'use_data_augmentation': True,
    'use_class_weights': True,
    'NUM_SELECTED_POSE_LANDMARKS': 7, # From focused data collection
}

os.makedirs(CONFIG['model_save_path'], exist_ok=True)

# --- Custom Attention Layer (Same as before) ---
@tf.keras.utils.register_keras_serializable(package='CustomLayers')
class AttentionWeightedAverage(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('A list of two tensors is expected. '
                             f'Got: {inputs}')
        gru_output, attention_probs = inputs
        # gru_output shape: (batch_size, time_steps, gru_features)
        # attention_probs shape: (batch_size, time_steps, 1)
        weighted_sequence = gru_output * attention_probs
        context_vector = tf.reduce_sum(weighted_sequence, axis=1) # Sum over time_steps
        return context_vector

    def get_config(self):
        base_config = super().get_config()
        return base_config

# --- Data Processor (Configured for 154 features) ---
class SimpleDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.expected_raw_size = (self.config['NUM_SELECTED_POSE_LANDMARKS'] * 4) + (2 * 21 * 3) 
        print(f"DataProcessor initialized. Expected raw size per frame (focused data): {self.expected_raw_size}")

    def load_raw_data(self):
        print(f"Loading FOCUSED RAW data from: {self.config['data_path']}")
        sequences = []
        labels = []
        padded_count = 0
        truncated_count = 0
        skipped_count = 0
        action_counts = {action: 0 for action in self.config['actions']}

        for action in self.config['actions']:
            action_path = os.path.join(self.config['data_path'], action)
            if not os.path.exists(action_path):
                print(f"Warning: No data directory found for action '{action}' at {action_path}")
                continue

            sequence_folders = [f for f in os.listdir(action_path)
                                if os.path.isdir(os.path.join(action_path, f)) and f.isdigit()]
            
            if not sequence_folders:
                print(f"Warning: No sequence folders (numeric) found for action '{action}' in {action_path}")
                continue
            
            for seq_folder in sequence_folders:
                seq_path = os.path.join(action_path, seq_folder)
                try:
                    frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')],
                                         key=lambda x: int(x.split('.')[0]))
                except FileNotFoundError:
                    print(f"Warning: Sequence folder {seq_path} not found or not accessible. Skipping.")
                    skipped_count +=1
                    continue

                if not frame_files: 
                    skipped_count +=1
                    continue

                current_sequence_frames = []
                valid_seq = True
                for frame_file in frame_files:
                    frame_path = os.path.join(seq_path, frame_file)
                    try:
                        keypoints = np.load(frame_path)
                        if keypoints.size != self.expected_raw_size:
                           print(f"Skipping {action}/{seq_folder}/{frame_file}: Incorrect size {keypoints.size}, expected {self.expected_raw_size}")
                           valid_seq = False
                           break
                        current_sequence_frames.append(keypoints)
                    except Exception as e:
                        print(f"Error loading {frame_path}: {e}. Skipping sequence.")
                        valid_seq = False
                        break
                
                if not valid_seq or not current_sequence_frames:
                    skipped_count += 1
                    continue
                
                sequence_array = np.array(current_sequence_frames)

                if sequence_array.shape[0] == 0: 
                    skipped_count +=1
                    continue

                if sequence_array.shape[0] < self.config['sequence_length']:
                    pad_width = self.config['sequence_length'] - sequence_array.shape[0]
                    sequence_array = np.pad(sequence_array, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
                    padded_count += 1
                elif sequence_array.shape[0] > self.config['sequence_length']:
                    sequence_array = sequence_array[:self.config['sequence_length'], :]
                    truncated_count += 1
                
                sequences.append(sequence_array)
                labels.append(action)
                action_counts[action] += 1

        if not sequences:
            raise ValueError("No valid sequences found! Check the raw_data_path and ensure .npy files exist for focused data.")

        print(f"Padded: {padded_count}, Truncated: {truncated_count}, Skipped: {skipped_count}")
        for action, count in action_counts.items():
            print(f"  Action '{action}': {count} sequences loaded.")

        X = np.array(sequences)
        y = np.array(labels)
        print(f"Loaded {len(X)} sequences with final shape: {X.shape}") 
        return X, y

    def preprocess_data(self, X, fit_scaler=True):
        print("Applying preprocessing (Scaling + Temporal)...")
        original_shape = X.shape 
        X_reshaped = X.reshape(-1, X.shape[-1]) 
        if fit_scaler:
            X_scaled_flat = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled_flat = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled_flat.reshape(original_shape)

        X_diff1 = np.diff(X_scaled, axis=1, prepend=X_scaled[:, :1, :])
        X_diff2 = np.diff(X_diff1, axis=1, prepend=X_diff1[:, :1, :])
        
        velocity_mag = np.linalg.norm(X_diff1, axis=-1, keepdims=True) 
        acceleration_mag = np.linalg.norm(X_diff2, axis=-1, keepdims=True)
        
        X_enhanced = np.concatenate([X_scaled, X_diff1, X_diff2, velocity_mag, acceleration_mag], axis=-1)

        print(f"Data shape after scaling: {X_scaled.shape}")
        print(f"Data shape after adding temporal features (final input to model): {X_enhanced.shape}")
        return X_enhanced

# --- Data Augmentation (Same as before) ---
class AdvancedDataAugmentation:
    @staticmethod
    def gaussian_noise(s, noise_factor=0.005): 
        noise = np.random.normal(0, noise_factor, s.shape)
        return s + noise
    @staticmethod
    def time_shifting(s, shift_range=0.05): 
        shift = int(s.shape[0] * shift_range * (np.random.random() - 0.5)) 
        augmented_s = np.copy(s)
        if shift > 0:
            augmented_s[shift:] = s[:-shift]
            augmented_s[:shift] = 0 
        elif shift < 0:
            augmented_s[:shift] = s[-shift:]
            augmented_s[shift:] = 0 
        return augmented_s

    @staticmethod
    def scaling(s, scale_range=0.05): 
        scale_factor = 1 + (np.random.random() - 0.5) * scale_range
        return s * scale_factor
    
    @classmethod
    def augment_sequence(cls, s, augment_prob=0.6): 
        if not isinstance(s, np.ndarray): return s
        augmented = s.copy()
        if np.random.random() < augment_prob:
            if np.random.random() < 0.5: augmented = cls.gaussian_noise(augmented)
            if np.random.random() < 0.4: augmented = cls.time_shifting(augmented) 
            if np.random.random() < 0.4: augmented = cls.scaling(augmented)   
        return augmented

# --- Enhanced Attention Model for Focused Data ---
class EnhancedAttentionSignLanguageModel:
    def __init__(self, config, input_shape, num_classes):
        self.config = config
        self.input_shape = input_shape 
        self.num_classes = num_classes

    def create_model(self):
        inputs = Input(shape=self.input_shape, name='sequence_input')

        # Conv Block 1 - Enhanced
        x = layers.Conv1D(filters=96, kernel_size=7, padding='same', activation='relu', name='conv1a')(inputs) 
        x = layers.BatchNormalization(name='bn1a')(x)
        x = layers.SpatialDropout1D(self.config['spatial_dropout_rate_conv1'])(x) 
        x = layers.Conv1D(filters=96, kernel_size=5, padding='same', activation='relu', name='conv1b')(x)
        x = layers.BatchNormalization(name='bn1b')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool1')(x)

        # Conv Block 2 - Enhanced
        x = layers.Conv1D(filters=192, kernel_size=3, padding='same', activation='relu', name='conv2a')(x)
        x = layers.BatchNormalization(name='bn2a')(x)
        x = layers.SpatialDropout1D(self.config['spatial_dropout_rate_conv2'])(x)
        x = layers.Conv1D(filters=192, kernel_size=3, padding='same', activation='relu', name='conv2b')(x)
        x = layers.BatchNormalization(name='bn2b')(x)
        x = layers.MaxPooling1D(pool_size=2, name='pool2')(x) # Output shape: (batch, time_steps_after_pooling, 192)
                                                            # e.g. (batch, 32/4=8, 192)
        
        x = layers.Bidirectional(layers.GRU(128, return_sequences=True, 
                                             dropout=self.config['gru_dropout'], 
                                             recurrent_dropout=self.config['gru_dropout'], 
                                             name='gru1'))(x)
        
        gru_output = layers.Bidirectional(layers.GRU(128, return_sequences=True, 
                                                 dropout=self.config['gru_dropout'],
                                                 recurrent_dropout=self.config['gru_dropout'],
                                                 name='gru2'))(x) # Shape: (batch, time_steps_after_pooling, 256)

        # Attention Mechanism - CORRECTED
        # 1. Get energy scores for each time step from GRU output
        # Input to dense: (batch_size, time_steps_after_pooling, 256)
        # Output of dense: (batch_size, time_steps_after_pooling, 1)
        attention_energy = layers.Dense(1, activation=None, name='attention_energy')(gru_output) 
        
        # 2. Apply softmax across the time steps (axis=1) to get probabilities
        # Input to softmax: (batch_size, time_steps_after_pooling, 1)
        # Output of softmax: (batch_size, time_steps_after_pooling, 1) - probabilities sum to 1 across time_steps
        attention_probs = layers.Softmax(axis=1, name='attention_probs')(attention_energy)
        
        # 3. Compute weighted average using the custom layer
        # Inputs to custom layer: [gru_output, attention_probs]
        # gru_output shape: (batch, time_steps_after_pooling, 256)
        # attention_probs shape: (batch, time_steps_after_pooling, 1)
        context_vector = AttentionWeightedAverage(name='attention_weighted_average')([gru_output, attention_probs])
        # context_vector shape: (batch, 256)


        # Classification Head
        x = layers.Dense(128, activation='relu', name='fc1')(context_vector)
        x = layers.BatchNormalization(name='fc1_bn')(x)
        x = layers.Dropout(self.config['dropout_rate'])(x)
        
        x = layers.Dense(64, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization(name='fc2_bn')(x)
        x = layers.Dropout(self.config['dropout_rate'] * 0.5)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=inputs, outputs=outputs, name='CorrectedEnhancedFocusedAttentionSignClassifier')
        return model

    def build_model(self):
        self.model = self.create_model()
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("\nCorrected Enhanced Focused Attention Model Summary:")
        self.model.summary(line_length=120)
        return self.model

# --- Data Generator (Same as before) ---
class EfficientDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, config, shuffle=True, augment=False):
        self.X = X 
        self.y = y 
        self.batch_size = batch_size
        self.config = config
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(X))
        self.on_epoch_end() 

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        X_batch = self.X[batch_indices].copy() 
        y_batch = self.y[batch_indices]
        
        if self.augment:
            X_batch_augmented = []
            for seq_idx in range(X_batch.shape[0]):
                X_batch_augmented.append(AdvancedDataAugmentation.augment_sequence(X_batch[seq_idx]))
            X_batch = np.array(X_batch_augmented)
        
        return X_batch.astype(np.float32), y_batch.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --- Trainer (Same as before) ---
class ModelTrainer: 
    def __init__(self, config):
        self.config = config

    def create_callbacks(self, model_name):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2, 
            patience=self.config['reduce_lr_patience'],
            min_lr=1e-7, 
            verbose=1,
            mode='max'
        )
        checkpoint_path = os.path.join(self.config['model_save_path'], f'{model_name}_best.keras')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        )
        return [early_stopping, reduce_lr, model_checkpoint]

    def train_model(self, model, X_train, y_train, X_val, y_val, class_weights=None, model_name='model'):
        print(f"\nTraining {model_name}...")
        train_generator = EfficientDataGenerator(X_train, y_train, self.config['batch_size'], self.config, shuffle=True, augment=self.config['use_data_augmentation'])
        val_generator = EfficientDataGenerator(X_val, y_val, self.config['batch_size'], self.config, shuffle=False, augment=False) 
        
        callbacks = self.create_callbacks(model_name)
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            class_weight=class_weights if self.config['use_class_weights'] else None,
            verbose=1
        )
        return history

# --- Main Execution ---
def main():
    print("=== Corrected Enhanced Focused Data Sign Language Classification (Attention Model) ===")

    data_processor = SimpleDataProcessor(CONFIG)
    try:
        X_raw, y = data_processor.load_raw_data() 
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    X = data_processor.preprocess_data(X_raw, fit_scaler=True) 

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_temp, X_test, y_temp_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=CONFIG['test_split'], random_state=42, stratify=y_encoded)
    
    X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
        X_temp, y_temp_encoded, test_size=CONFIG['validation_split']/(1-CONFIG['test_split']),
        random_state=42, stratify=y_temp_encoded) 

    y_train = to_categorical(y_train_encoded, num_classes=len(CONFIG['actions']))
    y_val = to_categorical(y_val_encoded, num_classes=len(CONFIG['actions']))
    y_test = to_categorical(y_test_encoded, num_classes=len(CONFIG['actions']))

    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shape: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    class_weight_dict = None
    if CONFIG['use_class_weights']:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_values = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weight_dict = dict(enumerate(class_weights_values))
        print("Class weights:", class_weight_dict)

    model_input_shape = (X_train.shape[1], X_train.shape[2]) 
    print(f"Model input shape will be: {model_input_shape}")

    model_builder = EnhancedAttentionSignLanguageModel(CONFIG, model_input_shape, len(CONFIG['actions']))
    model = model_builder.build_model()
    
    trainer = ModelTrainer(CONFIG)
    model_checkpoint_name = 'corrected_enhanced_focused_attention_classifier' 
    history = trainer.train_model(model, X_train, y_train, X_val, y_val, class_weight_dict, model_checkpoint_name)

    best_model_path = os.path.join(CONFIG['model_save_path'], f'{model_checkpoint_name}_best.keras')
    
    custom_objects_for_loading = {'AttentionWeightedAverage': AttentionWeightedAverage}

    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        loaded_model = tf.keras.models.load_model(
            best_model_path, 
            custom_objects=custom_objects_for_loading,
            compile=False 
        )
        loaded_model.compile(optimizer=Adam(learning_rate=CONFIG['learning_rate']),
                             loss='categorical_crossentropy', metrics=['accuracy'])
        model_to_evaluate = loaded_model
    else:
        print(f"Best model not found at {best_model_path}. Evaluating the last trained model.")
        model_to_evaluate = model

    print("\nEvaluating model...")
    test_loss, test_accuracy = model_to_evaluate.evaluate(X_test, y_test, verbose=0, batch_size=CONFIG['batch_size'])
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    
    y_pred_probs = model_to_evaluate.predict(X_test, batch_size=CONFIG['batch_size'])
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    y_true_indices = y_test_encoded

    print("\nClassification Report:")
    print(classification_report(y_true_indices, y_pred_indices, target_names=label_encoder.classes_))

    joblib.dump(label_encoder, os.path.join(CONFIG['model_save_path'], 'label_encoder.pkl'))
    joblib.dump(data_processor.scaler, os.path.join(CONFIG['model_save_path'], 'scaler.pkl'))

    if history is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['model_save_path'], 'training_history.png'))
        plt.show()

    print("\n--- Training and Evaluation Completed ---")

if __name__ == "__main__":
    main()
