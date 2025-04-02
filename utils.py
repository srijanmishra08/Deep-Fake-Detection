import cv2
import numpy as np
import tensorflow as tf
import os

def load_model(model_path):
    """
    Load the pre-trained model
    """
    try:
        print(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def create_dummy_model(input_shape=(None, 128, 128, 3)):
    """
    Create a simplified model that can handle variable input shapes
    """
    print("Creating a simplified dummy model for demonstration")
    
    # Create a model that can take one frame at a time (simpler and more robust)
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Dummy model created successfully")
    return model

def preprocess_video(video_path, target_size=(128, 128), max_frames=20):
    """
    Process video file and extract frames for prediction
    Handle videos of any dimension
    """
    try:
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
            
        # Get video properties
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count_total / fps if fps > 0 else 0
        
        print(f"Video info: {frame_count_total} frames, {fps} FPS, {duration:.2f} seconds")
        
        # For very short videos, take all frames
        # For longer videos, sample frames evenly throughout the video
        frames = []
        if frame_count_total <= max_frames:
            # For short videos, get all frames
            frame_index = 0
            while frame_index < frame_count_total:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Ensure the frame is in RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                frame = cv2.resize(frame, target_size)
                frame = frame / 255.0  # Normalize
                frames.append(frame)
                frame_index += 1
        else:
            # For longer videos, sample frames evenly
            frame_indices = np.linspace(0, frame_count_total - 1, max_frames, dtype=int)
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Ensure the frame is in RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                frame = cv2.resize(frame, target_size)
                frame = frame / 255.0  # Normalize
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            print("No frames could be extracted from the video")
            return None
            
        print(f"Extracted {len(frames)} frames for analysis")
        return np.array(frames)
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_frames(model, frames):
    """
    Make predictions on video frames using the model
    """
    try:
        if model is None:
            return None
            
        # Process each frame individually and average the results
        predictions = []
        for frame in frames:
            # Add batch dimension
            frame_batch = np.expand_dims(frame, axis=0)
            pred = model.predict(frame_batch, verbose=0)[0][0]
            predictions.append(pred)
            
        # Average the predictions
        final_prediction = np.mean(predictions)
        return final_prediction
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None 