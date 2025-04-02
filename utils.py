import cv2
import numpy as np
import tensorflow as tf
import os
import logging
import time
import gc

logger = logging.getLogger('deepfake-detector')

def load_model(model_path):
    """
    Load the pre-trained model
    """
    try:
        logger.info(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None

def create_dummy_model(input_shape=(None, 128, 128, 3)):
    """
    Create a simplified model that can handle variable input shapes
    """
    logger.info("Creating a simplified dummy model for demonstration")
    
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
    
    logger.info("Dummy model created successfully")
    return model

def preprocess_video(video_path, target_size=(128, 128), max_frames=10):
    """
    Process video file and extract frames for prediction
    Handle videos of any dimension with optimized memory usage
    """
    try:
        start_time = time.time()
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
            
        # Get video properties
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count_total / fps if fps > 0 else 0
        
        logger.info(f"Video info: {frame_count_total} frames, {fps} FPS, {duration:.2f} seconds")
        
        # Calculate frame indices to sample
        if frame_count_total <= max_frames:
            frame_indices = range(0, frame_count_total)
        else:
            frame_indices = np.linspace(0, frame_count_total - 1, max_frames, dtype=int)
        
        # Pre-allocate numpy array for frames
        frames = np.zeros((len(frame_indices), target_size[0], target_size[1], 3), dtype=np.float32)
        
        for i, frame_index in enumerate(frame_indices):
            frame_start = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame - use smaller intermediate size for faster processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if h > 720 or w > 720:
                intermediate_size = (min(720, w), min(720, h))
                frame = cv2.resize(frame, intermediate_size)
            frame = cv2.resize(frame, target_size)
            frames[i] = frame / 255.0
            
            frame_time = time.time() - frame_start
            logger.info(f"Frame {i+1}/{len(frame_indices)} processed in {frame_time:.2f}s")
            
            # Force garbage collection every few frames
            if i % 3 == 0:
                gc.collect()
        
        cap.release()
        
        if np.all(frames == 0):
            logger.error("No frames could be extracted from the video")
            return None
            
        total_time = time.time() - start_time
        logger.info(f"Extracted {len(frames)} frames in {total_time:.2f}s")
        return frames
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def predict_frames(model, frames, batch_size=4):
    """
    Make predictions on video frames using the model with batching
    """
    try:
        if model is None:
            return None
            
        start_time = time.time()
        logger.info(f"Starting predictions on {len(frames)} frames with batch size {batch_size}")
        
        # Process frames in batches
        predictions = []
        for i in range(0, len(frames), batch_size):
            batch_start = time.time()
            batch = frames[i:i+batch_size]
            
            # Clear GPU memory if available
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
            
            preds = model.predict(batch, verbose=0)
            predictions.extend(preds.flatten())
            
            batch_time = time.time() - batch_start
            logger.info(f"Batch {i//batch_size + 1} processed in {batch_time:.2f}s")
            
            # Force garbage collection after each batch
            gc.collect()
            
        # Average the predictions
        final_prediction = np.mean(predictions)
        total_time = time.time() - start_time
        logger.info(f"Predictions completed in {total_time:.2f}s. Final prediction value: {final_prediction}")
        return final_prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 