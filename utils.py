import cv2
import numpy as np
import tensorflow as tf
import os
import logging
import time
import gc

logger = logging.getLogger('deepfake-detector')

# Global variable to cache the model
_model_cache = None

def load_model(model_path):
    """
    Load the pre-trained model with caching
    """
    global _model_cache
    try:
        if _model_cache is not None:
            logger.info("Using cached model")
            return _model_cache
            
        logger.info(f"Loading model from {model_path}")
        
        # Configure TensorFlow for better performance
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        if tf.config.list_physical_devices('GPU'):
            # Limit GPU memory growth
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        
        model = tf.keras.models.load_model(model_path)
        # Optimize the model for inference
        model = tf.keras.models.clone_model(model)
        model.set_weights(model.get_weights())
        _model_cache = model
        
        logger.info("Model loaded and optimized successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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

def preprocess_video(video_path, target_size=(128, 128), max_frames=8):
    """
    Process video file and extract frames for prediction with optimized memory usage
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
        
        # Process frames in chunks to reduce memory usage
        chunk_size = 2
        for chunk_start in range(0, len(frame_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(frame_indices))
            chunk_indices = frame_indices[chunk_start:chunk_end]
            
            for i, frame_index in enumerate(chunk_indices, start=chunk_start):
                frame_start = time.time()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame with minimal operations
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                if h > 720 or w > 720:
                    scale = 720 / max(h, w)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frame = cv2.resize(frame, target_size)
                frames[i] = frame / 255.0
                
                frame_time = time.time() - frame_start
                logger.info(f"Frame {i+1}/{len(frame_indices)} processed in {frame_time:.2f}s")
            
            # Clear memory after each chunk
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
    Make predictions on video frames using the model with optimized batching
    """
    try:
        if model is None:
            return None
            
        start_time = time.time()
        logger.info(f"Starting predictions on {len(frames)} frames with batch size {batch_size}")
        
        # Convert frames to TensorFlow constant for better performance
        frames_tensor = tf.constant(frames)
        
        # Make a single prediction call for all frames
        predictions = model.predict(frames_tensor, batch_size=batch_size, verbose=0)
        predictions = predictions.flatten()
        
        # Calculate final prediction
        final_prediction = float(np.mean(predictions))
        
        total_time = time.time() - start_time
        logger.info(f"Predictions completed in {total_time:.2f}s. Final prediction value: {final_prediction:.4f}")
        return final_prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 