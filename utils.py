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
                # Limit memory to 1GB
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
                    )
                except:
                    pass
        
        model = tf.keras.models.load_model(model_path)
        # Optimize the model for inference
        model = tf.keras.models.clone_model(model)
        model.set_weights(model.get_weights())
        _model_cache = model
        
        # Enable mixed precision for faster inference
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
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

def preprocess_video(video_path, target_size=(128, 128), max_frames=6):
    """
    Process video file and extract frames for prediction with aggressive optimization
    """
    try:
        start_time = time.time()
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return None
            
        # Check file size before processing
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 50:  # Limit to 50MB
            logger.error(f"Video file too large: {file_size_mb:.1f}MB (max 50MB)")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
            
        # Get video properties
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count_total / fps if fps > 0 else 0
        
        if duration > 60:  # Limit to 1 minute
            logger.error(f"Video too long: {duration:.1f}s (max 60s)")
            return None
            
        logger.info(f"Video info: {frame_count_total} frames, {fps} FPS, {duration:.2f} seconds")
        
        # Calculate frame indices to sample
        if frame_count_total <= max_frames:
            frame_indices = range(0, frame_count_total)
        else:
            # Take frames from first and last portions of video
            half = max_frames // 2
            first_half = np.linspace(0, frame_count_total // 4, half, dtype=int)
            second_half = np.linspace(3 * frame_count_total // 4, frame_count_total - 1, max_frames - half, dtype=int)
            frame_indices = np.concatenate([first_half, second_half])
        
        # Pre-allocate numpy array for frames
        frames = np.zeros((len(frame_indices), target_size[0], target_size[1], 3), dtype=np.float32)
        
        for i, frame_index in enumerate(frame_indices):
            frame_start = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame with minimal operations
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if h > 480 or w > 480:  # More aggressive downsizing
                scale = 480 / max(h, w)
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames[i] = frame / 255.0
            
            frame_time = time.time() - frame_start
            logger.info(f"Frame {i+1}/{len(frame_indices)} processed in {frame_time:.2f}s")
            
            if time.time() - start_time > 25:  # Emergency timeout
                logger.error("Processing took too long, aborting")
                return None
        
        cap.release()
        gc.collect()  # Force garbage collection
        
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

def predict_frames(model, frames, batch_size=2):
    """
    Make predictions on video frames using the model with aggressive optimization
    """
    try:
        if model is None:
            return None
            
        start_time = time.time()
        logger.info(f"Starting predictions on {len(frames)} frames with batch size {batch_size}")
        
        # Process frames in smaller batches to avoid memory issues
        predictions = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_tensor = tf.constant(batch, dtype=tf.float16)  # Use float16 for faster processing
            preds = model.predict(batch_tensor, verbose=0)
            predictions.extend(preds.flatten())
            
            if time.time() - start_time > 25:  # Emergency timeout
                logger.error("Prediction took too long, aborting")
                return None
                
            gc.collect()  # Clear memory after each batch
        
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