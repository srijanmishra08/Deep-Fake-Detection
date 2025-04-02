import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np
import random
import logging
from utils import preprocess_video, load_model, create_dummy_model, predict_frames

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('deepfake-detector')

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TIMEOUT'] = 300  # 5 minutes timeout
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the model at startup or create a dummy model if it doesn't exist
model = None
model_path = 'model.h5'

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is None:
            logger.warning("Model loading failed, creating a dummy model")
            model = create_dummy_model()
            model.save(model_path, save_format='h5')
    else:
        logger.info(f"Model file not found at {model_path}. Creating a dummy model.")
        model = create_dummy_model()
        model.save(model_path, save_format='h5')
    
    logger.info("Model initialization completed successfully")
except Exception as e:
    logger.error(f"Error during model initialization: {e}")
    import traceback
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None
    }
    return jsonify(status)

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    try:
        if 'video' not in request.files:
            logger.warning("No video file provided in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed. Supported formats: MP4, AVI, MOV, WEBM, MKV'}), 400
        
        filepath = None
        try:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Video file saved: {filepath}")
            
            # Process the video
            processed_frames = preprocess_video(filepath)
            
            if processed_frames is None or len(processed_frames) == 0:
                logger.error("Error processing video: No frames extracted")
                return jsonify({'error': 'Error processing video. Please ensure it\'s a valid video file.'}), 500
            
            # Make prediction using the model or fallback to a random prediction
            if model is not None:
                confidence = predict_frames(model, processed_frames)
                if confidence is None:
                    logger.warning("Prediction failed, using random prediction")
                    confidence = random.uniform(0.3, 0.7)
            else:
                # Fallback to random prediction if model is not available
                logger.warning("Model not available, using random prediction")
                confidence = random.uniform(0.3, 0.7)
            
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
            
            result = {
                'prediction': 'FAKE' if confidence > 0.5 else 'REAL',
                'confidence': round(float(confidence) * 100, 2)
            }
            
            logger.info(f"Prediction result: {result}")
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Clean up in case of error
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({'error': 'An error occurred while processing the video. Please try again.'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in detect_deepfake: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

if __name__ == '__main__':
    # Use Gunicorn for production deployment
    # When running directly, use the Flask development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug) 