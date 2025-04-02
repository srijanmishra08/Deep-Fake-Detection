# DeepFake Detection Web Application

A web-based application for detecting deepfake videos using machine learning.

## Features

- Upload and analyze videos for potential deepfake content
- Handle videos of any dimension and format (MP4, AVI, MOV, WEBM, MKV)
- Responsive web interface with real-time feedback
- API endpoint for programmatic access
- Containerized deployment for easy setup

## System Requirements

- Python 3.9+ (3.12 also supported)
- Docker (for containerized deployment)
- 4GB+ RAM recommended for video processing

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

### Option 2: Docker Deployment

1. Build and run using Docker:
   ```
   docker build -t deepfake-detector .
   docker run -p 5000:5000 deepfake-detector
   ```

2. Or use the deployment script:
   ```
   chmod +x deploy.sh
   ./deploy.sh
   ```

## Usage

1. Open the web interface at `http://localhost:5000`
2. Upload a video file using the upload button or drag-and-drop
3. Wait for the analysis to complete
4. View the results showing whether the video is detected as real or fake

## API Usage

The application provides a simple API endpoint for programmatic access:

```
POST /detect
```

Example using curl:
```
curl -X POST -F "video=@path/to/your/video.mp4" http://localhost:5000/detect
```

Response format:
```json
{
  "prediction": "REAL|FAKE",
  "confidence": 87.5
}
```

## Configuration

Environment variables:
- `PORT`: Port for the web server (default: 5000)
- `FLASK_ENV`: Set to "development" for debug mode

## Directory Structure

```
deepfake-detection/
├── app.py               # Main Flask application
├── utils.py             # Helper utilities
├── model.h5             # Pre-trained or dummy model
├── requirements.txt     # Python dependencies
├── static/              # Static assets
│   ├── styles.css       # CSS styling
│   └── script.js        # Frontend JavaScript
├── templates/           # HTML templates
│   └── index.html       # Main page
├── uploads/             # Temporary video storage
├── Dockerfile           # Docker configuration
└── deploy.sh            # Deployment script
```

## Development

To contribute to this project:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

## License

MIT

## Acknowledgements

- TensorFlow and Keras for machine learning capabilities
- Flask for the web framework
- OpenCV for video processing
