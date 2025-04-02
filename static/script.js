document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const uploadBtn = document.getElementById('uploadBtn');
    const loadingSection = document.getElementById('loadingSection');
    const resultSection = document.getElementById('resultSection');
    const predictionBadge = document.getElementById('predictionBadge');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceScore = document.getElementById('confidenceScore');

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-primary');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle file selection
    function handleFile(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
        
        if (!validTypes.includes(file.type)) {
            alert('Please upload a valid video file (MP4, AVI, or MOV)');
            return;
        }

        fileName.textContent = file.name;
        fileInfo.classList.remove('d-none');
    }

    // Upload button click handler
    uploadBtn.addEventListener('click', async () => {
        const file = fileInput.files[0] || null;
        if (!file) {
            alert('Please select a video file first');
            return;
        }

        // Show loading state
        loadingSection.classList.remove('d-none');
        resultSection.classList.add('d-none');
        uploadBtn.disabled = true;

        // Create form data
        const formData = new FormData();
        formData.append('video', file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                throw new Error(result.error || 'Error processing video');
            }
        } catch (error) {
            alert(error.message);
        } finally {
            loadingSection.classList.add('d-none');
            uploadBtn.disabled = false;
        }
    });

    // Display results
    function displayResults(result) {
        resultSection.classList.remove('d-none');
        
        // Update prediction badge
        predictionBadge.textContent = result.prediction;
        predictionBadge.className = 'prediction-badge ' + result.prediction.toLowerCase();
        
        // Update confidence bar
        confidenceBar.style.width = `${result.confidence}%`;
        confidenceBar.className = `progress-bar ${result.prediction === 'REAL' ? 'bg-success' : 'bg-danger'}`;
        
        // Update confidence score text
        confidenceScore.textContent = `${result.confidence}% confidence`;
    }
}); 