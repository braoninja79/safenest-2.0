document.addEventListener('DOMContentLoaded', function() {
    const videoStream = document.getElementById('videoStream');
    const startBtn = document.getElementById('startMonitoring');
    const emergencyBtn = document.getElementById('emergencyBtn');
    const personCountText = document.querySelector('.person-count');
    const maleCountText = document.querySelector('.male-count');
    const femaleCountText = document.querySelector('.female-count');
    const statusText = document.querySelector('.status-text');
    const coverageText = document.querySelector('.coverage-text');

    let isMonitoring = false;
    let updateInterval = null;

    // Start/Stop video monitoring
    function toggleMonitoring() {
        if (!isMonitoring) {
            // Start monitoring
            videoStream.src = `/video_feed?t=${Date.now()}`;
            startBtn.innerHTML = '<i class="bi bi-stop-circle"></i> Stop Monitoring';
            startBtn.classList.remove('btn-primary');
            startBtn.classList.add('btn-danger');
            isMonitoring = true;
            updateInterval = setInterval(updateDetectionInfo, 1000);
        } else {
            // Stop monitoring
            videoStream.src = '';
            startBtn.innerHTML = '<i class="bi bi-camera-video"></i> Start Monitoring';
            startBtn.classList.remove('btn-danger');
            startBtn.classList.add('btn-primary');
            isMonitoring = false;
            clearInterval(updateInterval);
            resetDetectionInfo();
        }
    }

    // Update detection information
    function updateDetectionInfo() {
        fetch('/detection_info')
            .then(response => response.json())
            .then(data => {
                personCountText.textContent = `Person Count: ${data.num_persons}`;
                maleCountText.textContent = `Males: ${data.num_males}`;
                femaleCountText.textContent = `Females: ${data.num_females}`;
                statusText.textContent = data.status;
                coverageText.textContent = data.coverage_status;

                // Update warning styles
                statusText.classList.toggle('warning', 
                    data.status.includes('surrounded') || data.status.includes('Warning'));
                coverageText.classList.toggle('warning', data.coverage_ratio >= 0.4);

                // Show alerts for critical situations
                if (data.status.includes('surrounded') || data.coverage_ratio >= 0.4) {
                    showAlert(data.status);
                }
            })
            .catch(err => {
                console.error('Error fetching detection info:', err);
                showAlert('Error connecting to server. Please check if the server is running.');
                if (isMonitoring) {
                    toggleMonitoring();
                }
            });
    }

    // Reset detection information
    function resetDetectionInfo() {
        personCountText.textContent = 'Person Count: 0';
        maleCountText.textContent = 'Males: 0';
        femaleCountText.textContent = 'Females: 0';
        statusText.textContent = 'No person detected';
        coverageText.textContent = 'Coverage: 0.00';
        statusText.classList.remove('warning');
        coverageText.classList.remove('warning');
    }

    // Show alert message
    function showAlert(message) {
        // Remove existing alert if any
        const existingAlert = document.querySelector('.alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        // Create new alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <strong>Warning!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Insert alert before video container
        const videoContainer = document.querySelector('.video-container');
        videoContainer.parentNode.insertBefore(alertDiv, videoContainer);

        // Auto remove after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    // Handle emergency button click
    function handleEmergency() {
        // Play alarm sound
        const audio = new Audio('https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg');
        audio.play().catch(err => console.error('Error playing alarm:', err));
        
        // Show emergency alert
        showAlert('Emergency services are being contacted!');
        
        // Trigger emergency calls
        if (confirm('Do you want to call emergency services?')) {
            window.location.href = 'tel:112';
        }
    }

    // Event listeners
    startBtn.addEventListener('click', toggleMonitoring);
    emergencyBtn.addEventListener('click', handleEmergency);

    // Handle video stream errors
    videoStream.addEventListener('error', function() {
        console.error('Video stream error');
        showAlert('Error connecting to video stream. Please check if the server is running.');
        if (isMonitoring) {
            toggleMonitoring();
        }
    });

    // Start monitoring automatically when page loads
    toggleMonitoring();
});
