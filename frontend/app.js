const BACKEND_URL = 'http://localhost:3000';

const FRAME_INTERVAL_MS = 500;

const videoEl        = document.getElementById('videoFeed');
const canvasEl       = document.getElementById('overlayCanvas');
const placeholder    = document.getElementById('videoPlaceholder');
const spinner        = document.getElementById('spinnerOverlay');
const startBtn       = document.getElementById('startBtn');
const stopBtn        = document.getElementById('stopBtn');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const recBadge       = document.getElementById('recordingBadge');
const gestureDisplay = document.getElementById('gestureDisplay');
const confidenceVal  = document.getElementById('confidenceValue');
const confidenceBar  = document.getElementById('confidenceBar');
const textOutput     = document.getElementById('textOutput');
const speakBtn       = document.getElementById('speakBtn');
const clearBtn       = document.getElementById('clearBtn');
const replayBtn      = document.getElementById('replayBtn');
const themeToggle    = document.getElementById('themeToggle');

const ctx = canvasEl.getContext('2d');

let mediaStream     = null;   
let frameTimer      = null;   
let lastSpokenText  = '';     
let detectedWords   = [];     


async function startCamera() {
  try {
    spinner.classList.add('visible');

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },   
        height: { ideal: 480 },
        facingMode: 'user'        
      }
    });

    videoEl.srcObject = mediaStream;

    await new Promise((resolve) => {
      videoEl.onloadedmetadata = resolve;
    });

    videoEl.style.display = 'block';
    placeholder.style.display = 'none';
    spinner.classList.remove('visible');

    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;

    updateStatus(true, 'Camera active — detecting gestures...');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    recBadge.classList.add('visible');

    startFrameLoop();

  } catch (error) {
   
    spinner.classList.remove('visible');
    alert('Camera access failed: ' + error.message);
    console.error('Camera error:', error);
  }
}

function stopCamera() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  videoEl.style.display = 'none';
  placeholder.style.display = 'flex';

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  
  updateStatus(false, 'Camera stopped');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  recBadge.classList.remove('visible');
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);


function startFrameLoop() {
  frameTimer = setInterval(captureAndSend, FRAME_INTERVAL_MS);
}

async function captureAndSend() {
  if (!videoEl.srcObject || videoEl.readyState < 2) return;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoEl.videoWidth;
  tempCanvas.height = videoEl.videoHeight;

  const tempCtx = tempCanvas.getContext('2d');

  tempCtx.drawImage(videoEl, 0, 0);

  const base64Image = tempCanvas.toDataURL('image/jpeg', 0.7);

  try {
    const response = await fetch(`${BACKEND_URL}/api/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });

    if (!response.ok) throw new Error('Backend error');

    const result = await response.json();

    updateOutput(result);

  } catch (err) {
    console.warn('Frame send failed:', err.message);
  }
}

function updateOutput(result) {
  const { gesture, confidence, landmarks } = result;

  if (!gesture || gesture === 'none') {
    gestureDisplay.textContent = '—';
    confidenceVal.textContent = '0%';
    confidenceBar.style.width = '0%';
    return;
  }

  gestureDisplay.textContent = gesture;

  gestureDisplay.classList.remove('flash');
  void gestureDisplay.offsetWidth; 
  gestureDisplay.classList.add('flash');

  const percent = Math.round(confidence * 100);
  confidenceVal.textContent = `${percent}%`;
  confidenceBar.style.width = `${percent}%`;

  if (confidence > 0.75) {
    const lastWord = detectedWords[detectedWords.length - 1];

    if (gesture !== lastWord) {
      detectedWords.push(gesture);
      updateTextOutput();
    }
  }

  if (landmarks && landmarks.length > 0) {
    drawLandmarks(landmarks);
  }
}

function updateTextOutput() {
  if (detectedWords.length === 0) {
    textOutput.innerHTML = '<span class="text-placeholder">Detected gestures will appear here...</span>';
    return;
  }
  textOutput.textContent = detectedWords.join(' ');
}


function drawLandmarks(landmarks) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const w = canvasEl.width;
  const h = canvasEl.height;
  const connections = [
    [0,1],[1,2],[2,3],[3,4],           
    [0,5],[5,6],[6,7],[7,8],           
    [0,9],[9,10],[10,11],[11,12],      
    [0,13],[13,14],[14,15],[15,16],    
    [0,17],[17,18],[18,19],[19,20],    
    [5,9],[9,13],[13,17]               
  ];

  ctx.strokeStyle = 'rgba(99, 102, 241, 0.8)';
  ctx.lineWidth = 2;

  connections.forEach(([a, b]) => {
    const ptA = landmarks[a];
    const ptB = landmarks[b];
    if (!ptA || !ptB) return;

    ctx.beginPath();
    ctx.moveTo(ptA.x * w, ptA.y * h);
    ctx.lineTo(ptB.x * w, ptB.y * h);
    ctx.stroke();
  });

  landmarks.forEach((pt) => {
    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#6366f1';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}


function speak(text) {
  if (!text || text.trim() === '') return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);

  // Configure voice properties
  utterance.rate   = 0.9;   
  utterance.pitch  = 1.0;   
  utterance.volume = 1.0;   
  utterance.lang   = 'en-US';

  const voices = window.speechSynthesis.getVoices();
  const preferredVoice = voices.find(v =>
    v.name.includes('Google') || v.name.includes('Samantha')
  );
  if (preferredVoice) utterance.voice = preferredVoice;

  lastSpokenText = text;

  
  window.speechSynthesis.speak(utterance);
}

speakBtn.addEventListener('click', () => {
  const text = detectedWords.join(' ');
  if (!text) return alert('No text to speak yet!');
  speak(text);
});

replayBtn.addEventListener('click', () => {
  if (!lastSpokenText) return alert('Nothing to replay yet!');
  speak(lastSpokenText);
});

clearBtn.addEventListener('click', () => {
  detectedWords = [];
  lastSpokenText = '';
  gestureDisplay.textContent = '—';
  confidenceVal.textContent = '0%';
  confidenceBar.style.width = '0%';
  updateTextOutput();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
});


themeToggle.addEventListener('click', () => {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';

  // Switch theme
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');

  // Update button icon
  themeToggle.querySelector('.toggle-icon').textContent = isDark ? '☀️' : '🌙';
});

function updateStatus(isActive, message) {
  statusDot.className = 'status-dot' + (isActive ? ' active' : '');
  statusText.textContent = message;
}


stopBtn.disabled = true;
console.log('SignSpeak app loaded ✅');
