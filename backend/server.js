// ═══════════════════════════════════════════════════════════
// SignSpeak — server.js (Node.js + Express backend)
//
// What this server does:
//   1. Listens for POST requests from the frontend (browser)
//   2. Receives the base64 image frame
//   3. Forwards it to the Python AI module (via HTTP)
//   4. Returns the Python prediction back to the browser
//
// Think of Node.js as the "middleman" between browser and Python.
// ═══════════════════════════════════════════════════════════

// ── IMPORTS ────────────────────────────────────────────────
// express: the web framework that lets us create API endpoints
const express = require('express');

// cors: allows our frontend (different origin) to call this server
// Without CORS, browsers block cross-origin requests for security
const cors = require('cors');

// axios: HTTP client to make requests from Node.js to Python
const axios = require('axios');

// path: Node.js built-in module for file paths
const path = require('path');

// ── SETUP ──────────────────────────────────────────────────
const app = express();

// Python AI server address — must match where Python runs
const PYTHON_URL = 'http://localhost:5001';

// Our server will listen on port 3000
const PORT = 3000;

// ── MIDDLEWARE ─────────────────────────────────────────────
// Middleware runs on EVERY request before your route handlers.

// 1. CORS — allows the browser to call our API
//    Without this: browser shows "CORS error" and blocks the request
app.use(cors());

// 2. express.json() — parses incoming JSON request bodies
//    Without this: req.body would be undefined
//    We increase the limit to 10mb because base64 images are large
app.use(express.json({ limit: '10mb' }));

// 3. Serve frontend files statically
//    This lets you open http://localhost:3000 to see the UI
app.use(express.static(path.join(__dirname, '../frontend')));

// ── ROUTES ─────────────────────────────────────────────────
// Routes define what happens when someone calls a URL on our server.

// ── Health check route ──
// Visit http://localhost:3000/api/health to confirm server is running
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'SignSpeak backend is running!',
    timestamp: new Date().toISOString()
  });
});

// ── Main detection route ──
// Frontend POSTs a base64 image here every 500ms
// We forward it to Python, get the prediction, return it to browser
app.post('/api/detect', async (req, res) => {

  // req.body.image contains the base64-encoded JPEG from the browser
  const { image } = req.body;

  // Validate — make sure image data was sent
  if (!image) {
    return res.status(400).json({ error: 'No image data received' });
  }

  try {
    // Forward the image to our Python server
    // Python is running its own HTTP server on port 5001
    // axios.post() makes an HTTP POST request from Node.js to Python
    const pythonResponse = await axios.post(
      `${PYTHON_URL}/predict`,
      { image },                          // body sent to Python
      { timeout: 5000 }                   // 5 second timeout
    );

    // pythonResponse.data contains the prediction from Python
    // e.g. { gesture: "Hello", confidence: 0.92, landmarks: [...] }
    const prediction = pythonResponse.data;

    // Send the prediction back to the browser
    res.json(prediction);

  } catch (error) {

    // If Python server is not running or returned an error
    if (error.code === 'ECONNREFUSED') {
      // Python is not running — return a mock response so the UI still works
      console.warn('⚠️  Python server not running — returning mock response');
      return res.json(getMockPrediction());
    }

    console.error('Python error:', error.message);
    res.status(500).json({ error: 'Prediction failed', details: error.message });
  }
});

// ── MOCK PREDICTION (for testing without Python) ────────────
// This lets you test the frontend and backend connection
// even before the Python AI module is set up.
function getMockPrediction() {
  const gestures = ['Hello', 'Yes', 'No', 'Thanks', 'A', 'B', 'C', 'none'];
  const weights  = [0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25]; // 'none' most common

  // Weighted random selection
  const rand = Math.random();
  let cumulative = 0;
  let selected = 'none';

  for (let i = 0; i < gestures.length; i++) {
    cumulative += weights[i];
    if (rand < cumulative) {
      selected = gestures[i];
      break;
    }
  }

  const confidence = selected === 'none' ? 0 : 0.6 + Math.random() * 0.35;

  return {
    gesture: selected,
    confidence: parseFloat(confidence.toFixed(2)),
    landmarks: [],    // no real landmarks in mock mode
    mock: true        // flag to indicate this is fake data
  };
}

// ── START SERVER ───────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════╗
║   SignSpeak Backend — Running ✅      ║
╠══════════════════════════════════════╣
║  Local:   http://localhost:${PORT}       ║
║  API:     http://localhost:${PORT}/api   ║
║  Python:  expects http://localhost:5001 ║
╚══════════════════════════════════════╝
  `);
});
