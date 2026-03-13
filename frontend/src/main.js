// ═══════════════════════════════════════════════════════════════
//  Handwritten Mathematical Equation Solver — Frontend
// ═══════════════════════════════════════════════════════════════

// ─── Configuration ─────────────────────────────────────────────
const API_BASE = ""; // Same-origin (Flask serves frontend)
const IDLE_DELAY = 800; // ms of inactivity before auto-predict
const LINE_WIDTH = 4;
const STROKE_COLOR = "#333";
const DIGITAL_FONT = "'Cambria Math', 'Latin Modern Math', Georgia, serif";

// ─── DOM Elements ──────────────────────────────────────────────
const displayCanvas = document.getElementById("display-canvas");
const drawingCanvas = document.getElementById("drawing-canvas");
const displayCtx = displayCanvas.getContext("2d");
const drawCtx = drawingCanvas.getContext("2d");
const equationDisplay = document.getElementById("equation-display");
const equationResult = document.getElementById("equation-result");
const statusText = document.getElementById("status-text");
const confidenceText = document.getElementById("confidence-text");
const solveBtn = document.getElementById("solveBtn");
const undoBtn = document.getElementById("undoBtn");
const clearBtn = document.getElementById("clearBtn");

// ─── State ─────────────────────────────────────────────────────
let isDrawing = false;
let lastPoint = null;
let currentStroke = [];
let pendingStrokes = []; // Strokes not yet sent for prediction
let recognizedChars = []; // {display, math, bbox, isSuperscript}
let idleTimer = null;
let predicting = false; // Lock to prevent overlapping requests

// ─── Canvas Sizing ─────────────────────────────────────────────
function resizeCanvases() {
  const container = document.getElementById("canvas-container");
  const w = container.clientWidth;
  const h = container.clientHeight;

  if (displayCanvas.width === w && displayCanvas.height === h) return;

  displayCanvas.width = w;
  displayCanvas.height = h;
  drawingCanvas.width = w;
  drawingCanvas.height = h;

  redrawAll();
}

window.addEventListener("resize", resizeCanvases);
requestAnimationFrame(resizeCanvases); // Initial sizing after layout

// ─── Drawing Events ────────────────────────────────────────────
drawingCanvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  isDrawing = true;
  lastPoint = [e.offsetX, e.offsetY];
  currentStroke = [{ x: e.offsetX, y: e.offsetY }];
  clearIdleTimer();
});

drawingCanvas.addEventListener("mousemove", (e) => {
  if (!isDrawing || !lastPoint) return;

  drawCtx.beginPath();
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = STROKE_COLOR;
  drawCtx.lineWidth = LINE_WIDTH;
  drawCtx.moveTo(lastPoint[0], lastPoint[1]);
  drawCtx.lineTo(e.offsetX, e.offsetY);
  drawCtx.stroke();

  lastPoint = [e.offsetX, e.offsetY];
  currentStroke.push({ x: e.offsetX, y: e.offsetY });
});

drawingCanvas.addEventListener("mouseup", (e) => {
  if (e.button !== 0 || !isDrawing) return;
  finishStroke();
});

drawingCanvas.addEventListener("mouseleave", () => {
  if (isDrawing) finishStroke();
});

function finishStroke() {
  isDrawing = false;
  lastPoint = null;

  if (currentStroke.length > 1) {
    pendingStrokes.push(currentStroke);
  }
  currentStroke = [];

  startIdleTimer();
}

// ─── Idle Timer ────────────────────────────────────────────────
function clearIdleTimer() {
  if (idleTimer) {
    clearTimeout(idleTimer);
    idleTimer = null;
  }
}

function startIdleTimer() {
  clearIdleTimer();
  if (pendingStrokes.length === 0) return;

  idleTimer = setTimeout(() => {
    sendForPrediction();
  }, IDLE_DELAY);
}

// ─── API: Predict ──────────────────────────────────────────────
async function sendForPrediction() {
  if (pendingStrokes.length === 0 || predicting) return;
  predicting = true;

  const strokesToSend = [...pendingStrokes];

  // Previous character context for superscript detection
  let previousCharacter = null;
  if (recognizedChars.length > 0) {
    const last = recognizedChars[recognizedChars.length - 1];
    previousCharacter = { bbox: last.bbox };
  }

  setStatus("Recognizing...", "");

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strokes: strokesToSend,
        previousCharacter: previousCharacter,
      }),
    });

    const result = await response.json();

    if (!response.ok) {
      setStatus(result.error || "Server error", "", true);
      predicting = false;
      return;
    }

    if (result.recognized) {
      // Success: add character, clear pending, update display
      recognizedChars.push({
        display: result.display,
        math: result.mathChar,
        bbox: result.bbox,
        isSuperscript: result.isSuperscript,
      });

      pendingStrokes = [];
      updateEquationDisplay();
      redrawAll();

      const supTag = result.isSuperscript ? " [superscript]" : "";
      setStatus(
        `Recognized: ${result.display}${supTag}`,
        `Confidence: ${(result.confidence * 100).toFixed(0)}%`,
        false,
      );
    } else {
      // Not recognized: erase scribble, show message
      pendingStrokes = [];
      redrawAll();
      setStatus(
        "Not Recognized \u2014 try again",
        `Confidence: ${(result.confidence * 100).toFixed(0)}%`,
        true,
      );
    }
  } catch (err) {
    setStatus("Connection error \u2014 is the backend running?", "", true);
    console.error(err);
  }

  predicting = false;
}

// ─── API: Solve ────────────────────────────────────────────────
async function solveEquation() {
  const mathStr = buildMathString();
  if (!mathStr) {
    setStatus("Nothing to solve", "", true);
    return;
  }

  setStatus("Solving...", "");

  try {
    const response = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ equation: mathStr }),
    });

    const result = await response.json();

    if (result.success) {
      equationResult.textContent = "  " + result.result;
      setStatus("Solved!", "", false);
    } else {
      equationResult.textContent = "";
      setStatus(result.error, "", true);
    }
  } catch (err) {
    setStatus("Connection error \u2014 is the backend running?", "", true);
    console.error(err);
  }
}

// ─── Equation String Building ──────────────────────────────────
function buildDisplayHTML() {
  let html = "";
  let inSup = false;

  for (const ch of recognizedChars) {
    if (ch.isSuperscript) {
      if (!inSup) {
        html += "<sup>";
        inSup = true;
      }
      html += escapeHTML(ch.display);
    } else {
      if (inSup) {
        html += "</sup>";
        inSup = false;
      }
      html += escapeHTML(ch.display);
    }
  }
  if (inSup) html += "</sup>";

  return html;
}

function buildMathString() {
  let str = "";
  let inSup = false;

  for (const ch of recognizedChars) {
    if (ch.isSuperscript) {
      if (!inSup) {
        str += "^(";
        inSup = true;
      }
      str += ch.math;
    } else {
      if (inSup) {
        str += ")";
        inSup = false;
      }
      str += ch.math;
    }
  }
  if (inSup) str += ")";

  return str;
}

function escapeHTML(str) {
  const el = document.createElement("span");
  el.textContent = str;
  return el.innerHTML;
}

function updateEquationDisplay() {
  equationDisplay.innerHTML = buildDisplayHTML();
  equationResult.textContent = "";
}

// ─── Canvas Redraw ─────────────────────────────────────────────
function redrawAll() {
  const w = drawingCanvas.width;
  const h = drawingCanvas.height;

  // Clear both layers
  drawCtx.clearRect(0, 0, w, h);
  displayCtx.clearRect(0, 0, w, h);

  // Render digital characters on the display layer
  for (const ch of recognizedChars) {
    drawDigitalChar(ch);
  }

  // Re-render pending ink strokes on the drawing layer
  for (const stroke of pendingStrokes) {
    drawInkStroke(stroke);
  }
}

function drawDigitalChar(ch) {
  const bbox = ch.bbox;
  const bboxH = bbox.maxY - bbox.minY;
  const centerX = (bbox.minX + bbox.maxX) / 2;
  const centerY = (bbox.minY + bbox.maxY) / 2;

  const fontSize = Math.max(18, Math.min(bboxH * 0.85, 120));

  displayCtx.save();
  displayCtx.font = `${fontSize}px ${DIGITAL_FONT}`;
  displayCtx.fillStyle = "#1a1a2e";
  displayCtx.textAlign = "center";
  displayCtx.textBaseline = "middle";
  displayCtx.fillText(ch.display, centerX, centerY);
  displayCtx.restore();
}

function drawInkStroke(stroke) {
  if (stroke.length < 2) return;

  drawCtx.beginPath();
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = STROKE_COLOR;
  drawCtx.lineWidth = LINE_WIDTH;
  drawCtx.moveTo(stroke[0].x, stroke[0].y);

  for (let i = 1; i < stroke.length; i++) {
    drawCtx.lineTo(stroke[i].x, stroke[i].y);
  }
  drawCtx.stroke();
}

// ─── Status Bar ────────────────────────────────────────────────
function setStatus(text, confidence, isError) {
  statusText.textContent = text;
  if (isError) {
    statusText.className = "error";
  } else if (text.includes("Recognized") || text.includes("Solved")) {
    statusText.className = "success";
  } else {
    statusText.className = "";
  }
  confidenceText.textContent = confidence || "";
}

// ─── Button Handlers ───────────────────────────────────────────
solveBtn.addEventListener("click", solveEquation);

undoBtn.addEventListener("click", () => {
  if (recognizedChars.length > 0) {
    recognizedChars.pop();
    updateEquationDisplay();
    redrawAll();
    setStatus("Undone last character", "");
  }
});

clearBtn.addEventListener("click", () => {
  recognizedChars = [];
  pendingStrokes = [];
  currentStroke = [];
  clearIdleTimer();
  updateEquationDisplay();
  equationResult.textContent = "";
  redrawAll();
  setStatus("Cleared", "");
});
