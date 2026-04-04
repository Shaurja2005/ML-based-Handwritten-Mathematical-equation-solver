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
const themeToggleBtn = document.getElementById("themeToggleBtn");
const historyBtn = document.getElementById("historyBtn");
const plotBtn = document.getElementById("plotBtn");
const historySidebar = document.getElementById("history-sidebar");
const closeHistoryBtn = document.getElementById("closeHistoryBtn");
const historyList = document.getElementById("history-list");
const plotModal = document.getElementById("plot-modal");
const closePlotBtn = document.getElementById("closePlotBtn");
const plotCanvas = document.getElementById("plot-canvas");

// ─── State ─────────────────────────────────────────────────────
let isDrawing = false;
let lastPoint = null;
let currentStroke = [];
let pendingStrokes = []; // Strokes not yet sent for prediction
let recognizedChars = []; // {display, math, bbox, isSuperscript}
let idleTimer = null;
let predicting = false; // Lock to prevent overlapping requests
let finalSolution = null; // Store solved result to display on canvas
let solveHistory = []; // Array to store past solutions
let currentPlotChart = null; // Store current Chart.js instance

// ─── Theme Toggle ──────────────────────────────────────────────
let isDarkMode = localStorage.getItem("theme") === "dark";
if (isDarkMode) document.body.classList.add("dark-mode");
updateThemeButtonText();

themeToggleBtn.addEventListener("click", () => {
  isDarkMode = !isDarkMode;
  if (isDarkMode) {
    document.body.classList.add("dark-mode");
    localStorage.setItem("theme", "dark");
  } else {
    document.body.classList.remove("dark-mode");
    localStorage.setItem("theme", "light");
  }
  updateThemeButtonText();
  redrawAll();
});

function updateThemeButtonText() {
  themeToggleBtn.textContent = isDarkMode ? "☀️ Light Mode" : "🌙 Dark Mode";
}

function getDigitalTextColor() {
  return isDarkMode ? "#ffffff" : "#1a1a2e";
}

function getStrokeColor() {
  return isDarkMode ? "#e5e5ea" : "#333333";
}

// ─── History Sidebar ───────────────────────────────────────────
historyBtn.addEventListener("click", () => {
  historySidebar.classList.toggle("hidden");
});

closeHistoryBtn.addEventListener("click", () => {
  historySidebar.classList.add("hidden");
});

function addToHistory(mathStr, result, solutionState) {
  solveHistory.unshift({
    mathStr,
    result,
    chars: JSON.parse(JSON.stringify(recognizedChars)),
    solutionState,
  });
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = "";
  if (solveHistory.length === 0) {
    historyList.innerHTML =
      "<li style='padding: 16px; color: #8e8e93; font-size: 14px;'>No past solutions yet.</li>";
    return;
  }

  solveHistory.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "history-item";

    // Create elements to match UI style
    const mathDiv = document.createElement("div");
    mathDiv.className = "history-item-math";
    mathDiv.textContent = item.mathStr;

    const resultDiv = document.createElement("div");
    resultDiv.className = "history-item-result";
    resultDiv.textContent = item.result;

    li.appendChild(mathDiv);
    li.appendChild(resultDiv);

    // Restore logic on click
    li.addEventListener("click", () => {
      recognizedChars = JSON.parse(JSON.stringify(item.chars));
      finalSolution = item.solutionState;
      pendingStrokes = [];
      updateEquationDisplay();
      equationResult.textContent = "  " + item.result;
      redrawAll();
      setStatus("Restored from history", "", false);
      if (window.innerWidth < 768) {
        historySidebar.classList.add("hidden"); // Auto-close on mobile mapping
      }
    });

    historyList.appendChild(li);
  });
}
renderHistory();

// ─── Canvas Sizing ─────────────────────────────────────────────
function resizeCanvases() {
  const container = document.getElementById("canvas-container");
  const w = container.clientWidth;
  const h = container.clientHeight;

  // Ensure canvas width can grow but doesn't instantly shrink
  const targetW = Math.max(w, displayCanvas.width || w);

  if (displayCanvas.width === targetW && displayCanvas.height === h) return;

  displayCanvas.width = targetW;
  displayCanvas.height = h;
  drawingCanvas.width = targetW;
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
  drawCtx.strokeStyle = getStrokeColor();
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
  finalSolution = null; // Clear previous solution if the user inputs more characters

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
      // The backend already returns "= <value>" for arithmetic or "x = <value>" for algebra,
      // so we don't need to prepend an extra equals sign.
      let prefix = "";
      // If the result is an algebraic solve like "x = 3", we add an arrow instead of equals
      if (!result.result.trim().startsWith("=")) {
        prefix = "\u21d2 "; // =>
      }
      finalSolution = prefix + result.result;

      // Save completely structured history
      addToHistory(mathStr, result.result, finalSolution);

      redrawAll();
      setStatus("Solved!", "", false);
    } else {
      equationResult.textContent = "";
      finalSolution = null;
      redrawAll();
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
    const shouldSuperscript = ch.isSuperscript && isExponentEligible(ch.math);

    if (shouldSuperscript) {
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
    const shouldSuperscript = ch.isSuperscript && isExponentEligible(ch.math);

    if (shouldSuperscript) {
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

function isExponentEligible(token) {
  return /^[0-9a-zA-Z]$/.test(token);
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
  const container = document.getElementById("canvas-container");
  const w = container.clientWidth;
  const h = container.clientHeight;

  // Compute common baseline and base font size for relative sizing/alignment
  let baseSize = 60; // fallback
  let commonCenterY = h / 2; // fallback

  const validChars = recognizedChars.filter(
    (c) => !c.isSuperscript && /[0-9a-zA-Z]/.test(c.math),
  );
  const charsToMeasure =
    validChars.length > 0
      ? validChars
      : recognizedChars.filter((c) => !c.isSuperscript);

  if (charsToMeasure.length > 0) {
    let sizeSum = 0;
    let centerYSum = 0;
    for (const c of charsToMeasure) {
      sizeSum += Math.max(c.bbox.maxX - c.bbox.minX, c.bbox.maxY - c.bbox.minY);
      centerYSum += (c.bbox.minY + c.bbox.maxY) / 2;
    }
    baseSize = sizeSum / charsToMeasure.length;
    commonCenterY = centerYSum / charsToMeasure.length;
  }

  // -- Pre-calculate required canvas width to add padding and avoid cutoff
  let requiredWidth = w;
  let startX = recognizedChars.length > 0 ? recognizedChars[0].bbox.minX : 50;
  if (startX < 50) startX = 50; // enforce left padding
  let currentX = startX;

  displayCtx.save();
  for (const ch of recognizedChars) {
    let fontSize = baseSize * 0.85;
    if (ch.isSuperscript) fontSize *= 0.6;
    fontSize = Math.max(18, Math.min(fontSize, 120));
    displayCtx.font = `${fontSize}px ${DIGITAL_FONT}`;
    currentX += displayCtx.measureText(ch.display).width + fontSize * 0.15;
  }

  if (finalSolution) {
    let fontSize = Math.max(18, Math.min(baseSize * 0.85, 120));
    displayCtx.font = `bold ${fontSize}px ${DIGITAL_FONT}`;
    currentX +=
      fontSize * 0.5 + displayCtx.measureText(finalSolution).width + 100; // 100 right padding
  }
  displayCtx.restore();

  requiredWidth = Math.max(w, currentX);

  if (displayCanvas.width !== requiredWidth) {
    displayCanvas.width = requiredWidth;
    drawingCanvas.width = requiredWidth;
  }

  // Clear both layers
  drawCtx.clearRect(0, 0, requiredWidth, h);
  displayCtx.clearRect(0, 0, requiredWidth, h);

  // Render digital characters on the display layer
  currentX = startX;
  for (const ch of recognizedChars) {
    currentX = drawDigitalChar(ch, baseSize, commonCenterY, currentX);
  }

  // Draw final solution if available
  if (finalSolution) {
    drawFinalSolution(finalSolution, baseSize, commonCenterY, currentX);
  }

  // Re-render pending ink strokes on the drawing layer
  for (const stroke of pendingStrokes) {
    drawInkStroke(stroke);
  }
}

function drawFinalSolution(solutionText, baseSize, commonCenterY, finalStartX) {
  let fontSize = Math.max(18, Math.min(baseSize * 0.85, 120));

  displayCtx.save();
  displayCtx.font = `bold ${fontSize}px ${DIGITAL_FONT}`;
  displayCtx.fillStyle = "#e63946"; // Give it a distinctive color like red

  // Give a little extra padding before the "=" sign
  let drawX = finalStartX + fontSize * 0.5;
  let drawY = commonCenterY;

  // We don't shift digits or wrap text anymore, just draw it straight.
  // The canvas will naturally scroll horizontally due to pre-calculated resize.
  displayCtx.textAlign = "left";
  displayCtx.textBaseline = "middle";
  displayCtx.fillText(solutionText, drawX, drawY);
  displayCtx.restore();
}

function drawDigitalChar(ch, baseSize, commonCenterY, startX) {
  // Normalize font size to baseSize
  let fontSize = baseSize * 0.85;
  let drawY = commonCenterY;

  // Superscripts are smaller and higher up
  if (ch.isSuperscript) {
    fontSize *= 0.6;
    drawY = commonCenterY - baseSize * 0.4;
  }

  // Clamp font size
  fontSize = Math.max(18, Math.min(fontSize, 120));

  displayCtx.save();
  displayCtx.font = `${fontSize}px ${DIGITAL_FONT}`;

  const textWidth = displayCtx.measureText(ch.display).width;
  const drawX = startX + textWidth / 2;

  displayCtx.fillStyle = getDigitalTextColor();
  displayCtx.textAlign = "center";
  displayCtx.textBaseline = "middle";
  // Place character sequentially rather than at original drawing position
  displayCtx.fillText(ch.display, drawX, drawY);
  displayCtx.restore();

  // Return the next X position including a small gap
  return startX + textWidth + fontSize * 0.15;
}

function drawInkStroke(stroke) {
  if (stroke.length < 2) return;

  drawCtx.beginPath();
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = getStrokeColor();
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

// ─── API: Plot ──────────────────────────────────────────────────
async function plotEquation() {
  const mathStr = buildMathString();
  if (!mathStr) {
    setStatus("Nothing to plot", "", true);
    return;
  }

  setStatus("Generating plot...", "");

  try {
    const response = await fetch(`${API_BASE}/api/plot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ equation: mathStr }),
    });

    const result = await response.json();

    if (result.success) {
      setStatus("Plot generated!", "", false);
      addToHistory(mathStr, "Graph plotted", "\u21d2 Plot generated");
      
      // Destroy previous chart if exists
      if (currentPlotChart) {
        currentPlotChart.destroy();
      }

      plotModal.classList.remove("hidden");

      const ctx = plotCanvas.getContext("2d");
      
      const gridColor = isDarkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";
      const textColor = isDarkMode ? "#e5e5ea" : "#8e8e93";

      currentPlotChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: result.x,
          datasets: [{
            label: result.label || "f(x)",
            data: result.y,
            borderColor: "#0a84ff",
            backgroundColor: "rgba(10, 132, 255, 0.1)",
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          interaction: { intersect: false, mode: 'index' },
          plugins: {
            legend: { labels: { color: textColor } }
          },
          scales: {
            x: {
              type: 'linear',
              position: 'bottom',
              grid: { color: gridColor },
              ticks: { color: textColor }
            },
            y: {
              type: 'linear',
              grid: { color: gridColor },
              ticks: { color: textColor }
            }
          }
        }
      });
    } else {
      setStatus(result.error, "", true);
    }
  } catch (err) {
    setStatus("Error generating plot \u2014 is the backend running?", "", true);
    console.error(err);
  }
}

// ─── Button Handlers ───────────────────────────────────────────
plotBtn.addEventListener("click", plotEquation);

closePlotBtn.addEventListener("click", () => {
  plotModal.classList.add("hidden");
});

solveBtn.addEventListener("click", solveEquation);

undoBtn.addEventListener("click", () => {
  if (recognizedChars.length > 0) {
    recognizedChars.pop();
    finalSolution = null; // Clear if we undo a character
    updateEquationDisplay();
    redrawAll();
    setStatus("Undone last character", "");
  }
});

clearBtn.addEventListener("click", () => {
  recognizedChars = [];
  pendingStrokes = [];
  currentStroke = [];
  finalSolution = null; // Clear solution
  clearIdleTimer();
  updateEquationDisplay();
  equationResult.textContent = "";
  redrawAll();
  setStatus("Cleared", "");
});
