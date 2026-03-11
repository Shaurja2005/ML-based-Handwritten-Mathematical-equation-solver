const canvas = document.getElementById("graphics");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBTN");
ctx.lineJoin = "round";

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let isClicked = false;
let lineWidth = 10;
let temp = null;
let strokes = [];
let currentStroke = [];
let drawingStartTime = null;

function reportWindowSize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

window.addEventListener("resize", reportWindowSize);

canvas.addEventListener("mousedown", (event) => {
  if (event.button === 0 && !isClicked) {
    isClicked = true;
    temp = [event.offsetX, event.offsetY];

    if (drawingStartTime === null) {
      drawingStartTime = performance.now();
    }

    currentStroke = [];
    const t = Math.round(performance.now() - drawingStartTime);
    currentStroke.push({ x: event.offsetX, y: event.offsetY, t: t });
  }
});

canvas.addEventListener("mouseup", (event) => {
  if (event.button === 0 && isClicked) {
    isClicked = false;
    temp = null;

    if (currentStroke.length > 0) {
      strokes.push(currentStroke);
      currentStroke = [];
    }
  }
});

canvas.addEventListener("mouseleave", () => {
  if (isClicked) {
    isClicked = false;
    temp = null;

    // Save the stroke when leaving canvas
    if (currentStroke.length > 0) {
      strokes.push(currentStroke);
      currentStroke = [];
    }
  }
});

canvas.addEventListener("mousemove", (event) => {
  if (isClicked && temp) {
    ctx.beginPath();
    ctx.lineCap = "round";
    ctx.moveTo(temp[0], temp[1]);
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    temp = [event.offsetX, event.offsetY];

    const t = Math.round(performance.now() - drawingStartTime);
    currentStroke.push({ x: event.offsetX, y: event.offsetY, t: t });
  }
});

canvas.addEventListener("wheel", (event) => {
  if (event.deltaY > 0 && lineWidth > 1) {
    lineWidth--;
  } else if (event.deltaY < 0 && lineWidth < 10) {
    lineWidth++;
  }
});

clearBtn.addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  strokes = [];
  currentStroke = [];
  drawingStartTime = null;
  console.log("Canvas and strokes cleared");
});
