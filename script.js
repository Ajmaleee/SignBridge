/* =========================================================
   SignSpeak — prototype application logic
   Sections:
     1. Camera initialization
     2. MediaPipe (HandLandmarker) initialization
     3. Landmark processing / drawing
     4. Gesture classification
     5. Temporal smoothing
     6. UI updates
     7. Text-to-speech
     8. Settings / status panel
   ========================================================= */

import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ---------- DOM references ---------- */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const cameraToggle = document.getElementById("cameraToggle");
const cameraToggleLabel = document.getElementById("cameraToggleLabel");

const stageStatusDot = document.getElementById("stageStatusDot");
const stageStatusText = document.getElementById("stageStatusText");
const stageHandBadge = document.getElementById("stageHandBadge");
const stageEmpty = document.getElementById("stageEmpty");
const stageEmptyText = document.getElementById("stageEmptyText");
const stageError = document.getElementById("stageError");
const stageErrorText = document.getElementById("stageErrorText");
const fpsLine = document.getElementById("fpsLine");

const topbarCamDot = document.getElementById("topbarCamDot");
const topbarCamLabel = document.getElementById("topbarCamLabel");

const wordState = document.getElementById("wordState");
const wordText = document.getElementById("wordText");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const gestureCaptionValue = document.getElementById("gestureCaptionValue");

const clearBtn = document.getElementById("clearBtn");
const speakBtn = document.getElementById("speakBtn");
const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");

const settingsBtn = document.getElementById("settingsBtn");
const statusToggle = document.getElementById("statusToggle");
const statusBody = document.getElementById("statusBody");

const statCamera = document.getElementById("statCamera");
const statTracking = document.getElementById("statTracking");
const statHands = document.getElementById("statHands");
const statRecognition = document.getElementById("statRecognition");
const statFps = document.getElementById("statFps");
const statGesture = document.getElementById("statGesture");
const statConfidence = document.getElementById("statConfidence");
const statHandedness = document.getElementById("statHandedness");

const toast = document.getElementById("toast");

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- App state ---------- */

const state = {
  cameraOn: false,
  stream: null,
  landmarker: null,
  landmarkerReady: false,
  running: false,
  lastVideoTime: -1,
  currentSpokenWord: null,
  frameTimes: [],
};

/* Hand skeleton connections (MediaPipe Hands topology) */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
  [0, 17],                                  // palm base
];

const FINGERTIPS = [4, 8, 12, 16, 20];

/* =========================================================
   1. CAMERA INITIALIZATION
   ========================================================= */

async function enableCamera() {
  hideError();
  cameraToggle.disabled = true;
  cameraToggleLabel.textContent = "Starting…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });

    state.stream = stream;
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });
    await video.play();

    resizeOverlay();

    state.cameraOn = true;
    cameraToggle.setAttribute("aria-pressed", "true");
    cameraToggleLabel.textContent = "Disable camera";
    setStageStatus(true);
    stageEmpty.style.display = "none";
    setHandBadge("Show your hand inside the camera frame.", "info");
    setToast("Camera started");

    if (!state.landmarker) {
      await initHandLandmarker();
    }

    state.running = true;
    requestAnimationFrame(processFrame);
  } catch (err) {
    console.error("Camera error:", err);
    handleCameraError(err);
  } finally {
    cameraToggle.disabled = false;
  }
}

function disableCamera() {
  state.running = false;
  state.cameraOn = false;

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  video.srcObject = null;

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  cameraToggle.setAttribute("aria-pressed", "false");
  cameraToggleLabel.textContent = "Enable camera";
  setStageStatus(false);
  stageEmpty.style.display = "flex";
  stageEmptyText.textContent = "Camera is off. Turn it on to start recognizing gestures.";
  setHandBadge("Turn on the camera to begin", "info");
  fpsLine.textContent = "FPS: —";

  resetRecognitionUI();
  updateStatusPanel({ camera: "DISCONNECTED", tracking: "INACTIVE", hands: 0, recognition: "INACTIVE", fps: 0, gesture: "—", confidence: 0, handedness: "—" });
}

function handleCameraError(err) {
  let message = "Hand tracking temporarily unavailable.";

  if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
    message = "Camera access is required for hand recognition. Please allow camera access and try again.";
  } else if (err && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
    message = "No camera was found on this device. Connect a camera and try again.";
  } else if (err && err.name === "NotReadableError") {
    message = "Your camera is already in use by another application.";
  } else if (location.protocol !== "https:" && location.hostname !== "localhost") {
    message = "Camera access requires a secure (HTTPS) connection or localhost.";
  }

  showError(message);
  cameraToggle.setAttribute("aria-pressed", "false");
  cameraToggleLabel.textContent = "Enable camera";
  setStageStatus(false);
}

cameraToggle.addEventListener("click", () => {
  if (state.cameraOn) {
    disableCamera();
  } else {
    enableCamera();
  }
});

window.addEventListener("resize", resizeOverlay);

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = video.videoWidth || rect.width;
  overlay.height = video.videoHeight || rect.height;
}

/* =========================================================
   2. MEDIAPIPE (HandLandmarker) INITIALIZATION
   ========================================================= */

async function initHandLandmarker() {
  try {
    setHandBadge("Loading hand-tracking model…", "info");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    state.landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    state.landmarkerReady = true;
    updateStatusPanel({ tracking: "ACTIVE", recognition: "ACTIVE" });
    setHandBadge("Show your hand inside the camera frame.", "info");
  } catch (err) {
    console.error("HandLandmarker init failed:", err);
    state.landmarkerReady = false;
    showError("Hand tracking temporarily unavailable. The tracking model could not be loaded (check your internet connection).");
    updateStatusPanel({ tracking: "ERROR", recognition: "INACTIVE" });
  }
}

/* =========================================================
   3. LANDMARK PROCESSING / DRAWING  +  main loop
   ========================================================= */

function processFrame() {
  if (!state.running) return;

  if (state.landmarkerReady && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;

    const nowMs = performance.now();
    const result = state.landmarker.detectForVideo(video, nowMs);
    trackFps(nowMs);

    resizeOverlay();
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      const handednessInfo = result.handedness && result.handedness[0] ? result.handedness[0][0] : null;

      drawLandmarks(landmarks);
      setHandBadge("Hand detected", "success");
      stageEmptyText.textContent = "";

      const detectionScore = handednessInfo ? handednessInfo.score : 0.85;
      const classification = classifyGesture(landmarks);
      const smoothed = pushToSmoothingBuffer(classification, detectionScore);

      updateStatusPanel({
        camera: "CONNECTED",
        tracking: "ACTIVE",
        hands: 1,
        recognition: "ACTIVE",
        gesture: smoothed.word || "…",
        confidence: Math.round((smoothed.confidence || 0) * 100),
        handedness: handednessInfo ? `${handednessInfo.categoryName} (${Math.round(handednessInfo.score * 100)}%)` : "—",
      });

      applyRecognitionResult(smoothed);
    } else {
      setHandBadge("Show your hand inside the camera frame.", "info");
      pushToSmoothingBuffer(null, 0);
      updateStatusPanel({ camera: "CONNECTED", tracking: "ACTIVE", hands: 0, recognition: "ACTIVE", gesture: "—", confidence: 0, handedness: "—" });
      applyRecognitionResult({ word: null, confidence: 0, stable: false });
    }
  }

  requestAnimationFrame(processFrame);
}

function drawLandmarks(landmarks) {
  const w = overlay.width;
  const h = overlay.height;

  overlayCtx.lineWidth = 2.5;
  overlayCtx.strokeStyle = "rgba(196, 0, 26, 0.85)";
  overlayCtx.lineCap = "round";

  // skeleton
  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = landmarks[a];
    const p2 = landmarks[b];
    overlayCtx.beginPath();
    overlayCtx.moveTo(p1.x * w, p1.y * h);
    overlayCtx.lineTo(p2.x * w, p2.y * h);
    overlayCtx.stroke();
  }

  // joints
  landmarks.forEach((pt, i) => {
    const isTip = FINGERTIPS.includes(i);
    overlayCtx.beginPath();
    overlayCtx.arc(pt.x * w, pt.y * h, isTip ? 6 : 3.5, 0, Math.PI * 2);
    overlayCtx.fillStyle = isTip ? "#ffffff" : "rgba(196, 0, 26, 0.95)";
    overlayCtx.fill();
    if (isTip) {
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeStyle = "rgba(196, 0, 26, 0.95)";
      overlayCtx.stroke();
    }
  });
}

let fpsCounter = { frames: 0, last: performance.now() };
function trackFps(nowMs) {
  fpsCounter.frames += 1;
  const elapsed = nowMs - fpsCounter.last;
  if (elapsed >= 500) {
    const fps = Math.round((fpsCounter.frames * 1000) / elapsed);
    fpsLine.textContent = `FPS: ${fps}`;
    updateStatusPanel({ fps });
    fpsCounter.frames = 0;
    fpsCounter.last = nowMs;
  }
}

/* =========================================================
   4. GESTURE CLASSIFICATION
   Uses landmark geometry only: fingertip vs. joint distances
   from the wrist, plus thumb direction relative to the wrist.
   ========================================================= */

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

/**
 * Returns which fingers are extended: [thumb, index, middle, ring, pinky]
 * Fingers (not thumb): a finger counts as extended when its fingertip is
 * meaningfully farther from the wrist than its PIP joint is — this stays
 * correct even when the hand is rotated, unlike a plain y-coordinate check.
 * Thumb: extended when the tip is meaningfully farther from the pinky-side
 * of the palm than the thumb's IP joint is (thumb moves sideways, not up).
 */
function getFingerStates(lm) {
  const wrist = lm[0];

  const fingerDefs = [
    { tip: 8, pip: 6, mcp: 5 },   // index
    { tip: 12, pip: 10, mcp: 9 }, // middle
    { tip: 16, pip: 14, mcp: 13 }, // ring
    { tip: 20, pip: 18, mcp: 17 }, // pinky
  ];

  const straightFingers = fingerDefs.map(({ tip, pip, mcp }) => {
    const tipDist = dist(lm[tip], wrist);
    const pipDist = dist(lm[pip], wrist);
    const mcpDist = dist(lm[mcp], wrist);
    // extended if tip is clearly the farthest point of the finger from the wrist
    return tipDist > pipDist * 1.08 && tipDist > mcpDist * 1.15;
  });

  // thumb: compare tip distance to the pinky MCP against the thumb IP joint
  // distance to the same reference point (captures the thumb splaying outward)
  const pinkyMcp = lm[17];
  const thumbTipDist = dist(lm[4], pinkyMcp);
  const thumbIpDist = dist(lm[3], pinkyMcp);
  const thumbExtended = thumbTipDist > thumbIpDist * 1.05;

  return [thumbExtended, ...straightFingers];
}

function getThumbDirection(lm) {
  // "up" in screen space means smaller y (video y grows downward)
  const wrist = lm[0];
  const thumbTip = lm[4];
  const dy = thumbTip.y - wrist.y;
  if (dy < -0.08) return "up";
  if (dy > 0.08) return "down";
  return "neutral";
}

const GESTURE_LABELS = {
  OPEN_PALM: "HELLO",
  FIST: "HELP",
  INDEX_ONLY: "YES",
  INDEX_MIDDLE: "TWO",
  THUMB_UP: "YES",
  THUMB_DOWN: "NO",
  THUMB_INDEX: "QUESTION",
};

function classifyGesture(landmarks) {
  const [thumb, index, middle, ring, pinky] = getFingerStates(landmarks);
  const extendedCount = [thumb, index, middle, ring, pinky].filter(Boolean).length;

  if (thumb && index && middle && ring && pinky) {
    return { label: "OPEN_PALM", word: GESTURE_LABELS.OPEN_PALM };
  }
  if (extendedCount === 0) {
    return { label: "FIST", word: GESTURE_LABELS.FIST };
  }
  if (index && middle && !ring && !pinky && !thumb) {
    return { label: "INDEX_MIDDLE", word: GESTURE_LABELS.INDEX_MIDDLE };
  }
  if (index && !middle && !ring && !pinky && !thumb) {
    return { label: "INDEX_ONLY", word: GESTURE_LABELS.INDEX_ONLY };
  }
  if (thumb && index && !middle && !ring && !pinky) {
    return { label: "THUMB_INDEX", word: GESTURE_LABELS.THUMB_INDEX };
  }
  if (thumb && !index && !middle && !ring && !pinky) {
    const dir = getThumbDirection(landmarks);
    if (dir === "up") return { label: "THUMB_UP", word: GESTURE_LABELS.THUMB_UP };
    if (dir === "down") return { label: "THUMB_DOWN", word: GESTURE_LABELS.THUMB_DOWN };
    return { label: "THUMB_NEUTRAL", word: null };
  }

  return { label: "UNRECOGNIZED", word: null };
}

/* =========================================================
   5. TEMPORAL SMOOTHING
   Rolling buffer + majority vote so the output word only
   changes once a gesture has been stable for several frames.
   ========================================================= */

const SMOOTHING_WINDOW = 15;   // frames considered (~0.4-0.6s at typical fps)
const STABILITY_RATIO = 0.66;  // fraction of the window that must agree

const smoothingBuffer = [];

function pushToSmoothingBuffer(classification, detectionScore) {
  const word = classification && classification.word ? classification.word : null;
  smoothingBuffer.push({ word, score: detectionScore || 0 });
  if (smoothingBuffer.length > SMOOTHING_WINDOW) smoothingBuffer.shift();

  // majority vote among non-null entries
  const counts = new Map();
  let scoreSum = new Map();
  for (const entry of smoothingBuffer) {
    if (!entry.word) continue;
    counts.set(entry.word, (counts.get(entry.word) || 0) + 1);
    scoreSum.set(entry.word, (scoreSum.get(entry.word) || 0) + entry.score);
  }

  let bestWord = null;
  let bestCount = 0;
  for (const [word, count] of counts) {
    if (count > bestCount) {
      bestWord = word;
      bestCount = count;
    }
  }

  const ratio = bestWord ? bestCount / smoothingBuffer.length : 0;
  const stable = bestWord && ratio >= STABILITY_RATIO;

  if (stable) {
    const avgScore = scoreSum.get(bestWord) / bestCount;
    // blend detection confidence with agreement ratio for a display value
    const confidence = Math.min(0.99, avgScore * 0.65 + ratio * 0.35);
    return { word: bestWord, confidence, stable: true };
  }

  return { word: bestWord, confidence: bestWord ? bestCount / smoothingBuffer.length : 0, stable: false };
}

/* =========================================================
   6. UI UPDATES
   ========================================================= */

let lastAcceptedWord = null;

function applyRecognitionResult({ word, confidence, stable }) {
  if (!state.cameraOn) return;

  if (!word) {
    wordState.textContent = "Recognizing…";
    gestureCaptionValue.textContent = "none";
    setConfidence(0);
    return;
  }

  if (!stable) {
    wordState.textContent = "Recognizing…";
    gestureCaptionValue.textContent = word.toLowerCase();
    setConfidence(confidence);
    return;
  }

  setConfidence(confidence);
  gestureCaptionValue.textContent = word.toLowerCase();

  if (word !== lastAcceptedWord) {
    lastAcceptedWord = word;
    wordState.textContent = "Recognized";
    wordText.textContent = word;
    speakBtn.disabled = false;
    if (!prefersReducedMotion) {
      wordText.parentElement.classList.remove("pulse-once");
      void wordText.parentElement.offsetWidth; // restart animation
      wordText.parentElement.classList.add("pulse-once");
    }
    addToHistory(word);
  } else {
    wordState.textContent = "Recognized";
  }
}

function setConfidence(value) {
  const pct = Math.round((value || 0) * 100);
  confidenceFill.style.width = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
  confidenceBar.setAttribute("aria-valuenow", String(pct));
}

function addToHistory(word) {
  if (historyEmpty) historyEmpty.remove();

  const li = document.createElement("li");
  const time = new Date();
  const timeStr = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  li.innerHTML = `<span>${word}</span><span class="hist-time">${timeStr}</span>`;
  historyList.prepend(li);

  while (historyList.children.length > 6) {
    historyList.removeChild(historyList.lastChild);
  }
}

function resetRecognitionUI() {
  wordState.textContent = "Waiting for camera";
  wordText.textContent = "—";
  setConfidence(0);
  gestureCaptionValue.textContent = "none";
  speakBtn.disabled = true;
  lastAcceptedWord = null;
  smoothingBuffer.length = 0;
}

clearBtn.addEventListener("click", () => {
  resetRecognitionUI();
  if (state.cameraOn) wordState.textContent = "Recognizing…";
  historyList.innerHTML = "";
  const li = document.createElement("li");
  li.className = "history-empty";
  li.id = "historyEmpty";
  li.textContent = "Recognized words will appear here.";
  historyList.appendChild(li);
});

function setStageStatus(on) {
  stageStatusDot.classList.toggle("dot-on", on);
  stageStatusDot.classList.toggle("dot-off", !on);
  stageStatusText.textContent = on ? "Camera active" : "Camera off";
  topbarCamDot.classList.toggle("dot-on", on);
  topbarCamDot.classList.toggle("dot-off", !on);
  topbarCamLabel.textContent = on ? "Camera on" : "Camera off";
}

function setHandBadge(text, tone) {
  stageHandBadge.textContent = text;
}

function showError(message) {
  stageError.hidden = false;
  stageErrorText.textContent = message;
  stageEmpty.style.display = "none";
}

function hideError() {
  stageError.hidden = true;
}

let toastTimer = null;
function setToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
}

/* =========================================================
   7. TEXT-TO-SPEECH
   ========================================================= */

speakBtn.addEventListener("click", () => {
  const text = wordText.textContent;
  if (!text || text === "—") return;

  if (!("speechSynthesis" in window)) {
    setToast("Speech synthesis is not supported in this browser.");
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
});

/* =========================================================
   8. SETTINGS / STATUS PANEL
   ========================================================= */

function updateStatusPanel(partial) {
  if (partial.camera !== undefined) statCamera.textContent = partial.camera;
  if (partial.tracking !== undefined) statTracking.textContent = partial.tracking;
  if (partial.hands !== undefined) statHands.textContent = String(partial.hands);
  if (partial.recognition !== undefined) statRecognition.textContent = partial.recognition;
  if (partial.fps !== undefined) statFps.textContent = String(partial.fps);
  if (partial.gesture !== undefined) statGesture.textContent = partial.gesture;
  if (partial.confidence !== undefined) statConfidence.textContent = `${partial.confidence}%`;
  if (partial.handedness !== undefined) statHandedness.textContent = partial.handedness;
}

function openStatusPanel() {
  const expanded = statusToggle.getAttribute("aria-expanded") === "true";
  statusToggle.setAttribute("aria-expanded", String(!expanded));
  statusBody.hidden = expanded;
}

statusToggle.addEventListener("click", openStatusPanel);
settingsBtn.addEventListener("click", () => {
  document.getElementById("statusPanel").scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
  if (statusToggle.getAttribute("aria-expanded") !== "true") openStatusPanel();
});

/* =========================================================
   Init
   ========================================================= */

resetRecognitionUI();
updateStatusPanel({ camera: "DISCONNECTED", tracking: "INACTIVE", hands: 0, recognition: "INACTIVE", fps: 0, gesture: "—", confidence: 0, handedness: "—" });

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  showError("This browser does not support camera access. Try the latest Chrome, Edge, or Safari.");
  cameraToggle.disabled = true;
}

/* PWA: register service worker */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch((err) => {
      console.warn("Service worker registration failed:", err);
    });
  });
}
