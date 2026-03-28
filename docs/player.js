const state = {
  audioContext: null,
  example: null,
  isPlaying: false,
  startedAt: 0,
  pausedAt: 0,
  activeNodes: [],
  animationFrame: null,
};

const elements = {
  play: document.querySelector("[data-action='play']"),
  pause: document.querySelector("[data-action='pause']"),
  stop: document.querySelector("[data-action='stop']"),
  progressFill: document.querySelector("[data-progress-fill]"),
  progressLabel: document.querySelector("[data-progress-label]"),
  meta: document.querySelector("[data-example-meta]"),
  canvas: document.querySelector("[data-roll]"),
  summary: document.querySelector("[data-example-summary]"),
};

function midiToFrequency(noteNumber) {
  return 440 * Math.pow(2, (noteNumber - 69) / 12);
}

function formatSeconds(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function createAudioContext() {
  if (!state.audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioContextClass();
  }
  return state.audioContext;
}

function currentOffset() {
  if (!state.isPlaying) {
    return state.pausedAt;
  }
  const offset = state.audioContext.currentTime - state.startedAt;
  return Math.min(offset, state.example.duration_sec);
}

function clearActiveNodes() {
  for (const entry of state.activeNodes) {
    try {
      entry.oscA.stop();
    } catch (_) {}
    try {
      entry.oscB.stop();
    } catch (_) {}
    entry.oscA.disconnect();
    entry.oscB.disconnect();
    entry.gain.disconnect();
  }
  state.activeNodes = [];
}

function scheduleExample(offsetSeconds) {
  clearActiveNodes();
  const audioContext = createAudioContext();
  const now = audioContext.currentTime;

  for (const note of state.example.notes) {
    if (note.end <= offsetSeconds) {
      continue;
    }

    const noteStart = Math.max(note.start, offsetSeconds);
    const noteDuration = note.end - noteStart;
    if (noteDuration <= 0) {
      continue;
    }

    const when = now + (noteStart - offsetSeconds);
    const frequency = midiToFrequency(note.pitch);
    const velocity = Math.max(0.08, note.velocity / 127);

    const gain = audioContext.createGain();
    gain.gain.setValueAtTime(0.0001, when);
    gain.gain.linearRampToValueAtTime(0.09 * velocity, when + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, when + noteDuration + 0.14);
    gain.connect(audioContext.destination);

    const oscA = audioContext.createOscillator();
    oscA.type = "triangle";
    oscA.frequency.setValueAtTime(frequency, when);
    oscA.connect(gain);
    oscA.start(when);
    oscA.stop(when + noteDuration + 0.16);

    const oscB = audioContext.createOscillator();
    oscB.type = "sine";
    oscB.frequency.setValueAtTime(frequency * 2, when);
    oscB.connect(gain);
    oscB.start(when);
    oscB.stop(when + noteDuration + 0.12);

    state.activeNodes.push({ oscA, oscB, gain });
  }
}

function drawPianoRoll() {
  if (!state.example) {
    return;
  }
  const canvas = elements.canvas;
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const minPitch = state.example.pitch_range.min;
  const maxPitch = state.example.pitch_range.max;
  const pitchSpan = Math.max(1, maxPitch - minPitch + 1);
  const duration = Math.max(0.001, state.example.duration_sec);
  const playheadSeconds = currentOffset();

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f5efe3";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(31, 36, 33, 0.08)";
  context.lineWidth = 1;
  for (let beat = 0; beat <= Math.ceil(duration); beat += 1) {
    const x = (beat / duration) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }

  for (const note of state.example.notes) {
    const x = (note.start / duration) * width;
    const w = Math.max(1.5, ((note.end - note.start) / duration) * width);
    const pitchY = (maxPitch - note.pitch) / pitchSpan;
    const y = pitchY * (height - 10) + 5;
    context.fillStyle = `rgba(24, 57, 48, ${0.28 + (note.velocity / 127) * 0.42})`;
    context.fillRect(x, y, w, Math.max(3, height / pitchSpan));
  }

  const playheadX = (playheadSeconds / duration) * width;
  context.strokeStyle = "#b65e36";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(playheadX, 0);
  context.lineTo(playheadX, height);
  context.stroke();
}

function updateProgress() {
  if (!state.example) {
    return;
  }
  const offset = currentOffset();
  const progress = Math.min(1, offset / state.example.duration_sec);
  elements.progressFill.style.width = `${progress * 100}%`;
  elements.progressLabel.textContent = `${formatSeconds(offset)} / ${formatSeconds(state.example.duration_sec)}`;
  drawPianoRoll();

  if (state.isPlaying) {
    if (offset >= state.example.duration_sec) {
      stopPlayback();
      return;
    }
    state.animationFrame = requestAnimationFrame(updateProgress);
  }
}

async function startPlayback() {
  if (!state.example) {
    return;
  }

  const audioContext = createAudioContext();
  await audioContext.resume();

  if (state.pausedAt >= state.example.duration_sec) {
    state.pausedAt = 0;
  }

  scheduleExample(state.pausedAt);
  state.startedAt = audioContext.currentTime - state.pausedAt;
  state.isPlaying = true;
  cancelAnimationFrame(state.animationFrame);
  updateProgress();
}

function pausePlayback() {
  if (!state.isPlaying) {
    return;
  }
  state.pausedAt = currentOffset();
  state.isPlaying = false;
  clearActiveNodes();
  cancelAnimationFrame(state.animationFrame);
  updateProgress();
}

function stopPlayback() {
  if (!state.example) {
    return;
  }
  state.isPlaying = false;
  state.pausedAt = 0;
  clearActiveNodes();
  cancelAnimationFrame(state.animationFrame);
  updateProgress();
}

function renderMeta() {
  const meta = state.example.meta;
  const items = [
    ["Checkpoint", meta.checkpoint.split("/").slice(-1)[0]],
    ["Preset", meta.preset],
    ["Heuristic score", meta.heuristic_score.toFixed(3)],
    ["Prompt", `validation:${meta.prompt_index} (${meta.prompt_position})`],
    ["Generated tokens", String(meta.generated_tokens)],
    ["Notes", String(meta.note_count)],
  ];

  elements.meta.innerHTML = items
    .map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  elements.summary.innerHTML = `
    <p>${state.example.description}</p>
    <p>
      Duration ${formatSeconds(state.example.duration_sec)} · Tempo ${state.example.tempo_bpm.toFixed(1)} BPM ·
      Pitch span ${state.example.meta.pitch_span} semitones
    </p>
  `;
}

async function loadExample() {
  const response = await fetch("assets/generated-example.json");
  state.example = await response.json();
  renderMeta();
  updateProgress();
}

elements.play.addEventListener("click", () => {
  startPlayback();
});
elements.pause.addEventListener("click", () => {
  pausePlayback();
});
elements.stop.addEventListener("click", () => {
  stopPlayback();
});

loadExample().catch((error) => {
  elements.summary.innerHTML = `<p>Could not load example: ${error.message}</p>`;
});
