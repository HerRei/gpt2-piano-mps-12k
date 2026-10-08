import "./style.css";
import waveformPeaks from "./waveform.json";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);
const media = matchMedia("(prefers-reduced-motion: reduce)");
let paused = media.matches;
const motionButton = document.querySelector("#motion-toggle");
function setMotion(value) {
  paused = value;
  document.body.classList.toggle("motion-paused", paused);
  motionButton.setAttribute("aria-pressed", String(paused));
  motionButton.innerHTML = paused
    ? 'Resume motion <span aria-hidden="true">▷</span>'
    : 'Pause motion <span aria-hidden="true">Ⅱ</span>';
  window.dispatchEvent(new CustomEvent("piano-motion", { detail: paused }));
}
motionButton.addEventListener("click", () => setMotion(!paused));
media.addEventListener("change", (event) => setMotion(event.matches));
setMotion(paused);

const animations = gsap.matchMedia();
animations.add("(prefers-reduced-motion: no-preference)", () => {
  gsap.from(".hero h1 > *", {
    y: 65,
    opacity: 0,
    stagger: 0.1,
    duration: 1.1,
    ease: "power3.out",
    clearProps: "all",
  });
  gsap.from(".hero-description, .hero-copy .button", {
    y: 24,
    opacity: 0,
    delay: 0.25,
    stagger: 0.12,
    duration: 0.9,
    clearProps: "all",
  });
  gsap.utils.toArray(".reveal").forEach((element) => {
    gsap.from(element, {
      y: 38,
      opacity: 0.15,
      duration: 0.85,
      ease: "power2.out",
      scrollTrigger: { trigger: element, start: "top 94%", once: true },
      clearProps: "all",
    });
  });
  gsap.utils
    .toArray(".loss-track i, .budget-bar i, .latency-scale i")
    .forEach((element) => {
      gsap.from(element, {
        scaleX: 0,
        transformOrigin: "left",
        duration: 1.2,
        ease: "power3.out",
        scrollTrigger: { trigger: element, start: "top 90%", once: true },
      });
    });
});
const progress = document.querySelector(".reading-progress");
const updateProgress = () => {
  const distance = document.documentElement.scrollHeight - innerHeight;
  progress.style.transform = `scaleX(${distance > 0 ? scrollY / distance : 0})`;
};
window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("resize", updateProgress, { passive: true });
document.querySelectorAll("details").forEach((details) =>
  details.addEventListener("toggle", () => {
    ScrollTrigger.refresh();
    updateProgress();
  }),
);

const navigation = document.querySelectorAll(".site-header nav a");
const sectionObserver = new IntersectionObserver(
  (entries) =>
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navigation.forEach((link) => {
        if (link.hash === `#${entry.target.id}`)
          link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }),
  { rootMargin: "-15% 0px -55% 0px" },
);
document
  .querySelectorAll("main section[id]")
  .forEach((section) => sectionObserver.observe(section));

// The audio is an existing selected v1 render, never generated in the browser.
const audio = document.querySelector("#piano-audio");
const play = document.querySelector("#audio-play");
const seek = document.querySelector("#audio-seek");
const status = document.querySelector("#audio-status");
const waveform = document.querySelector(".waveform");
const bars = Array.from({ length: 66 }, (_, i) => {
  const bar = document.createElement("i");
  bar.style.height = `${8 + waveformPeaks[i] * 84}%`;
  waveform.append(bar);
  return bar;
});
function timeLabel(time) {
  return `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, "0")}`;
}
function audioProgress() {
  const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  const percent = duration ? audio.currentTime / duration : 0;
  seek.value = percent * 100;
  seek.setAttribute(
    "aria-valuetext",
    `${timeLabel(audio.currentTime)} of ${timeLabel(duration)}`,
  );
  document.querySelector("#audio-time").textContent = timeLabel(
    audio.currentTime,
  );
  document.querySelector("#audio-duration").textContent = duration
    ? timeLabel(duration)
    : "—";
  seek.disabled = !duration;
  bars.forEach((bar, i) =>
    bar.classList.toggle("played", i / bars.length < percent),
  );
}
play.disabled = false;
document.querySelector("#audio-player").classList.add("custom-audio");
play.addEventListener("click", async () => {
  if (!audio.paused) {
    audio.pause();
    return;
  }
  try {
    await audio.play();
  } catch {
    status.textContent = "Playback unavailable. Try the MIDI link.";
  }
});
audio.addEventListener("play", () => {
  play.classList.add("playing");
  play.setAttribute("aria-label", "Pause GPT-2 piano sample");
  status.textContent = "PLAYING / V1 SAMPLE";
});
audio.addEventListener("pause", () => {
  play.classList.remove("playing");
  play.setAttribute("aria-label", "Play GPT-2 piano sample");
  status.textContent = audio.ended ? "END OF SAMPLE" : "PAUSED / V1 SAMPLE";
});
audio.addEventListener("error", () => {
  status.textContent = "Audio unavailable. Download the original MIDI.";
  play.disabled = true;
});
audio.addEventListener("timeupdate", audioProgress);
audio.addEventListener("loadedmetadata", audioProgress);
seek.addEventListener("input", () => {
  if (Number.isFinite(audio.duration))
    audio.currentTime = (Number(seek.value) / 100) * audio.duration;
  audioProgress();
});
audioProgress();

// A deliberately illustrative phrase sketch. It is independent of both models.
const range = document.querySelector("#intensity-range");
const notes = document.querySelector(".control-notes");
const previewNotes = Array.from({ length: 30 }, (_, i) => {
  const note = document.createElement("span");
  note.className = "control-note";
  note.style.left = `${(i * 17.5) % 82}%`;
  note.style.top = `${9 + ((i * 31) % 80)}%`;
  note.style.width = `${8 + (i % 4) * 3}%`;
  note.dataset.threshold = String(i < 4 ? 0 : (i - 3) / 28);
  notes.append(note);
  return note;
});
const keyboard = document.querySelector(".mini-keyboard");
for (let i = 0; i < 21; i++) keyboard.append(document.createElement("i"));
function updateIntensity() {
  const value = Number(range.value);
  let count = 0;
  previewNotes.forEach((note) => {
    const active = value >= Number(note.dataset.threshold);
    note.style.opacity = active ? String(0.45 + value * 0.55) : ".07";
    note.style.transform = `scaleX(${active ? 0.85 + value * 0.15 : 0.55})`;
    if (active) count++;
  });
  document.querySelector("#intensity-value").textContent = value.toFixed(2);
  document.querySelector("#intensity-mode").textContent =
    value < 0.34 ? "SOFT / SPARSE" : value < 0.67 ? "BALANCED" : "LOUD / DENSE";
  document.querySelector("#intensity-note-count").textContent =
    `${count} preview notes`;
  document.querySelector("#intensity-velocity").textContent =
    `Velocity ${Math.round(32 + value * 72)} / 127`;
  range.setAttribute(
    "aria-valuetext",
    `${value.toFixed(2)}, ${value < 0.34 ? "soft and sparse" : value < 0.67 ? "balanced" : "loud and dense"}`,
  );
}
range.addEventListener("input", updateIntensity);
updateIntensity();

// Preserve the visual fallback if WebGL is unavailable.
import("./instrument.js")
  .then(({ createInstrument }) =>
    createInstrument(document.querySelector("#piano-scene"), () => paused),
  )
  .catch(() => {});
