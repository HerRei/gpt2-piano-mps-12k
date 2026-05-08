const sections = document.querySelectorAll("main section[id]");
const navLinks = document.querySelectorAll(".nav-links a[href^='#']");
const intensityRange = document.querySelector("#intensity-range");
const intensityValue = document.querySelector("#intensity-value");
const densityPreview = document.querySelector(".density-preview");
const intensityCard = document.querySelector(".intensity-card");
const intensityMode = document.querySelector("#intensity-mode");
const intensityNoteCount = document.querySelector("#intensity-note-count");
const intensityVelocity = document.querySelector("#intensity-velocity");
const intensityNotes = document.querySelectorAll(".intensity-note");

const setActiveLink = () => {
  let currentId = "";

  sections.forEach((section) => {
    const sectionTop = section.offsetTop - 120;
    if (window.scrollY >= sectionTop) {
      currentId = section.id;
    }
  });

  navLinks.forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${currentId}`);
  });
};

const updateIntensity = () => {
  if (!intensityRange || !intensityValue || !intensityCard) {
    return;
  }

  const value = Number(intensityRange.value);
  const velocity = Math.round(32 + value * 72);
  let activeCount = 0;

  intensityValue.textContent = value.toFixed(2);
  intensityCard.style.setProperty("--preview-intensity", value.toString());

  if (densityPreview) {
    densityPreview.style.setProperty("--preview-intensity", value.toString());
  }

  intensityNotes.forEach((note) => {
    const threshold = Number(note.dataset.min || 0);
    const isActive = value >= threshold;
    note.classList.toggle("is-active", isActive);

    if (isActive) {
      activeCount += 1;
    }
  });

  if (intensityMode) {
    if (value < 0.34) {
      intensityMode.textContent = "soft sparse phrase";
    } else if (value < 0.67) {
      intensityMode.textContent = "balanced phrase";
    } else {
      intensityMode.textContent = "dense energetic phrase";
    }
  }

  if (intensityNoteCount) {
    intensityNoteCount.textContent = `${activeCount} preview notes active`;
  }

  if (intensityVelocity) {
    intensityVelocity.textContent = `velocity target ${velocity} / 127`;
  }
};

window.addEventListener("scroll", setActiveLink, { passive: true });
window.addEventListener("load", setActiveLink);

if (intensityRange) {
  intensityRange.addEventListener("input", updateIntensity);
  updateIntensity();
}
