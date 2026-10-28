/**
 * PulseAI - Frontend Controller & Interactivity
 * Handles Theme Toggling, Web Speech API Voice Input, Quick-tag Symptom Chips,
 * Real-time AJAX Analysis, and Interactive Chart.js Visualizations.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initVoiceInput();
  initSymptomChips();
  initLiveAnalyzer();
  initDashboardCharts();
  initTableSearch();
});

/* ==========================================================================
   1. Theme Management (Light / Dark Mode)
   ========================================================================== */
function initTheme() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  const storedTheme = localStorage.getItem("pulseai_theme") || "light";
  document.documentElement.setAttribute("data-theme", storedTheme);
  updateThemeIcon(storedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("pulseai_theme", next);
      updateThemeIcon(next);

      // Re-render charts with updated theme colors if on dashboard
      if (typeof window.renderChartsWithTheme === "function") {
        window.renderChartsWithTheme(next);
      }
    });
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById("themeIcon");
  if (icon) {
    icon.textContent = theme === "dark" ? "☀️" : "🌙";
  }
}

/* ==========================================================================
   2. Voice Recognition (Web Speech API)
   ========================================================================== */
function initVoiceInput() {
  const voiceBtn = document.getElementById("voiceInputBtn");
  const textarea = document.getElementById("symptomsInput");
  if (!voiceBtn || !textarea) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceBtn.title = "Voice recognition not supported in this browser";
    voiceBtn.style.opacity = "0.5";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.lang = "en-US";

  let isListening = false;

  voiceBtn.addEventListener("click", () => {
    if (isListening) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (e) {
        console.error("Speech recognition error:", e);
      }
    }
  });

  recognition.onstart = () => {
    isListening = true;
    voiceBtn.classList.add("listening");
    voiceBtn.innerHTML = `<span>🎙️</span> Listening... Speak now`;
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    if (textarea.value.trim()) {
      textarea.value += ", " + transcript;
    } else {
      textarea.value = transcript;
    }
    textarea.dispatchEvent(new Event("input"));
  };

  recognition.onend = () => {
    isListening = false;
    voiceBtn.classList.remove("listening");
    voiceBtn.innerHTML = `<span>🎙️</span> Speak Symptoms`;
  };

  recognition.onerror = () => {
    isListening = false;
    voiceBtn.classList.remove("listening");
    voiceBtn.innerHTML = `<span>🎙️</span> Speak Symptoms`;
  };
}

/* ==========================================================================
   3. Quick-Tag Symptom Chips & Categories
   ========================================================================== */
function initSymptomChips() {
  const textarea = document.getElementById("symptomsInput");
  const chips = document.querySelectorAll(".symptom-chip");
  const catButtons = document.querySelectorAll(".cat-tab-btn");

  // Tab switching for categories
  catButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      catButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const targetCategory = btn.getAttribute("data-cat");

      document.querySelectorAll(".cat-cloud-group").forEach(group => {
        if (targetCategory === "all" || group.getAttribute("data-cat") === targetCategory) {
          group.style.display = "flex";
        } else {
          group.style.display = "none";
        }
      });
    });
  });

  // Clicking chip appends or removes from textarea
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      if (!textarea) return;
      const symptom = chip.getAttribute("data-symptom");
      let currentVal = textarea.value.trim();

      if (chip.classList.contains("selected")) {
        // Remove symptom
        chip.classList.remove("selected");
        const parts = currentVal.split(/,\s*/).filter(p => p.toLowerCase() !== symptom.toLowerCase());
        textarea.value = parts.join(", ");
      } else {
        // Add symptom
        chip.classList.add("selected");
        if (currentVal) {
          textarea.value = currentVal + ", " + symptom;
        } else {
          textarea.value = symptom;
        }
      }
      textarea.dispatchEvent(new Event("input"));
    });
  });

  // Quick preset sample buttons
  document.querySelectorAll(".quick-sample-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!textarea) return;
      textarea.value = btn.getAttribute("data-text");
      textarea.dispatchEvent(new Event("input"));
    });
  });
}

/* ==========================================================================
   4. Live Instant AJAX Symptom Analysis
   ========================================================================== */
function initLiveAnalyzer() {
  const form = document.getElementById("symptomForm");
  const liveCard = document.getElementById("liveResultCard");
  const textarea = document.getElementById("symptomsInput");
  if (!form || !liveCard || !textarea) return;

  let debounceTimer = null;

  textarea.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    const text = textarea.value.trim();
    if (text.length < 5) {
      liveCard.style.display = "none";
      return;
    }

    debounceTimer = setTimeout(() => {
      fetchLivePrediction();
    }, 600);
  });

  async function fetchLivePrediction() {
    const text = textarea.value.trim();
    if (!text) return;

    const severity = document.querySelector('input[name="severity"]:checked')?.value || "moderate";
    const duration = document.querySelector('input[name="duration"]')?.value || "2";

    try {
      const resp = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: text, severity, duration })
      });
      const data = await resp.json();
      if (resp.ok && data.prediction) {
        renderLiveResult(data);
      }
    } catch (err) {
      console.warn("Live analysis check skipped:", err);
    }
  }

  function renderLiveResult(data) {
    liveCard.style.display = "block";
    
    // Urgency styling
    const badge = document.getElementById("liveUrgencyBadge");
    if (badge) {
      badge.className = `badge-priority ${data.urgency}`;
      badge.textContent = `${data.urgency} Priority`;
    }

    // Prediction & confidence
    const predElem = document.getElementById("livePredictionText");
    if (predElem) predElem.textContent = data.prediction;

    const confElem = document.getElementById("liveConfidenceText");
    if (confElem) confElem.textContent = `${data.confidence.toFixed(1)}%`;

    const barElem = document.getElementById("liveConfidenceBar");
    if (barElem) barElem.style.width = `${data.confidence}%`;

    const specialistElem = document.getElementById("liveSpecialistText");
    if (specialistElem) specialistElem.textContent = data.specialist;

    const descElem = document.getElementById("liveDescText");
    if (descElem) descElem.textContent = data.description;
  }
}

/* ==========================================================================
   5. Interactive Clinical Dashboard Charts (Chart.js)
   ========================================================================== */
let conditionChartInstance = null;
let urgencyChartInstance = null;
let symptomChartInstance = null;

async function initDashboardCharts() {
  const condCanvas = document.getElementById("conditionChart");
  const urgCanvas = document.getElementById("urgencyChart");
  const symCanvas = document.getElementById("symptomChart");
  if (!condCanvas) return;

  try {
    const res = await fetch("/api/analytics");
    const stats = await res.json();
    renderCharts(stats);

    window.renderChartsWithTheme = (theme) => {
      renderCharts(stats, theme);
    };
  } catch (err) {
    console.error("Failed to load dashboard chart data:", err);
  }
}

function renderCharts(stats, theme = null) {
  if (typeof Chart === "undefined") return;

  const isDark = (theme || document.documentElement.getAttribute("data-theme")) === "dark";
  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.05)";

  // 1. Condition Breakdown Doughnut Chart
  const condCanvas = document.getElementById("conditionChart");
  if (condCanvas && stats.conditions) {
    if (conditionChartInstance) conditionChartInstance.destroy();

    const labels = Object.keys(stats.conditions);
    const dataVals = Object.values(stats.conditions);

    conditionChartInstance = new Chart(condCanvas, {
      type: "doughnut",
      data: {
        labels: labels.length ? labels : ["No data yet"],
        datasets: [{
          data: dataVals.length ? dataVals : [1],
          backgroundColor: [
            "#0284c7", "#0d9488", "#f59e0b", "#8b5cf6",
            "#ec4899", "#10b981", "#ef4444", "#3b82f6", "#6366f1"
          ],
          borderWidth: 2,
          borderColor: isDark ? "#131d33" : "#ffffff"
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { color: textColor, boxWidth: 12, font: { family: "'Plus Jakarta Sans', sans-serif" } }
          }
        },
        cutout: "65%"
      }
    });
  }

  // 2. Urgency Breakdown Bar Chart
  const urgCanvas = document.getElementById("urgencyChart");
  if (urgCanvas && stats.urgencies) {
    if (urgencyChartInstance) urgencyChartInstance.destroy();

    urgencyChartInstance = new Chart(urgCanvas, {
      type: "bar",
      data: {
        labels: ["Urgent Emergency", "Moderate Clinical", "Mild Home Care"],
        datasets: [{
          label: "Consultations",
          data: [
            stats.urgencies["Urgent"] || 0,
            stats.urgencies["Moderate"] || 0,
            stats.urgencies["Mild"] || 0
          ],
          backgroundColor: ["#ef4444", "#f59e0b", "#10b981"],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { color: textColor, stepSize: 1 },
            grid: { color: gridColor }
          },
          x: {
            ticks: { color: textColor },
            grid: { display: false }
          }
        }
      }
    });
  }

  // 3. Top Symptoms Encountered Horizontal Bar Chart
  const symCanvas = document.getElementById("symptomChart");
  if (symCanvas && stats.symptoms_freq) {
    if (symptomChartInstance) symptomChartInstance.destroy();

    const symLabels = Object.keys(stats.symptoms_freq);
    const symCounts = Object.values(stats.symptoms_freq);

    symptomChartInstance = new Chart(symCanvas, {
      type: "bar",
      indexAxis: "y",
      data: {
        labels: symLabels.length ? symLabels : ["None logged"],
        datasets: [{
          label: "Frequency",
          data: symCounts.length ? symCounts : [0],
          backgroundColor: "#38bdf8",
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            beginAtZero: true,
            ticks: { color: textColor, stepSize: 1 },
            grid: { color: gridColor }
          },
          y: {
            ticks: { color: textColor },
            grid: { display: false }
          }
        }
      }
    });
  }
}

/* ==========================================================================
   6. Table Live Search & Filter
   ========================================================================== */
function initTableSearch() {
  const searchInput = document.getElementById("tableSearchInput");
  const table = document.getElementById("historyTable");
  if (!searchInput || !table) return;

  searchInput.addEventListener("input", () => {
    const query = searchInput.value.toLowerCase();
    const rows = table.querySelectorAll("tbody tr");

    rows.forEach(row => {
      const text = row.textContent.toLowerCase();
      row.style.display = text.includes(query) ? "" : "none";
    });
  });
}
