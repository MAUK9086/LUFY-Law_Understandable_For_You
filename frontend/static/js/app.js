/**
 * LUFY app.js — state machine and API integration for the legal document assistant.
 *
 * States: IDLE → UPLOADING → UPLOADED → ANALYSING → READY
 */

"use strict";

// ── State ──────────────────────────────────────────────────────────────────────

const STATES = ["IDLE", "UPLOADING", "UPLOADED", "ANALYSING", "READY"];
let state = "IDLE";
let sessionId = null;

function setState(next) {
  state = next;
  _applyState();
}

// ── DOM refs ───────────────────────────────────────────────────────────────────

const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const demoBtn        = document.getElementById("demo-btn");
const analyseWrap    = document.getElementById("analyse-btn-wrap");
const analyseBtn     = document.getElementById("analyse-btn");
const docInfoCard    = document.getElementById("doc-info");
const docName        = document.getElementById("doc-name");
const docPages       = document.getElementById("doc-pages");
const docChars       = document.getElementById("doc-chars");
const docChunks      = document.getElementById("doc-chunks");
const personaSelect  = document.getElementById("persona-select");
const langSelect     = document.getElementById("lang-select");
const tabBtns        = document.querySelectorAll(".tab-btn");
const tabPanels      = document.querySelectorAll(".tab-panel");
const summaryContent = document.getElementById("summary-content");
const riskContent    = document.getElementById("risk-content");
const chatWindow     = document.getElementById("chat-window");
const chatInput      = document.getElementById("chat-input");
const chatSendBtn    = document.getElementById("chat-send-btn");

// ── State application ──────────────────────────────────────────────────────────

function _applyState() {
  const isUploading  = state === "UPLOADING";
  const isUploaded   = ["UPLOADED", "ANALYSING", "READY"].includes(state);
  const isAnalysing  = state === "ANALYSING";
  const isReady      = state === "READY";

  fileInput.disabled = isUploading;
  demoBtn.disabled   = isUploading || isAnalysing;

  analyseWrap.classList.toggle("visible", isUploaded);
  analyseBtn.disabled = isAnalysing || !isUploaded || isReady;
  analyseBtn.innerHTML = isAnalysing
    ? '<span class="spinner"></span>Analysing…'
    : "Analyse Document";

  chatInput.disabled   = !isReady;
  chatSendBtn.disabled = !isReady;
}

// ── Tabs ───────────────────────────────────────────────────────────────────────

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabBtns.forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

function _switchTab(id) {
  tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === id));
  tabPanels.forEach((p) => p.classList.toggle("active", p.id === id));
}

// ── Upload ─────────────────────────────────────────────────────────────────────

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadDocument(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadDocument(fileInput.files[0]);
});

async function uploadDocument(file) {
  setState("UPLOADING");
  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed." }));
      _showError(err.detail || "Upload failed.");
      setState("IDLE");
      return;
    }
    const data = await res.json();
    _onUploaded(data);
  } catch (e) {
    _showError("Network error during upload.");
    setState("IDLE");
  }
}

async function loadDemo() {
  setState("UPLOADING");
  try {
    const res = await fetch("/api/demo", { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Demo load failed." }));
      _showError(err.detail || "Demo load failed.");
      setState("IDLE");
      return;
    }
    const data = await res.json();
    _onUploaded(data);
  } catch (e) {
    _showError("Network error loading demo.");
    setState("IDLE");
  }
}

function _onUploaded(data) {
  sessionId = data.session_id;
  docInfoCard.classList.add("visible");
  docName.textContent   = data.filename;
  docPages.textContent  = `${data.page_count} page${data.page_count !== 1 ? "s" : ""}`;
  docChars.textContent  = `${(data.char_count / 1000).toFixed(1)}k chars`;
  docChunks.textContent = `${data.chunk_count} chunks`;
  summaryContent.innerHTML = _emptyState("📄", "Click <strong>Analyse Document</strong> to generate a summary.");
  riskContent.innerHTML    = _emptyState("🔍", "Click <strong>Analyse Document</strong> to identify risk flags.");
  chatWindow.innerHTML     = "";
  setState("UPLOADED");
}

// ── Analysis ───────────────────────────────────────────────────────────────────

analyseBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (!sessionId) return;
  setState("ANALYSING");

  const persona  = personaSelect.value;
  const language = langSelect.value;

  const summariseReq = fetch("/api/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, persona, language }),
  });

  const riskReq = fetch("/api/risk-analysis", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, persona, language }),
  });

  const [sumRes, riskRes] = await Promise.allSettled([summariseReq, riskReq]);

  if (sumRes.status === "fulfilled" && sumRes.value.ok) {
    const data = await sumRes.value.json();
    summaryContent.innerHTML = `<div class="summary-card">${_escapeHtml(data.summary)}</div>`;
  } else {
    summaryContent.innerHTML = _emptyState("⚠️", "Summarisation failed. Check your API key.");
  }

  if (riskRes.status === "fulfilled" && riskRes.value.ok) {
    const data = await riskRes.value.json();
    _renderRiskDashboard(data);
  } else {
    riskContent.innerHTML = _emptyState("⚠️", "Risk analysis failed. Check your API key.");
  }

  setState("READY");
  _switchTab("tab-summary");
}

// ── Risk dashboard ─────────────────────────────────────────────────────────────

function _renderRiskDashboard(data) {
  const sections = [
    { key: "red_flags",    label: "Red Flags",    cls: "red" },
    { key: "yellow_flags", label: "Yellow Flags", cls: "yellow" },
    { key: "green_flags",  label: "Green Flags",  cls: "green" },
  ];

  riskContent.innerHTML = sections.map(({ key, label, cls }) => {
    const flags = data[key] || [];
    const cards = flags.length
      ? flags.map((f) => `
          <div class="flag-card ${cls}">
            <div class="flag-clause">${_escapeHtml(f.clause)}</div>
            <div class="flag-explanation">${_escapeHtml(f.explanation)}</div>
            <div class="flag-advice">${_escapeHtml(f.advice)}</div>
          </div>`).join("")
      : `<p style="color:var(--muted);font-size:.85rem">None identified.</p>`;

    return `
      <div class="risk-section">
        <div class="risk-section-header">
          <h3>${_escapeHtml(label)}</h3>
          <span class="risk-badge ${cls}">${flags.length}</span>
        </div>
        ${cards}
      </div>`;
  }).join("");
}

// ── Chat / RAG ─────────────────────────────────────────────────────────────────

chatSendBtn.addEventListener("click", sendQuery);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

async function sendQuery() {
  const query = chatInput.value.trim();
  if (!query || !sessionId || state !== "READY") return;

  chatInput.value = "";
  _appendUserBubble(query);

  const loadingEl = _appendLoadingBubble();

  const persona  = personaSelect.value;
  const language = langSelect.value;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        persona,
        language,
      }),
    });

    loadingEl.remove();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Query failed." }));
      _appendBotResponse(err.detail || "Query failed.", []);
      return;
    }

    const data = await res.json();
    _appendBotResponse(data.answer, data.sources || []);
  } catch {
    loadingEl.remove();
    _appendBotResponse("Network error. Please try again.", []);
  }
}

function _appendUserBubble(text) {
  const el = document.createElement("div");
  el.className = "chat-bubble user";
  el.textContent = text;
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function _appendLoadingBubble() {
  const el = document.createElement("div");
  el.className = "loading-bubble";
  el.innerHTML = '<span class="spinner spinner-dark"></span>Thinking…';
  chatWindow.appendChild(el);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return el;
}

function _appendBotResponse(answer, sources) {
  const wrap = document.createElement("div");
  wrap.style.alignSelf = "flex-start";
  wrap.style.maxWidth = "75%";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble bot";
  bubble.innerHTML = _escapeHtml(answer).replace(/\n/g, "<br>");
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const toggle = document.createElement("button");
    toggle.className = "citation-toggle";
    toggle.textContent = `Show ${sources.length} source excerpt${sources.length !== 1 ? "s" : ""}`;

    const citList = document.createElement("div");
    citList.className = "citations";
    sources.forEach((src, i) => {
      const block = document.createElement("div");
      block.className = "citation-block";
      block.innerHTML = `<strong>Source ${i + 1}</strong><br>${_escapeHtml(src)}`;
      citList.appendChild(block);
    });

    toggle.addEventListener("click", () => {
      citList.classList.toggle("open");
      const open = citList.classList.contains("open");
      toggle.textContent = open
        ? `Hide ${sources.length} source excerpt${sources.length !== 1 ? "s" : ""}`
        : `Show ${sources.length} source excerpt${sources.length !== 1 ? "s" : ""}`;
    });

    wrap.appendChild(toggle);
    wrap.appendChild(citList);
  }

  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function _emptyState(icon, message) {
  return `
    <div class="empty-state">
      <span class="empty-icon">${icon}</span>
      <p>${message}</p>
    </div>`;
}

function _showError(message) {
  alert(message);
}

// ── Wire up demo button ────────────────────────────────────────────────────────

demoBtn.addEventListener("click", loadDemo);

// ── Initial state ──────────────────────────────────────────────────────────────

_applyState();
