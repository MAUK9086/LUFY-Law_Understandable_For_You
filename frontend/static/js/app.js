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
  demoBtn.innerHTML  = isUploading
    ? '<span class="spinner spinner-dark"></span>Loading…'
    : "Try with sample document";

  dropZone.classList.toggle("uploading", isUploading);

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
    summaryContent.innerHTML = `<div class="summary-card">${_renderMarkdown(data.summary)}</div>`;
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
  const labels = data.section_labels || {
    red: "Red Flags", yellow: "Yellow Flags", green: "Green Flags",
  };

  const sections = [
    { key: "red_flags",    label: labels.red,    cls: "red" },
    { key: "yellow_flags", label: labels.yellow, cls: "yellow" },
    { key: "green_flags",  label: labels.green,  cls: "green" },
  ];

  riskContent.innerHTML = sections.map(({ key, label, cls }) => {
    const flags = data[key] || [];
    const cards = flags.length
      ? flags.map((f) => `
          <details class="flag-card ${cls}">
            <summary class="flag-header">
              <span class="flag-clause">${_escapeHtml(f.clause)}</span>
              <span class="flag-chevron">▸</span>
            </summary>
            <div class="flag-body">
              <p class="flag-explanation">${_escapeHtml(f.explanation)}</p>
              <div class="flag-advice">${_escapeHtml(f.advice)}</div>
            </div>
          </details>`).join("")
      : `<p class="no-flags">None identified.</p>`;

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
    _appendBotResponse(data.answer, data.sources || [], query);
  } catch {
    loadingEl.remove();
    _appendBotResponse("Network error. Please try again.", [], "");
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

function _appendBotResponse(answer, sources, query) {
  const wrap = document.createElement("div");
  wrap.className = "bot-response-wrap";

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble bot";
  bubble.innerHTML = _escapeHtml(answer).replace(/\n/g, "<br>");
  wrap.appendChild(bubble);

  if (sources && sources.length) {
    const citContainer = document.createElement("div");
    citContainer.className = "citations-container";

    const toggle = document.createElement("button");
    toggle.className = "citation-toggle";
    toggle.innerHTML = `<span class="cit-icon">&#128206;</span> ${sources.length} passage${sources.length !== 1 ? "s" : ""} retrieved <span class="cit-arrow open">&#9660;</span>`;

    const citList = document.createElement("div");
    citList.className = "citations open";

    sources.forEach((src, i) => {
      const excerpt = _extractRelevantSentences(src, query || "", 3);
      const isTrimmed = excerpt.length < src.trim().length - 10;

      const block = document.createElement("div");
      block.className = "citation-block";

      const label = document.createElement("span");
      label.className = "cit-label";
      label.textContent = `Passage ${i + 1}`;
      block.appendChild(label);

      const text = document.createElement("span");
      text.className = "cit-text";
      text.textContent = excerpt;
      block.appendChild(text);

      if (isTrimmed) {
        const expand = document.createElement("button");
        expand.className = "cit-expand";
        expand.textContent = "Show full passage";
        expand.addEventListener("click", () => {
          text.textContent = src.trim();
          expand.remove();
        });
        block.appendChild(expand);
      }

      citList.appendChild(block);
    });

    toggle.addEventListener("click", () => {
      const isOpen = citList.classList.toggle("open");
      const arrow = toggle.querySelector(".cit-arrow");
      arrow.innerHTML = isOpen ? "&#9660;" : "&#9654;";
    });

    citContainer.appendChild(toggle);
    citContainer.appendChild(citList);
    wrap.appendChild(citContainer);
  }

  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/**
 * Extract the most query-relevant sentences from a retrieved chunk.
 *
 * Splits the chunk on sentence boundaries, scores each sentence by how many
 * non-trivial query words it contains, and returns the top sentences in their
 * original order joined by " … ". Falls back to the full text when the chunk
 * is short enough that trimming would not help.
 *
 * @param {string} chunkText - Full retrieved chunk text.
 * @param {string} query     - The user's question.
 * @param {number} maxSents  - Maximum number of sentences to keep (default 3).
 * @returns {string} Relevant excerpt, or the original text if short.
 */
function _extractRelevantSentences(chunkText, query, maxSents = 3) {
  const sentences = chunkText
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);

  if (sentences.length <= maxSents) return chunkText.trim();

  const stopWords = new Set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "was", "are", "were", "be", "been", "being",
    "it", "its", "that", "this", "these", "those", "by", "from", "as",
  ]);

  const queryWords = query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !stopWords.has(w));

  const scored = sentences.map((s, i) => {
    const lower = s.toLowerCase();
    const score = queryWords.reduce((acc, w) => acc + (lower.includes(w) ? 1 : 0), 0);
    return { i, s, score };
  });

  const topIndices = scored
    .slice()
    .sort((a, b) => b.score - a.score || a.i - b.i)
    .slice(0, maxSents)
    .map((x) => x.i)
    .sort((a, b) => a - b);

  return topIndices.map((idx) => sentences[idx]).join(" … ");
}

/**
 * Convert the structured summary text (with **bold** markers and • bullets)
 * into styled HTML. Handles the five-section format returned by the LLM.
 */
function _renderMarkdown(raw) {
  const lines = raw.split("\n");
  let html = "";
  let inList = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) { html += "</ul>"; inList = false; }
      continue;
    }

    // Section header: **Text:** at the start of a line
    const headerMatch = trimmed.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
    if (headerMatch && !trimmed.startsWith("•")) {
      if (inList) { html += "</ul>"; inList = false; }
      const title = _escapeHtml(headerMatch[1]);
      const rest  = headerMatch[2] ? " " + _escapeHtml(headerMatch[2]) : "";
      html += `<div class="sum-section"><span class="sum-label">${title}</span>${rest}</div>`;
      continue;
    }

    // Bullet point
    if (trimmed.startsWith("•") || trimmed.startsWith("-")) {
      if (!inList) { html += "<ul class='sum-list'>"; inList = true; }
      const content = _escapeHtml(trimmed.replace(/^[•\-]\s*/, ""));
      html += `<li>${content}</li>`;
      continue;
    }

    // Plain text continuation
    if (inList) { html += "</ul>"; inList = false; }
    html += `<p>${_escapeHtml(trimmed)}</p>`;
  }

  if (inList) html += "</ul>";
  return html;
}

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
