// DOM Elements
const statusEl = document.querySelector("#app-status");
const errorBanner = document.querySelector("#error-banner");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// Config Elements
const configForm = document.querySelector("#config-form");
const useLlmEl = document.querySelector("#cfg-use-llm");
const llmFallbackEl = document.querySelector("#cfg-llm-fallback");
const llmProviderEl = document.querySelector("#cfg-llm-provider");
const retrievalBackendEl = document.querySelector("#cfg-retrieval-backend");
const rerankBackendEl = document.querySelector("#cfg-rerank-backend");
const cohereKeyGroup = document.querySelector("#cohere-key-group");
const cohereKeyEl = document.querySelector("#cfg-cohere-key");
const chatModelEl = document.querySelector("#cfg-chat-model");
const ollamaModelEl = document.querySelector("#cfg-ollama-model");
const maxAttemptsEl = document.querySelector("#cfg-max-attempts");
const minScoreEl = document.querySelector("#cfg-min-score");

// Playground Elements
const ingestForm = document.querySelector("#ingest-form");
const askForm = document.querySelector("#ask-form");
const inputPathEl = document.querySelector("#input-path");
const questionEl = document.querySelector("#question");
const responseVerdict = document.querySelector("#response-verdict");
const answerText = document.querySelector("#answer-text");

// Metrics Elements
const metricTime = document.querySelector("#metric-time");
const metricSteps = document.querySelector("#metric-steps");
const metricAttempts = document.querySelector("#metric-attempts");
const metricRetrievalRetries = document.querySelector("#metric-retrieval-retries");
const metricGenerationRetries = document.querySelector("#metric-generation-retries");

// Critic & Chunks Elements
const criticDetails = document.querySelector("#critic-details");
const retrievedSources = document.querySelector("#retrieved-sources");
const timelineTrace = document.querySelector("#timeline-trace");

// Graph Elements
const graphNodes = document.querySelectorAll(".graph-node");
const loops = {
  retrieval: document.querySelector("#loop-retrieval"),
  generation: document.querySelector("#loop-generation"),
};

// Evaluation Elements
const runEvalBtn = document.querySelector("#run-eval-btn");
const ragasWarning = document.querySelector("#ragas-disabled-warning");
const evalApprovalRate = document.querySelector("#eval-approval-rate");
const evalCitationRate = document.querySelector("#eval-citation-rate");
const evalFallbackRate = document.querySelector("#eval-fallback-rate");
const evalAvgLoops = document.querySelector("#eval-avg-loops");
const evalTableBody = document.querySelector("#eval-table-body");

// Page Init
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  fetchConfig();
  setupConfigListeners();
});

// Tab Navigation
function setupTabs() {
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));
      
      btn.classList.add("active");
      document.querySelector(`#tab-${targetTab}`).classList.add("active");
    });
  });
}

// Config Listeners
function setupConfigListeners() {
  rerankBackendEl.addEventListener("change", () => {
    toggleCohereInput(rerankBackendEl.value);
  });

  configForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    setBusy("Updating configuration");
    
    const payload = {
      use_llm: useLlmEl.checked,
      llm_fallback_enabled: llmFallbackEl.checked,
      llm_provider: llmProviderEl.value,
      retrieval_backend: retrievalBackendEl.value,
      rerank_backend: rerankBackendEl.value,
      chat_model: chatModelEl.value,
      ollama_model: ollamaModelEl.value,
      max_attempts: parseInt(maxAttemptsEl.value) || 3,
      min_context_score: parseFloat(minScoreEl.value) || 0.05,
    };

    if (cohereKeyEl.value) {
      payload.cohere_api_key = cohereKeyEl.value;
    }

    try {
      await postJson("/config", payload);
      setStatus("Configuration updated");
    } catch (err) {
      showError(err);
    } finally {
      setButtonsDisabled(false);
    }
  });
}

function toggleCohereInput(backend) {
  if (backend === "cohere") {
    cohereKeyGroup.classList.remove("hidden");
  } else {
    cohereKeyGroup.classList.add("hidden");
  }
}

// Ingestion Form
ingestForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  setBusy("Ingesting source documents");

  try {
    const data = await postJson("/ingest", {
      input_path: inputPathEl.value,
    });
    setStatus(`Ingested ${data.chunks_indexed} chunks`);
  } catch (err) {
    showError(err);
  } finally {
    setButtonsDisabled(false);
  }
});

// Query Submit
askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  setBusy("Executing self-healing pipeline");
  clearPlaygroundDisplay();

  try {
    const data = await postJson("/ask", {
      question: questionEl.value,
    });
    renderAskResponse(data);
    setStatus(data.critique?.approved ? "Pipeline complete: Approved" : "Pipeline complete: Fallback");
  } catch (err) {
    showError(err);
  } finally {
    setButtonsDisabled(false);
  }
});

// Evaluation trigger
runEvalBtn.addEventListener("click", async () => {
  setBusy("Running evaluations (this may take a minute)");
  
  try {
    const data = await postJson("/evaluate", {});
    renderEvaluationResults(data);
    setStatus("Evaluations completed");
  } catch (err) {
    showError(err);
  } finally {
    setButtonsDisabled(false);
  }
});

// Helpers
async function fetchConfig() {
  try {
    const res = await fetch("/config");
    if (res.ok) {
      const cfg = await res.json();
      useLlmEl.checked = cfg.use_llm;
      llmFallbackEl.checked = cfg.llm_fallback_enabled;
      llmProviderEl.value = cfg.llm_provider;
      retrievalBackendEl.value = cfg.retrieval_backend;
      rerankBackendEl.value = cfg.rerank_backend;
      chatModelEl.value = cfg.chat_model || "";
      ollamaModelEl.value = cfg.ollama_model || "";
      maxAttemptsEl.value = cfg.max_attempts || 3;
      minScoreEl.value = cfg.min_context_score || 0.05;
      
      toggleCohereInput(cfg.rerank_backend);
    }
  } catch (err) {
    console.error("Failed to load settings configuration:", err);
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (response.ok) {
    return response.json();
  }

  let details = `Error: HTTP ${response.status}`;
  try {
    const errObj = await response.json();
    details = errObj.detail || details;
  } catch {}
  throw new Error(details);
}

function clearPlaygroundDisplay() {
  hideError();
  responseVerdict.className = "badge";
  responseVerdict.textContent = "Processing...";
  answerText.textContent = "Analyzing pipeline nodes...";
  answerText.className = "text-container empty";
  
  // Reset flowchart nodes and loops
  graphNodes.forEach((node) => node.classList.remove("active", "completed", "rejected"));
  Object.values(loops).forEach((arrow) => arrow.classList.add("hidden"));
}

function setBusy(message) {
  hideError();
  setStatus(message);
  statusEl.className = "status-badge busy";
  setButtonsDisabled(true);
}

function setStatus(message) {
  statusEl.textContent = message;
  statusEl.className = "status-badge";
}

function setButtonsDisabled(disabled) {
  document.querySelectorAll("button").forEach((btn) => btn.disabled = disabled);
}

function showError(err) {
  errorBanner.textContent = err.message || "An unexpected error occurred.";
  errorBanner.classList.remove("hidden");
  setStatus("Failed");
}

function hideError() {
  errorBanner.textContent = "";
  errorBanner.classList.add("hidden");
}

// Render Ask Output
function renderAskResponse(data) {
  const approved = Boolean(data.critique?.approved);
  
  // Verdict Badge
  responseVerdict.textContent = approved ? "Approved" : (data.needs_clarification ? "Clarification Required" : "Fallback Limit");
  responseVerdict.className = `badge ${approved ? "approved" : "rejected"}`;
  
  // Output Text
  answerText.textContent = data.answer;
  answerText.classList.remove("empty");

  // Metrics
  metricTime.textContent = `${Math.round(data.metrics.total_elapsed_ms)} ms`;
  metricSteps.textContent = data.metrics.total_steps;
  metricAttempts.textContent = data.attempts;
  metricRetrievalRetries.textContent = data.metrics.retrieval_attempts;
  metricGenerationRetries.textContent = data.metrics.generation_attempts;

  // Critic Details
  if (data.critique) {
    criticDetails.innerHTML = `
      <div class="critic-feedback-val">
        <p><strong>Approved:</strong> ${data.critique.approved ? "✅ Yes" : "❌ No"}</p>
        <p><strong>Reason:</strong> ${escapeHtml(data.critique.reason)}</p>
        <p><strong>Suggested Action:</strong> <code style="color:var(--secondary)">${escapeHtml(data.critique.retry_type)}</code></p>
        <p style="margin-top: 8px;"><strong>Feedback:</strong> <span class="muted-text">${escapeHtml(data.critique.feedback)}</span></p>
      </div>
    `;
  } else {
    criticDetails.innerHTML = `<p class="empty-state">No feedback generated.</p>`;
  }

  // Chunks
  if (data.retrieved_docs && data.retrieved_docs.length) {
    retrievedSources.innerHTML = data.retrieved_docs
      .map((doc) => `
        <li>
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <strong>${escapeHtml(getBasename(doc.source))}</strong>
            <span style="color:var(--secondary)">Score: ${doc.score.toFixed(3)}</span>
          </div>
          <div class="muted-text" style="font-size:0.75rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            ${escapeHtml(doc.content)}
          </div>
        </li>
      `)
      .join("");
  } else {
    retrievedSources.innerHTML = `<li class="empty-state">No chunks retrieved.</li>`;
  }

  // Trace
  if (data.trace && data.trace.length) {
    timelineTrace.innerHTML = data.trace
      .map((t) => {
        let cls = "completed";
        if (t.status === "rejected") cls = "rejected";
        if (t.status === "skipped") cls = "skipped";
        
        return `
          <li class="${cls}">
            <span class="trace-step">${escapeHtml(t.step)}</span>
            <span class="trace-status" style="color: ${t.status === 'rejected' ? 'var(--warning)' : (t.status === 'completed' || t.status === 'approved' ? 'var(--success)' : 'var(--text-muted)')}">${escapeHtml(t.status)}</span>
            <span class="trace-detail">${escapeHtml(t.detail)}</span>
            <span class="trace-time">${t.elapsed_ms ? Math.round(t.elapsed_ms) + 'ms' : ''}</span>
          </li>
        `;
      })
      .join("");
  } else {
    timelineTrace.innerHTML = `<li class="empty-state">No timeline trace.</li>`;
  }

  // Animate Graph Flow
  animateGraphPath(data.trace, data.needs_clarification, data.answer);
}

// Sequential Graph Highlights
async function animateGraphPath(trace, needsClarification, answer) {
  if (!trace || !trace.length) return;
  
  // Highlight nodes sequentially
  for (let i = 0; i < trace.length; i++) {
    const event = trace[i];
    const nodeName = getGraphNodeId(event.step);
    if (!nodeName) continue;

    const el = document.querySelector(`#${nodeName}`);
    if (!el) continue;

    // Apply color highlights based on step status
    el.classList.add("active");
    if (event.status === "rejected") {
      el.classList.add("rejected");
    } else {
      el.classList.add("completed");
    }

    // Toggle loop arrows if relevant
    if (event.step === "grade_context" && event.status === "rejected") {
      loops.retrieval.classList.remove("hidden");
    }
    if (event.step === "critique_answer" && event.status === "rejected") {
      // decide if loop back to generation or retrieval
      if (event.detail.includes("regenerate") || event.detail.includes("citation")) {
        loops.generation.classList.remove("hidden");
      } else {
        loops.retrieval.classList.remove("hidden");
      }
    }

    // Short delay for visual effect
    await sleep(250);
    el.classList.remove("active");
  }

  // Highlight final state node
  let finalNode = "finalize_answer";
  if (needsClarification) {
    finalNode = "clarify_question";
  } else if (answer.includes("I could not produce a sufficiently grounded answer")) {
    finalNode = "fallback_answer";
  }
  
  const finalEl = document.querySelector(`#node-${finalNode}`);
  if (finalEl) {
    finalEl.classList.add("active", "completed");
  }
}

function getGraphNodeId(step) {
  const mapping = {
    "rewrite_query": "node-rewrite_query",
    "retrieve_documents": "node-retrieve_documents",
    "grade_context": "node-grade_context",
    "generate_answer": "node-generate_answer",
    "critique_answer": "node-critique_answer",
    "finalize_answer": "node-finalize_answer",
    "clarify_question": "node-clarify_question",
    "fallback_answer": "node-fallback_answer"
  };
  return mapping[step] || null;
}

// Render Evaluation
function renderEvaluationResults(data) {
  // Update gauge meters
  const hasRagas = data.ragas && !data.ragas.error;
  if (hasRagas) {
    ragasWarning.classList.add("hidden");
    updateGauge("progress-faithfulness", data.ragas.faithfulness);
    updateGauge("progress-relevance", data.ragas.answer_relevance);
    updateGauge("progress-precision", data.ragas.context_precision);
    updateGauge("progress-recall", data.ragas.context_recall);
  } else {
    ragasWarning.classList.remove("hidden");
    if (data.ragas?.error) {
      ragasWarning.innerHTML = `<strong>Ragas Evaluation Execution Issue:</strong> ${escapeHtml(data.ragas.error)}. Displaying fallback simulation.`;
    }
    // Simulate fallback scores based on compliances
    updateGauge("progress-faithfulness", data.citation_rate);
    updateGauge("progress-relevance", data.approval_rate);
    updateGauge("progress-precision", data.approval_rate * 0.95);
    updateGauge("progress-recall", data.citation_rate * 0.9);
  }

  // Stats Card
  evalApprovalRate.textContent = `${Math.round(data.approval_rate * 100)}%`;
  evalCitationRate.textContent = `${Math.round(data.citation_rate * 100)}%`;
  evalFallbackRate.textContent = `${Math.round((1 - data.fallback_rate) * 100)}%`;
  evalAvgLoops.textContent = data.average_attempts.toFixed(2);

  // Table Detail Body
  if (data.results && data.results.length) {
    evalTableBody.innerHTML = data.results
      .map((r) => `
        <tr>
          <td><strong>${escapeHtml(r.question)}</strong></td>
          <td><span class="muted-text">${escapeHtml(r.expected_behavior || "General RAG")}</span></td>
          <td>
            <span class="${r.approved ? 'success-text' : 'danger-text'}">
              ${r.approved ? 'Approved' : 'Rejected'}
            </span>
          </td>
          <td style="font-family:var(--font-mono)">${r.attempts}</td>
          <td>${r.has_sources ? 'Yes' : 'No'}</td>
          <td>${r.used_fallback ? 'Yes' : 'No'}</td>
          <td class="muted-text">${escapeHtml(r.answer_preview)}...</td>
        </tr>
      `)
      .join("");
  } else {
    evalTableBody.innerHTML = `<tr><td colspan="7" class="empty-state">No test results found.</td></tr>`;
  }
}

function updateGauge(elementId, value) {
  const el = document.querySelector(`#${elementId}`);
  if (!el) return;
  
  const pct = Math.round(value * 100);
  el.setAttribute("data-value", value);
  el.querySelector(".progress-value").textContent = `${pct}%`;
  
  // Set radial background sweep
  const deg = value * 360;
  el.style.background = `conic-gradient(var(--primary) ${deg}deg, var(--border-color) 0deg)`;
}

// Helpers
function getBasename(path) {
  return path.split(/[/\\]/).pop();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
