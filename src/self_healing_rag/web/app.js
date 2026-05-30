const statusEl = document.querySelector("#status");
const errorEl = document.querySelector("#error");
const verdictEl = document.querySelector("#verdict");
const answerEl = document.querySelector("#answer");
const attemptsEl = document.querySelector("#attempts");
const sourceCountEl = document.querySelector("#source-count");
const traceCountEl = document.querySelector("#trace-count");
const critiqueEl = document.querySelector("#critique");
const sourcesEl = document.querySelector("#sources");
const traceEl = document.querySelector("#trace");
const ingestForm = document.querySelector("#ingest-form");
const askForm = document.querySelector("#ask-form");
const inputPathEl = document.querySelector("#input-path");
const questionEl = document.querySelector("#question");

ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy("Indexing documents");

  try {
    const payload = await postJson("/ingest", {
      input_path: inputPathEl.value,
    });
    setStatus(`Indexed ${payload.chunks_indexed} chunks`);
  } catch (error) {
    showError(error);
  } finally {
    setButtons(false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy("Running graph");

  try {
    const payload = await postJson("/ask", {
      question: questionEl.value,
    });
    renderAnswer(payload);
    setStatus(payload.critique?.approved ? "Answer approved" : "Fallback returned");
  } catch (error) {
    showError(error);
  } finally {
    setButtons(false);
  }
});

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (response.ok) {
    return response.json();
  }

  let detail = `Request failed with status ${response.status}`;
  try {
    const error = await response.json();
    detail = error.detail ?? detail;
  } catch {
    detail = response.statusText || detail;
  }
  throw new Error(detail);
}

function renderAnswer(payload) {
  hideError();
  const approved = Boolean(payload.critique?.approved);

  verdictEl.textContent = approved ? "Approved" : "Rejected";
  verdictEl.classList.toggle("approved", approved);
  answerEl.textContent = payload.answer;
  answerEl.classList.remove("muted");
  attemptsEl.textContent = String(payload.attempts ?? 0);
  sourceCountEl.textContent = String(payload.sources?.length ?? 0);
  traceCountEl.textContent = String(payload.trace?.length ?? 0);

  renderCritique(payload.critique);
  renderSources(payload.sources ?? []);
  renderTrace(payload.trace ?? []);
}

function renderCritique(critique) {
  if (!critique) {
    critiqueEl.innerHTML = `<dt>Reason</dt><dd class="muted">Pending</dd>`;
    return;
  }

  critiqueEl.innerHTML = `
    <dt>Reason</dt>
    <dd>${escapeHtml(critique.reason)}</dd>
    <dt>Repair route</dt>
    <dd>${escapeHtml(critique.retry_type)}</dd>
    <dt>Feedback</dt>
    <dd>${escapeHtml(critique.feedback)}</dd>
  `;
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesEl.innerHTML = `<li class="muted">Pending</li>`;
    return;
  }

  sourcesEl.innerHTML = sources.map((source) => `<li>${escapeHtml(source)}</li>`).join("");
}

function renderTrace(trace) {
  if (!trace.length) {
    traceEl.innerHTML = `<li class="muted">Pending</li>`;
    return;
  }

  traceEl.innerHTML = trace
    .map(
      (event) => `
        <li>
          <span class="trace-step">${escapeHtml(event.step)}</span>
          <span class="trace-status">${escapeHtml(event.status)}</span>
          <p class="trace-detail">${escapeHtml(event.detail)}</p>
        </li>
      `,
    )
    .join("");
}

function setBusy(message) {
  hideError();
  setStatus(message);
  setButtons(true);
}

function setStatus(message) {
  statusEl.textContent = message;
}

function setButtons(disabled) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = disabled;
  });
}

function showError(error) {
  errorEl.textContent = error instanceof Error ? error.message : "Request failed";
  errorEl.classList.remove("hidden");
  setStatus("Request failed");
}

function hideError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
