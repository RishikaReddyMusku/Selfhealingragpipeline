import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileSearch,
  Loader2,
  RefreshCcw,
  Send,
  ShieldCheck,
} from "lucide-react";
import { FormEvent, useMemo, useState } from "react";

import { AskResponse, askQuestion, ingestDocuments } from "./api";

const defaultQuestion = "What makes this RAG pipeline self-healing?";

function App() {
  const [inputPath, setInputPath] = useState("data/sample_docs");
  const [question, setQuestion] = useState(defaultQuestion);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [status, setStatus] = useState<string>("Ready");
  const [error, setError] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isAsking, setIsAsking] = useState(false);

  const approved = result?.critique?.approved ?? false;
  const sourceCount = result?.sources.length ?? 0;
  const traceCount = result?.trace.length ?? 0;

  const verdictLabel = useMemo(() => {
    if (!result?.critique) {
      return "No verdict yet";
    }

    return result.critique.approved ? "Approved" : "Rejected";
  }, [result]);

  async function handleIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsIngesting(true);
    setStatus("Indexing documents");

    try {
      const response = await ingestDocuments(inputPath);
      setStatus(`Indexed ${response.chunks_indexed} chunks`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Ingestion failed");
      setStatus("Ingestion failed");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setIsAsking(true);
    setStatus("Running self-healing graph");

    try {
      const response = await askQuestion(question);
      setResult(response);
      setStatus(response.critique?.approved ? "Answer approved" : "Fallback returned");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Question failed");
      setStatus("Request failed");
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Project summary">
        <div>
          <p className="eyebrow">LangGraph RAG Control Surface</p>
          <h1>Self-Healing RAG Pipeline</h1>
        </div>
        <div className="status-pill" title="Current system status">
          {isIngesting || isAsking ? <Loader2 className="spin" size={18} /> : <ShieldCheck size={18} />}
          <span>{status}</span>
        </div>
      </section>

      {error ? (
        <section className="alert" role="alert">
          <AlertTriangle size={18} />
          <span>{error}</span>
        </section>
      ) : null}

      <section className="workspace-grid">
        <div className="control-panel">
          <form className="tool-form" onSubmit={handleIngest}>
            <div className="field-group">
              <label htmlFor="input-path">Document path</label>
              <input
                id="input-path"
                value={inputPath}
                onChange={(event) => setInputPath(event.target.value)}
              />
            </div>
            <button className="primary-button" type="submit" disabled={isIngesting}>
              {isIngesting ? <Loader2 className="spin" size={18} /> : <Database size={18} />}
              <span>Ingest</span>
            </button>
          </form>

          <form className="tool-form question-form" onSubmit={handleAsk}>
            <div className="field-group">
              <label htmlFor="question">Question</label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={6}
              />
            </div>
            <button className="primary-button" type="submit" disabled={isAsking}>
              {isAsking ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              <span>Ask</span>
            </button>
          </form>
        </div>

        <div className="answer-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Grounded Response</p>
              <h2>Answer</h2>
            </div>
            <div className={approved ? "verdict approved" : "verdict"}>
              {approved ? <CheckCircle2 size={18} /> : <RefreshCcw size={18} />}
              <span>{verdictLabel}</span>
            </div>
          </div>
          <div className="answer-body">
            {result ? (
              <p>{result.answer}</p>
            ) : (
              <p className="placeholder">Run a question to see the critic-reviewed answer.</p>
            )}
          </div>
        </div>
      </section>

      <section className="metrics-row" aria-label="Run metrics">
        <Metric label="Attempts" value={String(result?.attempts ?? 0)} />
        <Metric label="Sources" value={String(sourceCount)} />
        <Metric label="Trace Events" value={String(traceCount)} />
      </section>

      <section className="details-grid">
        <div className="detail-panel">
          <div className="panel-header compact">
            <div>
              <p className="eyebrow">Critic</p>
              <h2>Verdict</h2>
            </div>
            <ShieldCheck size={20} />
          </div>
          {result?.critique ? (
            <dl className="verdict-list">
              <dt>Reason</dt>
              <dd>{result.critique.reason}</dd>
              <dt>Repair route</dt>
              <dd>{result.critique.retry_type}</dd>
              <dt>Feedback</dt>
              <dd>{result.critique.feedback}</dd>
            </dl>
          ) : (
            <p className="placeholder">The critic verdict will appear after the graph runs.</p>
          )}
        </div>

        <div className="detail-panel">
          <div className="panel-header compact">
            <div>
              <p className="eyebrow">Retrieval</p>
              <h2>Sources</h2>
            </div>
            <FileSearch size={20} />
          </div>
          {result?.sources.length ? (
            <ul className="source-list">
              {result.sources.map((source) => (
                <li key={source}>{source}</li>
              ))}
            </ul>
          ) : (
            <p className="placeholder">Retrieved sources will appear here.</p>
          )}
        </div>
      </section>

      <section className="trace-panel">
        <div className="panel-header compact">
          <div>
            <p className="eyebrow">LangGraph</p>
            <h2>Workflow Trace</h2>
          </div>
          <RefreshCcw size={20} />
        </div>
        {result?.trace.length ? (
          <ol className="trace-list">
            {result.trace.map((event, index) => (
              <li key={`${event.step}-${index}`}>
                <span className="trace-step">{event.step}</span>
                <span className="trace-status">{event.status}</span>
                <p>{event.detail}</p>
              </li>
            ))}
          </ol>
        ) : (
          <p className="placeholder">Each graph node will log its decision here.</p>
        )}
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
