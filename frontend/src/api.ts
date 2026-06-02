const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type RetrievedDocument = {
  content: string;
  source: string;
  score: number;
};

export type Critique = {
  approved: boolean;
  reason: string;
  retry_type: string;
  feedback: string;
};

export type TraceEvent = {
  step: string;
  status: string;
  detail: string;
  at?: string;
  elapsed_ms?: number;
};

export type ObservabilityMetrics = {
  total_elapsed_ms: number;
  total_steps: number;
  retrieval_attempts: number;
  generation_attempts: number;
};

export type AskResponse = {
  answer: string;
  attempts: number;
  sources: string[];
  needs_clarification?: boolean;
  retrieved_docs: RetrievedDocument[];
  critique: Critique | null;
  trace: TraceEvent[];
  metrics: ObservabilityMetrics;
};

export type IngestResponse = {
  chunks_indexed: number;
  output_path: string;
};

export async function ingestDocuments(inputPath: string): Promise<IngestResponse> {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input_path: inputPath }),
  });

  return parseResponse<IngestResponse>(response);
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question }),
  });

  return parseResponse<AskResponse>(response);
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }

  let detail = `Request failed with status ${response.status}`;
  try {
    const error = (await response.json()) as { detail?: string };
    detail = error.detail ?? detail;
  } catch {
    // Keep the HTTP status fallback when the API does not return JSON.
  }

  throw new Error(detail);
}
