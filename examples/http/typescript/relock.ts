// Minimal typed client for relock's HTTP API (Node 18+, Deno, Bun, browsers). No dependencies.
// Full schema: GET /v1/openapi.json, or docs/openapi.json in the repository.

export interface Clause { id: string; text: string }
export interface Item {
  ref: string; id: string; key: string | null; version: number; status: string; title: string;
  clauses: Clause[]; tools: string[]; confidence: number; confirmations: number; failures: number;
}
export interface Decision {
  expectation_id: string; matched: boolean; verdict: string; reason: string;
  version: number; new_version: number | null; evidence: number; required: number;
}
export interface ToolError {
  message?: string;
  correction?: string; // the rule that would have been right; drives the rewrite
  status?: number;     // HTTP status of the failed call; 408/425/429/5xx are transient
  transient?: boolean;
}

export class RelockError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export class Relock {
  private baseUrl: string;
  private token: string | undefined;
  constructor(baseUrl = "http://127.0.0.1:8765", token?: string) {
    this.baseUrl = baseUrl;
    this.token = token ?? (globalThis as any).process?.env?.RELOCK_TOKEN;
  }

  private async call<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    const res = await fetch(`${this.baseUrl}/v1${path}`, {
      method, headers, body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new RelockError(res.status, data.error?.code, data.error?.message);
    return data as T;
  }

  addItem(title: string, clauses: string[], opts: { key?: string; tools?: string[]; exist_ok?: boolean } = {}) {
    return this.call<Item>("POST", "/items", { title, clauses, ...opts });
  }
  item(ref: string) { return this.call<Item>("GET", `/items/${encodeURIComponent(ref)}`); }
  history(ref: string) {
    return this.call<{ versions: Item[] }>("GET", `/items/${encodeURIComponent(ref)}/history`);
  }

  async episode(task: string): Promise<Episode> {
    const { episode_id } = await this.call<{ episode_id: string }>("POST", "/episodes", { task });
    return new Episode(this, episode_id);
  }

  /** @internal */
  post<T>(path: string, body?: unknown) { return this.call<T>("POST", path, body); }
}

export class Episode {
  private client: Relock;
  readonly id: string;
  constructor(client: Relock, id: string) {
    this.client = client;
    this.id = id;
  }
  /** Recalled items become editable in this episode. Put `context` in the prompt. */
  recall(query: string, k?: number) {
    return this.client.post<{ context: string; items: Item[] }>(`/episodes/${this.id}/recall`, { query, k });
  }
  /** Report a tool call's real outcome. This is the trusted signal relock learns from. */
  async toolResult(tool: string, ok: boolean, error?: ToolError, result?: unknown) {
    const out = await this.client.post<{ observations: Decision[] }>(
      `/episodes/${this.id}/tool-result`, { tool, ok, error, result });
    return out.observations;
  }
  close() { return this.client.post<Record<string, unknown>>(`/episodes/${this.id}/close`); }
}
