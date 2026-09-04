import { requestUrl } from "obsidian";

export interface BrainEntry {
  name: string;
  path: string;
  exists: boolean;
  collection: string;
  docs: number;
  vectors: number;
}

export interface Hit {
  score: number;
  id: string;
  title: string | null;
  url: string | null;
  text: string;
  layer?: string | null;
}

export interface Related {
  id: string;
  title: string | null;
  score: number;
  kind: string;
  url: string | null;
  layer?: string | null;
}

export interface GraphNode {
  id: string;
  name: string;
  layer: string;
  kind: string;
  state: string | null;
  title: string;
  tags: string[];
  in: number;
  out: number;
  has_thinking: boolean;
  generated: boolean;
}

export interface GraphLink {
  source: string;
  target: string;
  kind: "wikilink" | "similar";
  score?: number;
}

export interface Graph {
  brain: string;
  indexed: boolean;
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface Health {
  ok: boolean;
  version: string;
  root: string;
  brains: BrainEntry[];
}

/** Thin client over the JSON routes of `mmdb serve`. Uses Obsidian's requestUrl, which skips CORS. */
export class MmdbClient {
  constructor(private base: string) {}

  setBase(base: string): void {
    this.base = base.replace(/\/+$/, "");
  }

  private async call<T>(method: "GET" | "POST", path: string, params: Record<string, string | number> = {}): Promise<T> {
    const query = Object.entries(params)
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
      .join("&");
    const url = `${this.base}${path}${query ? "?" + query : ""}`;
    const res = await requestUrl({ url, method, throw: false });
    if (res.status >= 400) {
      let msg = `${res.status} from ${path}`;
      try {
        msg = (res.json as { error?: string }).error ?? msg;
      } catch {
        /* not json */
      }
      throw new Error(msg);
    }
    return res.json as T;
  }

  health(): Promise<Health> {
    return this.call("GET", "/api/health");
  }

  brains(): Promise<BrainEntry[]> {
    return this.call("GET", "/api/brains");
  }

  search(brain: string, q: string, k = 12): Promise<Hit[]> {
    return this.call("GET", `/api/brains/${encodeURIComponent(brain)}/search`, { q, k });
  }

  related(brain: string, note: string, k = 10): Promise<Related[]> {
    return this.call("GET", `/api/brains/${encodeURIComponent(brain)}/related`, { note, k });
  }

  graph(brain: string, k: number, min: number, similar: boolean): Promise<Graph> {
    return this.call("GET", `/api/brains/${encodeURIComponent(brain)}/graph`, { k, min, similar: similar ? 1 : 0 });
  }

  load(brain: string): Promise<{ counts: Record<string, number>; docs: number; vectors: number }> {
    return this.call("POST", `/api/brains/${encodeURIComponent(brain)}/load`);
  }
}
