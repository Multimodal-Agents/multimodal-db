import { App } from "obsidian";

/** What Obsidian's own graph view saved for this vault (.obsidian/graph.json): colour
 *  groups, the saved filter, forces and display multipliers. We start from the same
 *  values so the plugin graph looks like the one you tuned. */
export interface ColourGroup {
  query: string;
  colour: string;
}

export interface GraphConfig {
  groups: ColourGroup[];
  search: string;
  centerStrength: number;
  repelStrength: number;
  linkStrength: number;
  linkDistance: number;
  nodeSize: number;
  lineSize: number;
  textFade: number;
}

export const DEFAULT_GRAPH_CONFIG: GraphConfig = {
  groups: [],
  search: "",
  centerStrength: 0.2,
  repelStrength: 10,
  linkStrength: 0.6,
  linkDistance: 80,
  nodeSize: 1,
  lineSize: 1,
  textFade: 0,
};

export async function loadGraphConfig(app: App): Promise<GraphConfig> {
  const cfg = { ...DEFAULT_GRAPH_CONFIG };
  try {
    const raw = JSON.parse(await app.vault.adapter.read(".obsidian/graph.json")) as Record<string, unknown>;
    const num = (k: string, d: number) => (typeof raw[k] === "number" ? (raw[k] as number) : d);
    cfg.search = typeof raw.search === "string" ? raw.search : "";
    cfg.centerStrength = num("centerStrength", cfg.centerStrength);
    cfg.repelStrength = num("repelStrength", cfg.repelStrength);
    cfg.linkStrength = num("linkStrength", cfg.linkStrength);
    cfg.linkDistance = num("linkDistance", cfg.linkDistance);
    cfg.nodeSize = num("nodeSizeMultiplier", cfg.nodeSize);
    cfg.lineSize = num("lineSizeMultiplier", cfg.lineSize);
    cfg.textFade = num("textFadeMultiplier", cfg.textFade);
    const groups = Array.isArray(raw.colorGroups) ? (raw.colorGroups as { query: string; color: { rgb: number } }[]) : [];
    cfg.groups = groups
      .filter((g) => g && typeof g.query === "string" && g.color && typeof g.color.rgb === "number")
      .map((g) => ({ query: g.query, colour: "#" + g.color.rgb.toString(16).padStart(6, "0") }));
  } catch {
    /* no graph.json yet: defaults */
  }
  return cfg;
}

/** The subset of Obsidian search syntax the graph uses: `tag:#x`, `path:X`, `file:X`, bare
 *  words (match the name), `-` to negate, space = AND, `OR` = OR, quotes allowed. */
export interface Queryable {
  id: string;
  name: string;
  tags: string[];
}

interface Term {
  neg: boolean;
  field: "tag" | "path" | "file" | "text";
  value: string;
}

const TOKEN = /(-?)(?:(tag|path|file):)?(?:"([^"]*)"|(\S+))/g;

function parse(query: string): Term[][] {
  const groups: Term[][] = [[]];
  for (const m of query.matchAll(TOKEN)) {
    const value = (m[3] ?? m[4] ?? "").trim();
    if (!m[2] && value === "OR") {
      groups.push([]);
      continue;
    }
    if (!value) continue;
    const field = (m[2] as Term["field"] | undefined) ?? "text";
    groups[groups.length - 1].push({ neg: m[1] === "-", field, value: value.replace(/^#/, "").toLowerCase() });
  }
  return groups.filter((g) => g.length);
}

function termMatches(t: Term, q: Queryable): boolean {
  let hit: boolean;
  switch (t.field) {
    case "tag":
      hit = q.tags.some((tag) => {
        const lower = tag.toLowerCase();
        return lower === t.value || lower.startsWith(t.value + "/");
      });
      break;
    case "path":
      hit = (q.id + ".md").toLowerCase().includes(t.value);
      break;
    case "file":
      hit = q.name.toLowerCase().includes(t.value);
      break;
    default:
      hit = q.name.toLowerCase().includes(t.value) || q.id.toLowerCase().includes(t.value);
  }
  return t.neg ? !hit : hit;
}

export function compileQuery(query: string): (q: Queryable) => boolean {
  const groups = parse(query);
  if (!groups.length) return () => true;
  return (q) => groups.some((terms) => terms.every((t) => termMatches(t, q)));
}

/** First matching colour group wins, like Obsidian. */
export function groupColour(groups: ColourGroup[], q: Queryable): string | null {
  for (const g of groups) {
    if (!g.query.trim()) continue;
    if (compileQuery(g.query)(q)) return g.colour;
  }
  return null;
}
