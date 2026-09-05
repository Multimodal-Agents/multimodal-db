import { ItemView, TFile, WorkspaceLeaf } from "obsidian";
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  Simulation,
  SimulationLinkDatum,
  SimulationNodeDatum,
} from "d3-force";
import { select } from "d3-selection";
import { zoom, zoomIdentity, ZoomTransform } from "d3-zoom";
import type MmdbPlugin from "../main";
import type { Graph, GraphLink, GraphNode, MapPoint } from "../api";
import { compileQuery, GraphConfig, groupColour, loadGraphConfig } from "../vaultgraph";

export const GRAPH_VIEW = "mmdb-graph";

type Mode = "global" | "local" | "map" | "clusters" | "tags";
type ColourBy = "groups" | "layer" | "state" | "age";

const MODES: [Mode, string][] = [
  ["global", "Global"],
  ["local", "Local (this note)"],
  ["map", "Semantic map"],
  ["clusters", "Clusters"],
  ["tags", "Tags"],
];
const COLOUR_BY: [ColourBy, string][] = [
  ["groups", "vault colour groups"],
  ["layer", "layer"],
  ["state", "state"],
  ["age", "age"],
];
const LAYERS = ["Open", "Archive", "Clusters", "Concepts", "Threads", "Reference", "Hardware", "DAILY TODOS"];

const LAYER_COLOURS: Record<string, string> = {
  Open: "#2ecc40",
  Archive: "#8a8a8a",
  Clusters: "#7afdff",
  Concepts: "#00eeff",
  Threads: "#9df5fb",
  Reference: "#ffb700",
  Hardware: "#0040ff",
  "DAILY TODOS": "#ffdc2e",
  Discord: "#ff8e42",
  "(root)": "#ffffff",
};
const STATE_COLOURS: Record<string, string> = { open: "#2ecc40", merged: "#7851a9", closed: "#ff2b2b" };
const NEUTRAL = "#a0a0a0";

interface N extends SimulationNodeDatum {
  id: string;
  label: string;
  kind: "note" | "cluster" | "tag";
  size: number;
  colour: string;
  note?: GraphNode;
  members?: number;
  open?: string;
}
interface L extends SimulationLinkDatum<N> {
  kind: "wikilink" | "similar" | "weight";
  score?: number;
  weight?: number;
}

/** The graph pane. Five modes over one canvas: the whole vault, the neighbourhood of the
 *  note you are reading, a semantic map laid out from the embeddings, the clusters as
 *  super-nodes, and tag co-occurrence. Colours, the saved filter and the forces come from
 *  the vault's own graph.json so it starts out looking like Obsidian's graph. */
export class GraphView extends ItemView {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private status!: HTMLElement;
  private filterInput!: HTMLInputElement;
  private depthWrap!: HTMLElement;
  private layerWrap!: HTMLElement;
  private forcesWrap!: HTMLElement;
  private sim: Simulation<N, L> | null = null;
  private nodes: N[] = [];
  private links: L[] = [];
  private byId = new Map<string, N>();
  private neighbours = new Map<string, Set<string>>();
  private transform: ZoomTransform = zoomIdentity;
  private hover: N | null = null;
  private drag: N | null = null;
  private moved = false;
  private graph: Graph | null = null;
  private mapPoints: Map<string, MapPoint> | null = null;
  private cfg: GraphConfig | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private raf = 0;

  private mode: Mode = "global";
  private colourBy: ColourBy = "groups";
  private showSimilar = true;
  private showWikilinks = true;
  private minScore: number;
  private depth = 1;
  private hidden: Set<string>;
  private filter = "";
  private centreId: string | null = null;
  private repel: number;
  private linkDistance: number;
  private centre: number;

  constructor(leaf: WorkspaceLeaf, private plugin: MmdbPlugin) {
    super(leaf);
    this.hidden = new Set(plugin.settings.hideLayers);
    this.minScore = plugin.settings.minScore;
    this.repel = 10;
    this.linkDistance = 80;
    this.centre = 0.2;
  }

  getViewType(): string {
    return GRAPH_VIEW;
  }

  getDisplayText(): string {
    return "Multimodal DB graph";
  }

  getIcon(): string {
    return "radar";
  }

  // ---- ui ---------------------------------------------------------------------------------

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("mmdb-view");
    root.addClass("mmdb-graph-root");

    this.cfg = await loadGraphConfig(this.app);
    this.repel = this.cfg.repelStrength;
    this.linkDistance = this.cfg.linkDistance;
    this.centre = this.cfg.centerStrength;
    this.filter = this.cfg.search;

    const row1 = root.createDiv({ cls: "mmdb-bar" });
    const modeSel = row1.createEl("select", { cls: "dropdown mmdb-select" });
    for (const [v, label] of MODES) modeSel.createEl("option", { value: v, text: label });
    modeSel.value = this.mode;
    modeSel.addEventListener("change", () => {
      this.mode = modeSel.value as Mode;
      this.syncControls();
      void this.rebuild();
    });
    this.filterInput = row1.createEl("input", { type: "text", placeholder: "filter: tag:#state/open -path:Views", cls: "mmdb-input" });
    this.filterInput.value = this.filter;
    this.filterInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        this.filter = this.filterInput.value;
        void this.rebuild();
      }
    });
    const colourSel = row1.createEl("select", { cls: "dropdown mmdb-select" });
    for (const [v, label] of COLOUR_BY) colourSel.createEl("option", { value: v, text: "colour: " + label });
    colourSel.value = this.colourBy;
    colourSel.addEventListener("change", () => {
      this.colourBy = colourSel.value as ColourBy;
      this.recolour();
    });
    const reload = row1.createEl("button", { cls: "mmdb-button", text: "reload" });
    reload.addEventListener("click", () => void this.fetch(true));

    const row2 = root.createDiv({ cls: "mmdb-bar" });
    this.toggle(row2, "wikilinks", this.showWikilinks, (v) => {
      this.showWikilinks = v;
      void this.rebuild();
    });
    this.toggle(row2, "similar", this.showSimilar, (v) => {
      this.showSimilar = v;
      void this.rebuild();
    });
    this.slider(row2, "min similarity", 0.5, 0.99, 0.01, this.minScore, (v) => {
      this.minScore = v;
      void this.rebuild();
    });
    this.depthWrap = row2.createSpan();
    this.slider(this.depthWrap, "depth", 1, 4, 1, this.depth, (v) => {
      this.depth = v;
      void this.rebuild();
    });
    const forcesBtn = row2.createEl("button", { cls: "mmdb-button", text: "forces" });
    forcesBtn.addEventListener("click", () => this.forcesWrap.toggleClass("mmdb-hidden", !this.forcesWrap.hasClass("mmdb-hidden")));

    this.layerWrap = root.createDiv({ cls: "mmdb-bar" });
    for (const layer of LAYERS) {
      this.toggle(this.layerWrap, layer, !this.hidden.has(layer), (v) => {
        if (v) this.hidden.delete(layer);
        else this.hidden.add(layer);
        void this.rebuild();
      });
    }

    this.forcesWrap = root.createDiv({ cls: "mmdb-bar mmdb-hidden" });
    this.slider(this.forcesWrap, "repel", 1, 40, 0.5, this.repel, (v) => {
      this.repel = v;
      this.reheat();
    });
    this.slider(this.forcesWrap, "link distance", 20, 300, 5, this.linkDistance, (v) => {
      this.linkDistance = v;
      this.reheat();
    });
    this.slider(this.forcesWrap, "centre", 0, 1, 0.02, this.centre, (v) => {
      this.centre = v;
      this.reheat();
    });

    this.status = root.createDiv({ cls: "mmdb-status" });
    this.canvas = root.createEl("canvas", { cls: "mmdb-canvas" });
    this.ctx = this.canvas.getContext("2d")!;
    this.wire();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(root);
    this.registerEvent(this.app.workspace.on("active-leaf-change", () => this.onActiveNote()));
    this.registerEvent(this.app.workspace.on("file-open", () => this.onActiveNote()));
    this.syncControls();
    this.resize();
    await this.fetch(false);
  }

  async onClose(): Promise<void> {
    this.sim?.stop();
    this.resizeObserver?.disconnect();
    cancelAnimationFrame(this.raf);
  }

  private toggle(parent: HTMLElement, label: string, initial: boolean, onChange: (v: boolean) => void): void {
    const wrap = parent.createEl("label", { cls: "mmdb-toggle" });
    const box = wrap.createEl("input", { type: "checkbox" });
    box.checked = initial;
    wrap.createSpan({ text: label });
    box.addEventListener("change", () => onChange(box.checked));
  }

  private slider(parent: HTMLElement, label: string, min: number, max: number, step: number, value: number, onChange: (v: number) => void): void {
    const wrap = parent.createEl("label", { cls: "mmdb-toggle" });
    wrap.createSpan({ text: label });
    const input = wrap.createEl("input", { type: "range", cls: "mmdb-range" });
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    const out = wrap.createSpan({ cls: "mmdb-range-value", text: String(value) });
    input.addEventListener("input", () => {
      const v = Number(input.value);
      out.setText(step < 1 ? v.toFixed(2) : String(v));
      onChange(v);
    });
  }

  private syncControls(): void {
    this.depthWrap.toggleClass("mmdb-hidden", this.mode !== "local");
    this.layerWrap.toggleClass("mmdb-hidden", this.mode === "tags" || this.mode === "clusters");
  }

  private onActiveNote(): void {
    const f = this.app.workspace.getActiveFile();
    if (!(f instanceof TFile) || f.extension !== "md") return;
    const id = f.path.replace(/\.md$/, "");
    if (id === this.centreId) return;
    this.centreId = id;
    if (this.mode === "local") void this.rebuild();
    else this.draw();
  }

  // ---- data -------------------------------------------------------------------------------

  async fetch(force: boolean): Promise<void> {
    const brain = this.plugin.brain;
    if (!brain) {
      this.status.setText("no brain matched this vault; set one in the plugin settings");
      return;
    }
    this.status.setText(`loading ${brain}...`);
    try {
      if (force || !this.graph) {
        this.graph = await this.plugin.client.graph(brain, this.plugin.settings.similarK, 0.5, true);
        this.mapPoints = null;
      }
      if (!this.graph.indexed) this.status.setText("brain is not indexed yet: run the reindex command");
      if (this.mode === "map" && !this.mapPoints) {
        const m = await this.plugin.client.map(brain);
        this.mapPoints = new Map(m.points.map((p) => [p.id, p]));
      }
      const f = this.app.workspace.getActiveFile();
      if (f instanceof TFile && f.extension === "md") this.centreId = f.path.replace(/\.md$/, "");
      await this.rebuild();
    } catch (e) {
      this.status.setText(`error: ${(e as Error).message}`);
    }
  }

  private visibleNotes(): GraphNode[] {
    if (!this.graph) return [];
    const q = compileQuery(this.filter);
    return this.graph.nodes.filter((n) => !this.hidden.has(n.layer) && q({ id: n.id, name: n.name, tags: n.tags }));
  }

  private noteLinks(keep: Set<string>): GraphLink[] {
    if (!this.graph) return [];
    return this.graph.links.filter((l) => {
      if (!keep.has(l.source) || !keep.has(l.target)) return false;
      if (l.kind === "similar") return this.showSimilar && (l.score ?? 0) >= this.minScore;
      return this.showWikilinks;
    });
  }

  private colourOf(n: GraphNode): string {
    switch (this.colourBy) {
      case "layer":
        return LAYER_COLOURS[n.layer] ?? NEUTRAL;
      case "state":
        return STATE_COLOURS[n.state ?? ""] ?? NEUTRAL;
      case "age": {
        const when = Date.parse(n.updated || n.created || "") || (n.mtime ? n.mtime * 1000 : 0);
        if (!when) return NEUTRAL;
        const days = (Date.now() - when) / 86400000;
        const t = Math.max(0, Math.min(1, days / 120));
        return `hsl(${Math.round(160 - 160 * t)}, ${Math.round(90 - 50 * t)}%, ${Math.round(60 - 25 * t)}%)`;
      }
      default:
        return groupColour(this.cfg?.groups ?? [], { id: n.id, name: n.name, tags: n.tags }) ?? LAYER_COLOURS[n.layer] ?? NEUTRAL;
    }
  }

  private recolour(): void {
    for (const n of this.nodes) if (n.note) n.colour = this.colourOf(n.note);
    this.draw();
  }

  private mkNote(n: GraphNode, prev?: N): N {
    const size = (2.5 + Math.sqrt(n.in + n.out)) * 0.8 * (this.cfg?.nodeSize ?? 1);
    const node: N = prev ?? { id: n.id, label: n.name, kind: "note", size, colour: NEUTRAL, note: n };
    node.note = n;
    node.size = size;
    node.colour = this.colourOf(n);
    node.open = n.id;
    return node;
  }

  private async rebuild(): Promise<void> {
    if (!this.graph) return;
    const old = this.byId;
    let nodes: N[] = [];
    let links: L[] = [];
    let sizeNote = "";

    if (this.mode === "global" || this.mode === "local" || this.mode === "map") {
      let notes = this.visibleNotes();
      let keep = new Set(notes.map((n) => n.id));
      let raw = this.noteLinks(keep);

      if (this.mode === "local") {
        if (!this.centreId || !keep.has(this.centreId)) {
          this.status.setText(this.centreId ? "the active note is filtered out or not in the brain" : "open a note to see its neighbourhood");
          this.nodes = [];
          this.links = [];
          this.draw();
          return;
        }
        const adj = new Map<string, Set<string>>();
        for (const l of raw) {
          if (!adj.has(l.source)) adj.set(l.source, new Set());
          if (!adj.has(l.target)) adj.set(l.target, new Set());
          adj.get(l.source)!.add(l.target);
          adj.get(l.target)!.add(l.source);
        }
        const reached = new Set([this.centreId]);
        let frontier = [this.centreId];
        for (let d = 0; d < this.depth && frontier.length; d++) {
          const next: string[] = [];
          for (const id of frontier) for (const nb of adj.get(id) ?? []) if (!reached.has(nb)) {
            reached.add(nb);
            next.push(nb);
          }
          frontier = next;
        }
        notes = notes.filter((n) => reached.has(n.id));
        keep = reached;
        raw = raw.filter((l) => keep.has(l.source) && keep.has(l.target));
      }

      if (this.mode === "map") {
        if (!this.mapPoints) {
          await this.fetch(false);
          return;
        }
        notes = notes.filter((n) => this.mapPoints!.has(n.id));
        keep = new Set(notes.map((n) => n.id));
        raw = raw.filter((l) => keep.has(l.source) && keep.has(l.target));
        sizeNote = " · position = meaning (t-SNE of the note embeddings)";
      }

      nodes = notes.map((n) => this.mkNote(n, old.get(n.id)));
      const byId = new Map(nodes.map((n) => [n.id, n]));
      links = raw.map((l) => ({ source: byId.get(l.source)!, target: byId.get(l.target)!, kind: l.kind, score: l.score }));

      if (this.mode === "map") {
        const { width, height } = this.size();
        const scale = Math.min(width, height) * 0.45;
        for (const n of nodes) {
          const p = this.mapPoints!.get(n.id)!;
          n.x = n.fx = width / 2 + p.x * scale;
          n.y = n.fy = height / 2 + p.y * scale;
        }
      } else {
        for (const n of nodes) {
          n.fx = null;
          n.fy = null;
        }
      }
    } else if (this.mode === "clusters") {
      const q = compileQuery(this.filter);
      const notes = this.graph.nodes.filter((n) => q({ id: n.id, name: n.name, tags: n.tags }));
      const clusterOf = new Map<string, string>();
      const clusterNote = new Map<string, GraphNode>();
      for (const n of notes) {
        const m = n.layer === "Clusters" ? /^Cluster\s+(\S+)/.exec(n.name) : null;
        if (m) clusterNote.set(m[1].replace(/[^\w]/g, ""), n);
        const tag = n.tags.find((t) => t.toLowerCase().startsWith("cluster/"));
        if (tag) clusterOf.set(n.id, tag.slice(8));
      }
      const members = new Map<string, number>();
      for (const c of clusterOf.values()) members.set(c, (members.get(c) ?? 0) + 1);
      for (const c of clusterNote.keys()) if (!members.has(c)) members.set(c, 0);
      const clusters = [...members.keys()].sort();
      nodes = clusters.map((c, i) => {
        const note = clusterNote.get(c);
        const count = members.get(c) ?? 0;
        const prev = old.get("cluster:" + c);
        const node: N = prev ?? { id: "cluster:" + c, label: "", kind: "cluster", size: 1, colour: NEUTRAL };
        node.label = note ? note.name : `Cluster ${c}`;
        node.members = count;
        node.size = 6 + Math.sqrt(count) * 3;
        node.colour = note ? this.colourOf(note) : `hsl(${(i * 47) % 360}, 70%, 60%)`;
        node.open = note?.id;
        node.note = note;
        node.fx = null;
        node.fy = null;
        return node;
      });
      const byId = new Map(nodes.map((n) => [n.id, n]));
      const weights = new Map<string, number>();
      for (const l of this.graph.links) {
        if (l.kind === "similar" ? !this.showSimilar || (l.score ?? 0) < this.minScore : !this.showWikilinks) continue;
        const a = clusterOf.get(l.source), b = clusterOf.get(l.target);
        if (!a || !b || a === b) continue;
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        weights.set(key, (weights.get(key) ?? 0) + 1);
      }
      for (const [key, w] of weights) {
        const [a, b] = key.split("|");
        links.push({ source: byId.get("cluster:" + a)!, target: byId.get("cluster:" + b)!, kind: "weight", weight: w });
      }
      sizeNote = " · size = members, line = cross-links";
    } else {
      const q = compileQuery(this.filter);
      const notes = this.graph.nodes.filter((n) => !this.hidden.has(n.layer));
      const count = new Map<string, number>();
      const pair = new Map<string, number>();
      for (const n of notes) {
        const tags = [...new Set(n.tags)].filter((t) => q({ id: t, name: t, tags: [t] }));
        for (const t of tags) count.set(t, (count.get(t) ?? 0) + 1);
        tags.sort();
        for (let i = 0; i < tags.length; i++) for (let j = i + 1; j < tags.length; j++) {
          const key = `${tags[i]}|${tags[j]}`;
          pair.set(key, (pair.get(key) ?? 0) + 1);
        }
      }
      nodes = [...count.entries()].map(([t, c], i) => {
        const prev = old.get("tag:" + t);
        const node: N = prev ?? { id: "tag:" + t, label: t, kind: "tag", size: 1, colour: NEUTRAL };
        node.members = c;
        node.size = 3 + Math.sqrt(c) * 1.6;
        const ns = t.split("/")[0];
        node.colour = ns === "state" ? STATE_COLOURS[t.split("/")[1]] ?? NEUTRAL : `hsl(${(hash(ns) + i * 0) % 360}, 65%, 60%)`;
        node.fx = null;
        node.fy = null;
        return node;
      });
      const byId = new Map(nodes.map((n) => [n.id, n]));
      for (const [key, w] of pair) {
        if (w < 2) continue;
        const [a, b] = key.split("|");
        links.push({ source: byId.get("tag:" + a)!, target: byId.get("tag:" + b)!, kind: "weight", weight: w });
      }
      sizeNote = " · size = notes with the tag, line = co-occurrence";
    }

    this.nodes = nodes;
    this.links = links;
    this.byId = new Map(nodes.map((n) => [n.id, n]));
    this.neighbours = new Map();
    for (const l of links) {
      const s = (l.source as N).id, t = (l.target as N).id;
      if (!this.neighbours.has(s)) this.neighbours.set(s, new Set());
      if (!this.neighbours.has(t)) this.neighbours.set(t, new Set());
      this.neighbours.get(s)!.add(t);
      this.neighbours.get(t)!.add(s);
    }
    const similar = links.filter((l) => l.kind === "similar").length;
    const wiki = links.filter((l) => l.kind === "wikilink").length;
    const what = this.mode === "tags" ? "tags" : this.mode === "clusters" ? "clusters" : "notes";
    const edges = this.mode === "tags" || this.mode === "clusters" ? `${links.length} links` : `${wiki} wikilinks, ${similar} similar`;
    this.status.setText(`${nodes.length} ${what}, ${edges}${sizeNote}`);
    this.restart();
  }

  // ---- simulation -------------------------------------------------------------------------

  private restart(): void {
    this.sim?.stop();
    const { width, height } = this.size();
    const fixed = this.mode === "map";
    const sim = forceSimulation<N, L>(this.nodes).on("tick", () => this.schedule());
    if (!fixed) {
      sim
        .force("link", forceLink<N, L>(this.links).id((d) => d.id).distance((l) => this.distance(l)).strength((l) => this.strength(l)))
        .force("charge", forceManyBody<N>().strength(-this.repel * 4.5))
        .force("x", forceX<N>(width / 2).strength(this.centre * 0.4))
        .force("y", forceY<N>(height / 2).strength(this.centre * 0.4))
        .force("collide", forceCollide<N>((d) => d.size + 1.5));
      if (this.mode === "local" && this.centreId) {
        const c = this.byId.get(this.centreId);
        if (c) {
          c.fx = width / 2;
          c.fy = height / 2;
        }
      }
      sim.alpha(1);
    } else {
      sim.alpha(0).stop();
      this.draw();
    }
    this.sim = sim;
  }

  private distance(l: L): number {
    const base = this.linkDistance * 0.6;
    if (l.kind === "similar") return base * 0.6;
    if (l.kind === "weight") return base * 1.4;
    return base;
  }

  private strength(l: L): number {
    const base = (this.cfg?.linkStrength ?? 0.6) * 0.5;
    if (l.kind === "similar") return base * (l.score ?? 0.8);
    if (l.kind === "weight") return Math.min(1, base * Math.log2(1 + (l.weight ?? 1)) * 0.3);
    return base;
  }

  private reheat(): void {
    if (!this.sim || this.mode === "map") return;
    const { width, height } = this.size();
    this.sim
      .force("link", forceLink<N, L>(this.links).id((d) => d.id).distance((l) => this.distance(l)).strength((l) => this.strength(l)))
      .force("charge", forceManyBody<N>().strength(-this.repel * 4.5))
      .force("x", forceX<N>(width / 2).strength(this.centre * 0.4))
      .force("y", forceY<N>(height / 2).strength(this.centre * 0.4))
      .alpha(0.6)
      .restart();
  }

  // ---- drawing ----------------------------------------------------------------------------

  private resize(): void {
    const rect = this.contentEl.getBoundingClientRect();
    const top = this.status ? this.status.offsetTop + this.status.offsetHeight : 0;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, rect.width);
    const h = Math.max(100, rect.height - top);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  private size(): { width: number; height: number } {
    const dpr = window.devicePixelRatio || 1;
    return { width: this.canvas.width / dpr, height: this.canvas.height / dpr };
  }

  private schedule(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.draw();
    });
  }

  private labelThreshold(): number {
    const fade = this.cfg?.textFade ?? 0;
    return Math.max(0.9, 1.6 * Math.pow(2, fade / 2));
  }

  private draw(): void {
    const ctx = this.ctx;
    const { width, height } = this.size();
    const k = this.transform.k;
    const focus = this.hover ?? (this.mode !== "local" && this.centreId ? this.byId.get(this.centreId) ?? null : null);
    const near = focus ? this.neighbours.get(focus.id) ?? new Set<string>() : null;
    const dim = (n: N) => !!focus && !!near && n !== focus && !near.has(n.id);
    const lineSize = this.cfg?.lineSize ?? 1;

    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.translate(this.transform.x, this.transform.y);
    ctx.scale(k, k);

    for (const l of this.links) {
      const s = l.source as N, t = l.target as N;
      if (s.x == null || t.x == null) continue;
      const faded = focus ? s !== focus && t !== focus : false;
      let alpha: number, width: number, colour: string;
      if (l.kind === "similar") {
        alpha = Math.max(0.12, ((l.score ?? 0.8) - 0.6) * 2);
        width = 1.1 * lineSize;
        colour = "255, 120, 60";
      } else if (l.kind === "weight") {
        alpha = Math.min(0.9, 0.15 + Math.log2(1 + (l.weight ?? 1)) * 0.12);
        width = Math.min(8, 0.6 + Math.log2(1 + (l.weight ?? 1))) * lineSize;
        colour = "160, 160, 220";
      } else {
        alpha = 0.28;
        width = 0.7 * lineSize;
        colour = "170, 170, 170";
      }
      if (faded) alpha *= 0.15;
      else if (focus) alpha = Math.min(1, alpha * 2.2);
      ctx.strokeStyle = `rgba(${colour}, ${alpha})`;
      ctx.lineWidth = width / Math.sqrt(k);
      ctx.beginPath();
      ctx.moveTo(s.x, s.y!);
      ctx.lineTo(t.x, t.y!);
      ctx.stroke();
    }

    for (const n of this.nodes) {
      if (n.x == null) continue;
      ctx.beginPath();
      ctx.arc(n.x, n.y!, n.size, 0, Math.PI * 2);
      ctx.fillStyle = n.colour;
      ctx.globalAlpha = dim(n) ? 0.18 : n.note && n.note.generated && !n.note.has_thinking ? 0.75 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (n === focus) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2 / k;
        ctx.stroke();
      }
    }

    const threshold = this.labelThreshold();
    const all = k >= threshold;
    const fade = all ? Math.min(1, (k - threshold) / (threshold * 0.6) + 0.35) : 0;
    const big = this.nodes.length <= 60 ? new Set(this.nodes) : new Set([...this.nodes].sort((a, b) => b.size - a.size).slice(0, 12));
    ctx.font = `${11 / k}px sans-serif`;
    ctx.textBaseline = "middle";
    for (const n of this.nodes) {
      if (n.x == null) continue;
      const isFocus = n === focus, isNear = !!near && near.has(n.id);
      let alpha = 0;
      if (isFocus) alpha = 1;
      else if (isNear) alpha = 0.95;
      else if (focus) alpha = 0;
      else if (all) alpha = fade;
      else if (big.has(n)) alpha = 0.8;
      if (alpha <= 0) continue;
      ctx.fillStyle = `rgba(235, 235, 235, ${alpha})`;
      ctx.fillText(n.label, n.x + n.size + 2 / k, n.y!);
    }
    ctx.restore();
  }

  // ---- interaction ------------------------------------------------------------------------

  private nodeAt(px: number, py: number): N | null {
    const [x, y] = this.transform.invert([px, py]);
    let best: N | null = null, bestD = Infinity;
    for (const n of this.nodes) {
      if (n.x == null) continue;
      const d = Math.hypot(n.x - x, n.y! - y);
      if (d < n.size + 3 / this.transform.k && d < bestD) {
        best = n;
        bestD = d;
      }
    }
    return best;
  }

  private wire(): void {
    const canvas = this.canvas;
    const z = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.08, 10])
      .filter((e: Event) => !this.drag && (e.type !== "mousedown" || (e as MouseEvent).button === 0))
      .on("zoom", (e) => {
        this.transform = e.transform;
        this.draw();
      });
    select(canvas).call(z).on("dblclick.zoom", null);

    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (this.drag) {
        const [x, y] = this.transform.invert([px, py]);
        this.drag.fx = x;
        this.drag.fy = y;
        this.moved = true;
        if (this.mode === "map") {
          this.drag.x = x;
          this.drag.y = y;
          this.draw();
        } else this.sim?.alpha(0.3).restart();
        return;
      }
      const h = this.nodeAt(px, py);
      if (h !== this.hover) {
        this.hover = h;
        canvas.style.cursor = h ? "pointer" : "default";
        canvas.title = h?.note ? h.note.title || h.note.name : h ? `${h.label} (${h.members ?? 0})` : "";
        this.draw();
      }
    });
    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const r = canvas.getBoundingClientRect();
      const n = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (n) {
        this.drag = n;
        this.moved = false;
        n.fx = n.x;
        n.fy = n.y;
        e.stopPropagation();
      }
    });
    const release = () => {
      if (this.drag) {
        if (this.mode !== "map" && !(this.mode === "local" && this.drag.id === this.centreId)) {
          this.drag.fx = null;
          this.drag.fy = null;
        }
        this.drag = null;
      }
    };
    canvas.addEventListener("mouseup", release);
    canvas.addEventListener("mouseleave", () => {
      release();
      if (this.hover) {
        this.hover = null;
        this.draw();
      }
    });
    canvas.addEventListener("click", (e) => {
      if (this.moved) return;
      const r = canvas.getBoundingClientRect();
      const n = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (!n) return;
      if (n.kind === "tag") {
        this.searchTag(n.label);
        return;
      }
      if (n.open) void this.plugin.openNote(n.open, e.ctrlKey || e.metaKey);
    });
    canvas.addEventListener("dblclick", (e) => {
      const r = canvas.getBoundingClientRect();
      const n = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (n?.note && this.mode !== "local") {
        this.centreId = n.note.id;
        this.mode = "local";
        const sel = this.contentEl.querySelector("select.mmdb-select") as HTMLSelectElement | null;
        if (sel) sel.value = "local";
        this.syncControls();
        void this.rebuild();
      }
    });
  }

  private searchTag(tag: string): void {
    const plugins = (this.app as unknown as { internalPlugins?: { getPluginById?: (id: string) => { instance?: { openGlobalSearch?: (q: string) => void } } | null } }).internalPlugins;
    const search = plugins?.getPluginById?.("global-search")?.instance;
    if (search?.openGlobalSearch) search.openGlobalSearch(`tag:#${tag}`);
  }
}

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}
