import { ItemView, WorkspaceLeaf } from "obsidian";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, Simulation, SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import { select } from "d3-selection";
import { zoom, zoomIdentity, ZoomTransform } from "d3-zoom";
import type MmdbPlugin from "../main";
import type { Graph, GraphNode } from "../api";

export const GRAPH_VIEW = "mmdb-graph";

interface N extends SimulationNodeDatum, GraphNode {}
interface L extends SimulationLinkDatum<N> {
  kind: "wikilink" | "similar";
  score?: number;
}

const LAYER_COLOURS: Record<string, string> = {
  Open: "#2ec4b6",
  Archive: "#7a7a7a",
  Clusters: "#7bd3ff",
  Concepts: "#00eeff",
  Threads: "#9df3ff",
  Reference: "#ffb300",
  Hardware: "#0040ff",
  "DAILY TODOS": "#ffe66d",
  "(root)": "#ffffff",
};

/** A force-directed graph of the vault with two edge kinds: Obsidian's wikilinks and
 *  multimodal-db's nearest-neighbour similarity, drawn on a canvas so a thousand nodes stay fluid. */
export class GraphView extends ItemView {
  private canvas!: HTMLCanvasElement;
  private ctx!: CanvasRenderingContext2D;
  private status!: HTMLElement;
  private sim: Simulation<N, L> | null = null;
  private nodes: N[] = [];
  private links: L[] = [];
  private transform: ZoomTransform = zoomIdentity;
  private hover: N | null = null;
  private drag: N | null = null;
  private showSimilar = true;
  private showWikilinks = true;
  private hidden = new Set<string>();
  private graph: Graph | null = null;
  private resizeObserver: ResizeObserver | null = null;

  constructor(leaf: WorkspaceLeaf, private plugin: MmdbPlugin) {
    super(leaf);
    this.hidden = new Set(plugin.settings.hideLayers);
  }

  getViewType(): string {
    return GRAPH_VIEW;
  }

  getDisplayText(): string {
    return "Multimodal DB graph";
  }

  getIcon(): string {
    return "git-fork";
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("mmdb-view");
    root.addClass("mmdb-graph-root");

    const bar = root.createDiv({ cls: "mmdb-bar mmdb-graph-bar" });
    const toggle = (label: string, initial: boolean, onChange: (v: boolean) => void) => {
      const wrap = bar.createEl("label", { cls: "mmdb-toggle" });
      const box = wrap.createEl("input", { type: "checkbox" });
      box.checked = initial;
      wrap.createSpan({ text: label });
      box.addEventListener("change", () => onChange(box.checked));
    };
    toggle("wikilinks", true, (v) => {
      this.showWikilinks = v;
      this.rebuild();
    });
    toggle("similar", true, (v) => {
      this.showSimilar = v;
      this.rebuild();
    });
    for (const layer of ["Open", "Archive", "Clusters", "Concepts", "Threads", "Reference", "Hardware"]) {
      toggle(layer, !this.hidden.has(layer), (v) => {
        if (v) this.hidden.delete(layer);
        else this.hidden.add(layer);
        this.rebuild();
      });
    }
    const reload = bar.createEl("button", { cls: "mmdb-button", text: "reload" });
    reload.addEventListener("click", () => void this.fetch());
    this.status = root.createDiv({ cls: "mmdb-status" });

    this.canvas = root.createEl("canvas", { cls: "mmdb-canvas" });
    this.ctx = this.canvas.getContext("2d")!;
    this.wire();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(root);
    this.resize();
    await this.fetch();
  }

  async onClose(): Promise<void> {
    this.sim?.stop();
    this.resizeObserver?.disconnect();
  }

  private resize(): void {
    const rect = this.contentEl.getBoundingClientRect();
    const barH = this.status.offsetTop + this.status.offsetHeight;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(100, rect.width);
    const h = Math.max(100, rect.height - barH);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  async fetch(): Promise<void> {
    const brain = this.plugin.brain;
    if (!brain) {
      this.status.setText("no brain matched this vault; set one in the plugin settings");
      return;
    }
    this.status.setText(`loading graph of ${brain}...`);
    try {
      const s = this.plugin.settings;
      this.graph = await this.plugin.client.graph(brain, s.similarK, s.minScore, true);
      if (!this.graph.indexed) this.status.setText("brain is not indexed yet: run `mmdb brain load` or the reindex command");
      this.rebuild();
    } catch (e) {
      this.status.setText(`error: ${(e as Error).message}`);
    }
  }

  private rebuild(): void {
    if (!this.graph) return;
    const keep = new Map<string, N>();
    const old = new Map(this.nodes.map((n) => [n.id, n]));
    for (const g of this.graph.nodes) {
      if (this.hidden.has(g.layer)) continue;
      const prev = old.get(g.id);
      keep.set(g.id, prev ? Object.assign(prev, g) : { ...g });
    }
    this.nodes = [...keep.values()];
    this.links = this.graph.links
      .filter((l) => (l.kind === "similar" ? this.showSimilar : this.showWikilinks))
      .filter((l) => keep.has(l.source) && keep.has(l.target))
      .map((l) => ({ source: keep.get(l.source)!, target: keep.get(l.target)!, kind: l.kind, score: l.score }));
    const similarCount = this.links.filter((l) => l.kind === "similar").length;
    this.status.setText(`${this.nodes.length} notes, ${this.links.length - similarCount} wikilinks, ${similarCount} similar links`);

    this.sim?.stop();
    const { width, height } = this.size();
    this.sim = forceSimulation<N, L>(this.nodes)
      .force("link", forceLink<N, L>(this.links).id((d) => d.id).distance((l) => (l.kind === "similar" ? 40 : 70)).strength((l) => (l.kind === "similar" ? 0.6 * (l.score ?? 0.8) : 0.25)))
      .force("charge", forceManyBody().strength(-60))
      .force("center", forceCenter(width / 2, height / 2))
      .force("collide", forceCollide(6))
      .alpha(1)
      .on("tick", () => this.draw());
  }

  private size(): { width: number; height: number } {
    const dpr = window.devicePixelRatio || 1;
    return { width: this.canvas.width / dpr, height: this.canvas.height / dpr };
  }

  private radius(n: N): number {
    return 3 + Math.min(9, Math.sqrt(n.in + n.out));
  }

  private draw(): void {
    const ctx = this.ctx;
    const { width, height } = this.size();
    ctx.save();
    ctx.clearRect(0, 0, width, height);
    ctx.translate(this.transform.x, this.transform.y);
    ctx.scale(this.transform.k, this.transform.k);

    for (const l of this.links) {
      const s = l.source as N, t = l.target as N;
      if (s.x == null || t.x == null) continue;
      if (l.kind === "similar") {
        const a = Math.max(0.15, ((l.score ?? 0.8) - 0.7) * 2.5);
        ctx.strokeStyle = `rgba(255, 120, 60, ${a})`;
        ctx.lineWidth = 1.2;
      } else {
        ctx.strokeStyle = "rgba(160, 160, 160, 0.25)";
        ctx.lineWidth = 0.6;
      }
      ctx.beginPath();
      ctx.moveTo(s.x!, s.y!);
      ctx.lineTo(t.x!, t.y!);
      ctx.stroke();
    }

    for (const n of this.nodes) {
      if (n.x == null) continue;
      ctx.beginPath();
      ctx.arc(n.x, n.y!, this.radius(n), 0, Math.PI * 2);
      ctx.fillStyle = LAYER_COLOURS[n.layer] ?? "#c0c0c0";
      ctx.globalAlpha = n.has_thinking || !n.generated ? 1 : 0.6;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (n === this.hover) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    const labelled = this.transform.k > 1.6 ? this.nodes : this.hover ? [this.hover] : [];
    ctx.font = `${11 / this.transform.k}px sans-serif`;
    ctx.fillStyle = "rgba(230,230,230,0.95)";
    for (const n of labelled) {
      if (n.x == null) continue;
      ctx.fillText(n.name, n.x + this.radius(n) + 2, n.y! + 3 / this.transform.k);
    }
    ctx.restore();
  }

  private nodeAt(px: number, py: number): N | null {
    const [x, y] = this.transform.invert([px, py]);
    let best: N | null = null, bestD = Infinity;
    for (const n of this.nodes) {
      if (n.x == null) continue;
      const d = Math.hypot(n.x - x, n.y! - y);
      const r = this.radius(n) + 3 / this.transform.k;
      if (d < r && d < bestD) {
        best = n;
        bestD = d;
      }
    }
    return best;
  }

  private wire(): void {
    const canvas = this.canvas;
    const z = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.1, 8])
      .filter((e: Event) => !this.drag && (e.type !== "mousedown" || (e as MouseEvent).button === 0))
      .on("zoom", (e) => {
        this.transform = e.transform;
        this.draw();
      });
    select(canvas).call(z);

    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (this.drag) {
        const [x, y] = this.transform.invert([px, py]);
        this.drag.fx = x;
        this.drag.fy = y;
        this.sim?.alpha(0.3).restart();
        return;
      }
      const h = this.nodeAt(px, py);
      if (h !== this.hover) {
        this.hover = h;
        canvas.style.cursor = h ? "pointer" : "default";
        this.draw();
      }
    });
    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const r = canvas.getBoundingClientRect();
      const n = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (n) {
        this.drag = n;
        n.fx = n.x;
        n.fy = n.y;
        e.stopPropagation();
      }
    });
    const release = () => {
      if (this.drag) {
        this.drag.fx = null;
        this.drag.fy = null;
        this.drag = null;
      }
    };
    canvas.addEventListener("mouseup", release);
    canvas.addEventListener("mouseleave", release);
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const n = this.nodeAt(e.clientX - r.left, e.clientY - r.top);
      if (n) void this.plugin.openNote(n.id);
    });
  }
}
