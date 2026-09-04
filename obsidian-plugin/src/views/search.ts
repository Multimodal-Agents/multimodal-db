import { ItemView, WorkspaceLeaf, setIcon } from "obsidian";
import type MmdbPlugin from "../main";
import type { Hit } from "../api";

export const SEARCH_VIEW = "mmdb-search";

export class SearchView extends ItemView {
  private input!: HTMLInputElement;
  private results!: HTMLElement;
  private status!: HTMLElement;

  constructor(leaf: WorkspaceLeaf, private plugin: MmdbPlugin) {
    super(leaf);
  }

  getViewType(): string {
    return SEARCH_VIEW;
  }

  getDisplayText(): string {
    return "Multimodal DB search";
  }

  getIcon(): string {
    return "search";
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("mmdb-view");

    const bar = root.createDiv({ cls: "mmdb-bar" });
    this.input = bar.createEl("input", { type: "text", placeholder: "search this brain semantically" });
    this.input.addClass("mmdb-input");
    const go = bar.createEl("button", { cls: "mmdb-button" });
    setIcon(go, "search");
    this.status = root.createDiv({ cls: "mmdb-status" });
    this.results = root.createDiv({ cls: "mmdb-results" });

    const run = () => void this.search(this.input.value.trim());
    go.addEventListener("click", run);
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") run();
    });
  }

  focusInput(): void {
    this.input?.focus();
  }

  async search(q: string): Promise<void> {
    if (!q) return;
    const brain = this.plugin.brain;
    if (!brain) {
      this.status.setText("no brain matched this vault; set one in the plugin settings");
      return;
    }
    this.status.setText("searching...");
    this.results.empty();
    try {
      const hits = await this.plugin.client.search(brain, q, 15);
      this.status.setText(hits.length ? `${hits.length} hits in ${brain}` : "no hits");
      for (const h of hits) this.renderHit(h);
    } catch (e) {
      this.status.setText(`error: ${(e as Error).message}`);
    }
  }

  private renderHit(h: Hit): void {
    const row = this.results.createDiv({ cls: "mmdb-hit" });
    const head = row.createDiv({ cls: "mmdb-hit-head" });
    head.createSpan({ cls: "mmdb-score", text: h.score.toFixed(3) });
    const link = head.createEl("a", { cls: "mmdb-link", text: h.title || h.id });
    link.addEventListener("click", (e) => {
      e.preventDefault();
      void this.plugin.openNote(h.id);
    });
    if (h.layer) head.createSpan({ cls: "mmdb-layer", text: h.layer });
    row.createDiv({ cls: "mmdb-hit-id", text: h.id });
    row.createDiv({ cls: "mmdb-snippet", text: h.text.replace(/\s+/g, " ").slice(0, 260) });
  }
}
