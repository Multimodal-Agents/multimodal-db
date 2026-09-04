import { ItemView, TFile, WorkspaceLeaf } from "obsidian";
import type MmdbPlugin from "../main";

export const RELATED_VIEW = "mmdb-related";

/** Nearest notes to the one you have open, refreshed on every leaf change. */
export class RelatedView extends ItemView {
  private body!: HTMLElement;
  private title!: HTMLElement;
  private current: string | null = null;

  constructor(leaf: WorkspaceLeaf, private plugin: MmdbPlugin) {
    super(leaf);
  }

  getViewType(): string {
    return RELATED_VIEW;
  }

  getDisplayText(): string {
    return "Related notes";
  }

  getIcon(): string {
    return "git-branch";
  }

  async onOpen(): Promise<void> {
    const root = this.contentEl;
    root.empty();
    root.addClass("mmdb-view");
    this.title = root.createDiv({ cls: "mmdb-status" });
    this.body = root.createDiv({ cls: "mmdb-results" });
    this.registerEvent(this.app.workspace.on("active-leaf-change", () => void this.refresh()));
    this.registerEvent(this.app.workspace.on("file-open", () => void this.refresh()));
    await this.refresh();
  }

  async refresh(): Promise<void> {
    const file = this.app.workspace.getActiveFile();
    if (!(file instanceof TFile) || file.extension !== "md") return;
    const id = file.path.replace(/\.md$/, "");
    if (id === this.current) return;
    this.current = id;
    const brain = this.plugin.brain;
    if (!brain) {
      this.title.setText("no brain matched this vault");
      return;
    }
    this.title.setText(`related to ${file.basename}`);
    this.body.empty();
    try {
      const rows = await this.plugin.client.related(brain, id, 12);
      if (!rows.length) this.body.createDiv({ cls: "mmdb-empty", text: "nothing yet (is the brain loaded?)" });
      for (const r of rows) {
        const row = this.body.createDiv({ cls: "mmdb-hit" });
        const head = row.createDiv({ cls: "mmdb-hit-head" });
        head.createSpan({ cls: "mmdb-score", text: r.score.toFixed(3) });
        const link = head.createEl("a", { cls: "mmdb-link", text: r.title || r.id });
        link.addEventListener("click", (e) => {
          e.preventDefault();
          void this.plugin.openNote(r.id);
        });
        if (r.layer) head.createSpan({ cls: "mmdb-layer", text: r.layer });
        row.createDiv({ cls: "mmdb-hit-id", text: r.id });
      }
    } catch (e) {
      this.body.createDiv({ cls: "mmdb-empty", text: `error: ${(e as Error).message}` });
    }
  }
}
