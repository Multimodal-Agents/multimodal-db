import { FileSystemAdapter, Notice, Plugin, WorkspaceLeaf } from "obsidian";
import { MmdbClient } from "./api";
import { DEFAULT_SETTINGS, MmdbSettings, MmdbSettingTab } from "./settings";
import { GRAPH_VIEW, GraphView } from "./views/graph";
import { RELATED_VIEW, RelatedView } from "./views/related";
import { SEARCH_VIEW, SearchView } from "./views/search";

/** Thin client for a local `mmdb serve`. The vault stays the source of truth; this plugin
 *  only reads from the server and opens notes. Nothing here writes to the vault. */
export default class MmdbPlugin extends Plugin {
  settings: MmdbSettings = DEFAULT_SETTINGS;
  client!: MmdbClient;
  brain: string | null = null;
  private statusEl!: HTMLElement;

  async onload(): Promise<void> {
    await this.loadSettings();
    this.client = new MmdbClient(this.settings.serverUrl);

    this.registerView(SEARCH_VIEW, (leaf) => new SearchView(leaf, this));
    this.registerView(RELATED_VIEW, (leaf) => new RelatedView(leaf, this));
    this.registerView(GRAPH_VIEW, (leaf) => new GraphView(leaf, this));

    this.addRibbonIcon("git-fork", "Multimodal DB graph", () => void this.activate(GRAPH_VIEW, "main"));
    this.addCommand({ id: "open-search", name: "Open semantic search", callback: () => void this.activate(SEARCH_VIEW, "right") });
    this.addCommand({ id: "open-related", name: "Open related notes", callback: () => void this.activate(RELATED_VIEW, "right") });
    this.addCommand({ id: "open-graph", name: "Open similarity graph", callback: () => void this.activate(GRAPH_VIEW, "main") });
    this.addCommand({ id: "reindex", name: "Reindex this brain in multimodal-db", callback: () => void this.reindex() });
    this.addCommand({ id: "reconnect", name: "Reconnect to multimodal-db", callback: () => void this.detectBrain() });

    this.addSettingTab(new MmdbSettingTab(this.app, this));
    this.statusEl = this.addStatusBarItem();
    this.app.workspace.onLayoutReady(() => void this.detectBrain());
  }

  onunload(): void {
    /* views are detached by Obsidian */
  }

  async loadSettings(): Promise<void> {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  /** Which registered brain is this vault? The setting wins; else match the vault path. */
  async detectBrain(): Promise<void> {
    try {
      const health = await this.client.health();
      const mine = this.vaultPath();
      const norm = (p: string) => p.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
      let match = health.brains.find((b) => b.name === this.settings.brain);
      if (!match && mine) match = health.brains.find((b) => norm(b.path) === norm(mine));
      this.brain = match?.name ?? null;
      this.statusEl.setText(this.brain ? `mmdb: ${this.brain} (${match!.docs} indexed)` : "mmdb: vault not registered");
      if (!this.brain) new Notice("multimodal-db: this vault is not a registered brain. Add it under [brains] in mmdb.toml or name it in the plugin settings.");
    } catch (e) {
      this.brain = null;
      this.statusEl.setText("mmdb: offline");
      new Notice(`multimodal-db: cannot reach ${this.settings.serverUrl} (${(e as Error).message}). Is \`mmdb serve\` running?`);
    }
  }

  vaultPath(): string | null {
    const adapter = this.app.vault.adapter;
    return adapter instanceof FileSystemAdapter ? adapter.getBasePath() : null;
  }

  async openNote(id: string): Promise<void> {
    await this.app.workspace.openLinkText(id, "", false);
  }

  async reindex(): Promise<void> {
    if (!this.brain) {
      new Notice("multimodal-db: no brain matched this vault");
      return;
    }
    new Notice(`multimodal-db: reindexing ${this.brain}...`);
    try {
      const r = await this.client.load(this.brain);
      const summary = Object.entries(r.counts).map(([k, v]) => `${v} ${k}`).join(", ");
      new Notice(`multimodal-db: ${summary || "nothing to do"} (${r.docs} notes, ${r.vectors} vectors)`);
      this.statusEl.setText(`mmdb: ${this.brain} (${r.docs} indexed)`);
    } catch (e) {
      new Notice(`multimodal-db: reindex failed: ${(e as Error).message}`);
    }
  }

  private async activate(type: string, side: "right" | "main"): Promise<void> {
    const existing = this.app.workspace.getLeavesOfType(type);
    let leaf: WorkspaceLeaf | null = existing[0] ?? null;
    if (!leaf) {
      leaf = side === "right" ? this.app.workspace.getRightLeaf(false) : this.app.workspace.getLeaf("tab");
      if (!leaf) return;
      await leaf.setViewState({ type, active: true });
    }
    this.app.workspace.revealLeaf(leaf);
    if (type === SEARCH_VIEW) (leaf.view as SearchView).focusInput();
  }
}
