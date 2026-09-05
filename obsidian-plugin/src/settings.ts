import { App, PluginSettingTab, Setting } from "obsidian";
import type MmdbPlugin from "./main";

export interface MmdbSettings {
  serverUrl: string;
  serveCommand: string;
  brain: string;
  similarK: number;
  minScore: number;
  hideLayers: string[];
}

export const DEFAULT_SETTINGS: MmdbSettings = {
  serverUrl: "http://127.0.0.1:8765",
  serveCommand: "mmdb serve",
  brain: "",
  similarK: 5,
  minScore: 0.8,
  hideLayers: ["Archive"],
};

export class MmdbSettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: MmdbPlugin) {
    super(app, plugin);
  }

  private async refreshStatus(status: Setting): Promise<void> {
    try {
      const h = await this.plugin.client.health();
      await this.plugin.detectBrain(true);
      const mine = h.brains.find((b) => b.name === this.plugin.brain);
      status.setDesc(
        `online, multimodal-db ${h.version}. ` +
          (mine ? `This vault is brain "${mine.name}" (${mine.docs} notes indexed).` : "This vault is not a registered brain.")
      );
    } catch (e) {
      status.setDesc(`offline at ${this.plugin.settings.serverUrl} (${(e as Error).message}). Launch it here or run it yourself.`);
    }
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("Server")
      .setDesc("Where `mmdb serve` is listening.")
      .addText((t) =>
        t.setValue(this.plugin.settings.serverUrl).onChange(async (v) => {
          this.plugin.settings.serverUrl = v.trim();
          this.plugin.client.setBase(this.plugin.settings.serverUrl);
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Server launch command")
      .setDesc("What the Launch button runs: a script path (.cmd, .sh) or a command line like `mmdb serve`. Opens in its own window; close that window to stop the server.")
      .addText((t) =>
        t.setValue(this.plugin.settings.serveCommand).onChange(async (v) => {
          this.plugin.settings.serveCommand = v.trim();
          await this.plugin.saveSettings();
        })
      );

    const status = new Setting(containerEl).setName("Server status").setDesc("checking...");
    status.addButton((b) =>
      b.setButtonText("Launch server").setCta().onClick(async () => {
        b.setDisabled(true);
        await this.plugin.launchServer((msg) => status.setDesc(msg));
        b.setDisabled(false);
        await this.refreshStatus(status);
      })
    );
    status.addButton((b) => b.setButtonText("Check").onClick(() => void this.refreshStatus(status)));
    void this.refreshStatus(status);

    new Setting(containerEl)
      .setName("Brain")
      .setDesc("Registered name of this vault in mmdb.toml. Leave empty to match the vault path automatically.")
      .addText((t) =>
        t.setValue(this.plugin.settings.brain).onChange(async (v) => {
          this.plugin.settings.brain = v.trim();
          await this.plugin.saveSettings();
          await this.plugin.detectBrain();
        })
      );

    new Setting(containerEl)
      .setName("Similar links per note")
      .setDesc("How many nearest neighbours each note contributes to the graph.")
      .addSlider((s) =>
        s.setLimits(1, 15, 1).setValue(this.plugin.settings.similarK).setDynamicTooltip().onChange(async (v) => {
          this.plugin.settings.similarK = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Minimum similarity")
      .setDesc("Cosine score below which a similar link is dropped.")
      .addSlider((s) =>
        s.setLimits(0.5, 0.99, 0.01).setValue(this.plugin.settings.minScore).setDynamicTooltip().onChange(async (v) => {
          this.plugin.settings.minScore = v;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Hidden layers")
      .setDesc("Top-level folders left out of the graph by default, comma separated.")
      .addText((t) =>
        t.setValue(this.plugin.settings.hideLayers.join(", ")).onChange(async (v) => {
          this.plugin.settings.hideLayers = v.split(",").map((s) => s.trim()).filter(Boolean);
          await this.plugin.saveSettings();
        })
      );
  }
}
