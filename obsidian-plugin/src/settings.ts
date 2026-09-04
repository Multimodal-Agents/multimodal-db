import { App, PluginSettingTab, Setting } from "obsidian";
import type MmdbPlugin from "./main";

export interface MmdbSettings {
  serverUrl: string;
  brain: string;
  similarK: number;
  minScore: number;
  hideLayers: string[];
}

export const DEFAULT_SETTINGS: MmdbSettings = {
  serverUrl: "http://127.0.0.1:8765",
  brain: "",
  similarK: 5,
  minScore: 0.8,
  hideLayers: ["Archive"],
};

export class MmdbSettingTab extends PluginSettingTab {
  constructor(app: App, private plugin: MmdbPlugin) {
    super(app, plugin);
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
