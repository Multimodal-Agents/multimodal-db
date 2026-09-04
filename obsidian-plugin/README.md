# Multimodal DB for Obsidian

A thin client for a local `mmdb serve`. Three views: semantic search over this vault, the
notes nearest to the one you have open, and a force graph with two edge kinds, Obsidian's
wikilinks and multimodal-db's similarity links. The vault stays the source of truth; the
plugin reads from the server and opens notes. It never writes into the vault.

## Run

1. Register the vault as a brain in `mmdb.toml` and index it once:

   ```bash
   mmdb brain load amd
   mmdb serve
   ```

2. Build and copy the plugin into the vault:

   ```bash
   cd obsidian-plugin
   npm install
   npm run build
   mkdir -p "<vault>/.obsidian/plugins/multimodal-db"
   cp main.js manifest.json styles.css "<vault>/.obsidian/plugins/multimodal-db/"
   ```

3. Enable it under Settings, Community plugins. It matches the vault to a registered brain by
   path; the status bar says which one. Commands: open semantic search, open related notes,
   open similarity graph, reindex this brain, reconnect.

## Views

| view | what | where |
|---|---|---|
| search | query the brain semantically, click a hit to open it | right sidebar |
| related | nearest notes to the active note by vector similarity, no query needed | right sidebar |
| graph | every note as a node coloured by layer, wikilinks grey, similarity links orange with score as opacity; drag, zoom, click to open; toggle layers and edge kinds | main pane, ribbon icon |

The graph hides `Archive/` by default (settings). Similar-link count and the minimum score are
settings too, and they map straight onto `GET /api/brains/<name>/graph?k=&min=`.

## Develop

`npm run dev` watches and rebuilds `main.js`. Reload the plugin in Obsidian (or use the Hot
Reload community plugin) to pick it up.
