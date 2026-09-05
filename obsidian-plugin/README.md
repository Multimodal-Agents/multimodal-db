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
| graph | five modes over one canvas, see below | main pane, ribbon icon |

The graph reads the vault's own `.obsidian/graph.json`, so it starts with the same colour
groups, saved filter and forces as Obsidian's graph. On top of that:

| mode | what you see |
|---|---|
| Global | every note; wikilinks grey, similarity links orange with the score as opacity |
| Local | the note you are reading and its neighbourhood, depth 1 to 4, by both edge kinds; double-click any node in another mode to jump here |
| Semantic map | position is meaning: a t-SNE of the note embeddings, so notes that say the same thing sit together even with no link between them |
| Clusters | one node per `cluster/X`, sized by members, lines weighted by how many links cross between two clusters; click opens the cluster note |
| Tags | tag co-occurrence, sized by how many notes carry the tag; click runs an Obsidian search for it |

The filter box takes Obsidian's syntax (`tag:#state/open -path:Views`, `OR`). Colour by vault
groups, layer, state or age. Hover highlights a node's neighbours and fades the rest. The
"forces" button exposes repel, link distance and centring. `Archive/` is hidden by default
(settings); the minimum similarity and neighbour count are settings too.

## Develop

`npm run dev` watches and rebuilds `main.js`. Reload the plugin in Obsidian (or use the Hot
Reload community plugin) to pick it up.
