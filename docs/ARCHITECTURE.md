# Indoor Map Processor — Architecture

This document describes the end-to-end design of the map and navigation-graph generation
pipeline: what each stage does, why it exists, how data flows between stages, and the
logic applied at every step.

For setup and usage, see [README.md](../README.md).

---

## 1. System purpose

The Indoor Map Processor is a **server-side ingestion pipeline** for an indoor navigation
platform. It accepts a standard 2D floor plan image (PNG or JPEG) and produces:

1. **`building_graph.json`** — a weighted graph (nodes + edges) the iOS client downloads
   for GPS-free shortest-path routing and turn-by-turn navigation.
2. **`vector_map_overlay.png`** — a human-readable QA artifact showing the extracted graph
   drawn on top of the original floor plan.

The core design bet is that a **live vision-language model** (Gemini 2.5 Flash) can
semantically interpret a floor plan well enough to emit a usable navigation topology,
without hand-authored map data, OCR pipelines, or classical CV room segmentation.

---

## 2. High-level architecture

The pipeline is **strictly unidirectional**. Each stage consumes the previous stage's
output and never reaches back. Stages are exposed as independent functions in
`process_maps.py` so they can be demoed or tested in isolation during development.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           process_maps.py                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ./maps/*.{png,jpg,jpeg}                                                    │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐   OpenCV (cv2)                                            │
│  │  Stage 1     │   find_maps → load_image                                  │
│  │  Ingestion   │   Output: BGR image matrix, width, height               │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐   Google GenAI SDK (Gemini 2.5 Flash)                     │
│  │  Stage 2     │   get_client → analyze_floorplan                          │
│  │  Vision LLM  │   Output: FloorPlan (Pydantic-validated nodes + edges)    │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐   networkx + math.hypot                                   │
│  │  Stage 3     │   build_graph → export_graph                              │
│  │  Compilation │   Output: building_graph.json                             │
│  └──────┬───────┘                                                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐   matplotlib + OpenCV                                     │
│  │  Stage 4     │   render_overlay                                          │
│  │  Visualization│  Output: vector_map_overlay.png                          │
│  └──────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Orchestration entry points

| Function | Role |
|----------|------|
| `main()` | Discovers images, constructs the Gemini client once, processes every map, returns exit code. |
| `process_image(client, path)` | Runs stages 1–4 for a single file. Returns `True` on success, `False` if ingestion or vision fails. |
| Individual stage functions | Callable directly for stage-by-stage demos or unit tests. |

**Multi-image behavior:** `find_maps()` returns all supported images in `./maps/`, sorted
alphabetically. Each image is processed sequentially. Output paths are fixed
(`./output/building_graph.json`, `./output/vector_map_overlay.png`), so **the last
successfully processed image overwrites prior output**. This is intentional for hackathon
demos where one floor plan is processed at a time.

---

## 3. Coordinate system

All spatial data lives in **native image pixel space** — the width and height OpenCV reads
from the file, with no scaling, normalization, or crop applied anywhere in the pipeline.

| Property | Convention |
|----------|------------|
| Origin | Top-left corner `(0, 0)` |
| X axis | Increases to the right |
| Y axis | Increases downward |
| Valid range | `x ∈ [0, width)`, `y ∈ [0, height)` |
| Node placement | Visual center of a room, POI, or hallway junction |

The vision model is told explicitly (system prompt + user prompt with dimensions) to emit
coordinates in this space. The iOS client uses the same coordinate system to overlay the
user's position and draw routes on the floor plan bitmap.

**Why pixel space:** The floor plan image *is* the map. Keeping coordinates in raw pixels
means the client can place nodes and route polylines directly on the image without an
affine transform layer.

---

## 4. Data contracts

### 4.1 Pydantic schema (LLM ↔ compiler)

The `FloorPlan`, `Node`, and `Edge` models in `process_maps.py` are the **single source
of truth** for structured map data. The same schema is bound to Gemini's
`response_schema` and consumed by `build_graph()`.

```python
class Node(BaseModel):
    id: str      # Unique string identifier
    name: str    # Human-readable label (e.g. "Room 101", "Lobby")
    x: int       # Pixel X of visual center
    y: int       # Pixel Y of visual center

class Edge(BaseModel):
    source: str  # Must reference an existing node id
    target: str  # Must reference an existing node id

class FloorPlan(BaseModel):
    nodes: list[Node]
    edges: list[Edge]
```

**Design decisions:**

- **Edge weights are not in the schema.** The model only declares topology (which nodes
  connect). Weights are derived deterministically in `build_graph()` so routing math is
  reproducible and not subject to model hallucination.
- **Integer coordinates.** Pydantic enforces `int` for `x`/`y`, which matches pixel
  indexing and keeps JSON payloads compact.
- **Undirected graph implied.** Edges have `source` and `target` but `build_graph()`
  creates a `networkx.Graph` (undirected). An edge `A → B` means walkable in both
  directions.

### 4.2 Exported JSON (compiler → iOS client)

`export_graph()` writes a client-facing payload:

```json
{
  "image": { "width": 1052, "height": 1314 },
  "nodes": [
    { "id": "lobby", "name": "Lobby", "x": 80, "y": 80 }
  ],
  "edges": [
    { "source": "lobby", "target": "room_101", "weight": 223.61 }
  ]
}
```

| Field | Source | Purpose |
|-------|--------|---------|
| `image.width`, `image.height` | OpenCV at ingestion | Client knows the coordinate bounds and can scale the floor plan image. |
| `nodes[].id` | Gemini (validated) | Stable key for routing and UI. |
| `nodes[].name` | Gemini | Turn-by-turn labels and search. |
| `nodes[].x`, `nodes[].y` | Gemini | Map overlay and distance context. |
| `edges[].source`, `edges[].target` | Gemini (sanitized) | Graph adjacency. |
| `edges[].weight` | `math.hypot` in compiler | Shortest-path cost in pixel units. |

---

## 5. Stage 0 — Discovery and bootstrap

Before any image work, `main()` performs environment setup.

### 5.1 `find_maps(maps_dir="./maps")`

1. If `./maps/` does not exist → log message, return empty list, exit 0 (not an error).
2. Iterate directory entries; keep files whose extension (case-insensitive) is `.png`,
   `.jpg`, or `.jpeg`.
3. Sort paths alphabetically for deterministic processing order.
4. If directory exists but has no supported files → log message, return empty list.

No recursion into subdirectories. Only top-level files in `./maps/` are considered.

### 5.2 `get_client()`

Constructs a `google.genai.Client` with API key from:

1. `GEMINI_API_KEY` environment variable, or
2. `GOOGLE_API_KEY` as fallback.

If neither is set → raises `EnvironmentError` with an actionable message. `main()`
catches this and exits with code 1 **before any network call**.

Optional: `python-dotenv` loads a `.env` file at import time if installed.

---

## 6. Stage 1 — Ingestion (OpenCV)

**Functions:** `load_image(path)`

**Library:** OpenCV (`cv2`)

### 6.1 What happens

1. `cv2.imread(str(path))` decodes the image into a BGR `numpy` array.
2. If decode fails (`None`) → raise `ValueError` (corrupt file, wrong format, missing file).
3. Read `height, width = image.shape[:2]`.
4. If `width <= 0` or `height <= 0` → raise `ValueError`.
5. Return `(image_bgr, width, height)`.

### 6.2 Why OpenCV here

OpenCV is used for **reliable binary image I/O and dimension probing**, not for computer
vision analysis. The pipeline does not threshold walls, detect contours, or run OCR.

The BGR matrix is retained for stage 4 (overlay rendering). For stage 2 (Gemini), the
orchestrator reads **raw file bytes** via `path.read_bytes()` rather than re-encoding the
OpenCV matrix. This avoids accidental quality loss from re-compression and preserves the
original MIME type.

### 6.3 Failure mode in orchestration

`process_image()` wraps `load_image()` in try/except. On `ValueError`, the file is
**skipped** with a log line; other images in the batch still process.

---

## 7. Stage 2 — Vision analysis (Gemini)

**Functions:** `analyze_floorplan(client, image_bytes, mime_type, width, height)`

**Library:** Google GenAI SDK (`google-genai`), model `gemini-2.5-flash`

This stage is the **semantic extraction engine**. It answers: *What places exist on this
floor plan, where are they in pixel space, and which places are directly walkable from
each other?*

### 7.1 Request construction

| Component | Value |
|-----------|-------|
| Model | `gemini-2.5-flash` |
| System instruction | `SYSTEM_PROMPT` (see §7.2) |
| User content | Text prompt with `width` × `height` + image bytes as a `Part` |
| Response format | `application/json` with `response_schema=FloorPlan` |
| Timeout | 60,000 ms per HTTP request |

MIME type is inferred from file extension: `.png` → `image/png`, `.jpg`/`.jpeg` →
`image/jpeg`.

The SDK base64-encodes image bytes on the wire; the caller passes raw bytes.

### 7.2 System prompt logic

The system prompt instructs the model to behave as a floor-plan digitizer:

1. **Identify** every room, POI, and hallway junction.
2. **Place nodes** at the **visual center** of each identified region, with concise names.
3. **Emit edges** only for pairs that a person can walk between directly (through a door
   or along a contiguous hallway segment). Explicitly: do **not** connect rooms
   separated by a wall with no doorway.
4. **Coordinate rules:** pixel space, top-left origin, bounds within image dimensions.
5. **Referential integrity:** unique node ids; every edge endpoint must reference a
   declared node id.

The user prompt reinforces dimensions: *"This floor plan image is {width} pixels wide and
{height} pixels tall."*

Together, these prompts anchor the model to the same coordinate system OpenCV measured.

### 7.3 Response parsing

1. Call `client.models.generate_content(...)`.
2. Prefer `response.parsed` — the SDK auto-deserializes into `FloorPlan` when
   `response_schema` is set.
3. Fallback: if `parsed` is not a `FloorPlan`, run `FloorPlan.model_validate_json(response.text)`.
4. Pydantic validates types, required fields, and schema shape. Invalid JSON or schema
   violations surface as validation errors (not silently accepted).

### 7.4 Retry and backoff

Transient failures are retried up to `MAX_RETRIES` (2) total attempts:

- Caught exceptions: `genai_errors.APIError`, `TimeoutError`, `ConnectionError`
- Backoff: `2 ** attempt` seconds (2s after attempt 1, 4s after attempt 2)
- After all attempts fail → raise `RuntimeError`; orchestrator skips the file

Non-transient failures (bad API key, invalid request schema) are not specially retried;
they fail on the first attempt.

### 7.5 What the model decides vs. what code decides

| Decision | Owner |
|----------|-------|
| Which rooms/POIs/junctions exist | Gemini |
| Node names and ids | Gemini |
| Node pixel coordinates | Gemini |
| Which pairs are walkably connected | Gemini |
| Edge travel cost (weight) | **Code** (`build_graph`) |
| Dropping invalid edges | **Code** |
| Resolving duplicate node ids | **Code** |

### 7.6 Typical graph topology from the model

On complex floor plans (e.g. a university building), the model often produces:

- **Room/POI nodes** — labeled spaces like "Restrooms", "Elevator", "Cal Student Store"
- **Junction nodes** — synthetic hallway waypoints (e.g. "Hallway Center Junction") that
  discretize long corridors into a routable mesh

This junction pattern is emergent from the prompt's "hallway junction" instruction. It
allows shortest-path algorithms to route through corridors rather than only room-to-room
chords.

---

## 8. Stage 3 — Graph compilation (networkx)

**Functions:** `build_graph(floorplan)`, `export_graph(graph, width, height)`

**Library:** networkx

This stage transforms validated semantic output into a **formal weighted graph** suitable
for standard shortest-path algorithms (Dijkstra, A*, etc.) on the client.

### 8.1 `build_graph(floorplan)` — node insertion

```
graph = empty undirected Graph
nodes_by_id = {}

for each node in floorplan.nodes:
    if node.id already in nodes_by_id:
        log warning, skip duplicate (keep first occurrence)
    else:
        nodes_by_id[node.id] = node
        graph.add_node(id, name=..., x=..., y=...)
```

**Duplicate id policy:** First wins. Later duplicates are dropped with a `[compile]`
warning. This prevents networkx from silently overwriting node attributes.

### 8.2 `build_graph(floorplan)` — edge insertion and weighting

For each edge in `floorplan.edges`:

1. Resolve `source` and `target` against `nodes_by_id`.
2. If either endpoint is missing → log `[compile] Dropping dangling edge`, skip.
3. Compute weight:

   ```
   weight = round(sqrt((x_tgt - x_src)² + (y_tgt - y_src)²), 2)
   ```

   Implemented as `math.hypot(tgt.x - src.x, tgt.y - src.y)`.

4. `graph.add_edge(source, target, weight=weight)`.

**Why Euclidean distance in pixels:**

- It is deterministic given node coordinates.
- It approximates walking distance when edges represent direct walkable segments.
- It is cheap to compute and needs no calibration constants.

**Caveat:** Euclidean weight is a straight-line distance between node centers. If an edge
 represents a path that bends through a hallway, the weight **underestimates** path length
 unless junction nodes break the corridor into smaller segments. The junction-node pattern
 from the vision stage mitigates this.

**Parallel edges:** networkx `Graph` collapses duplicate `(u, v)` pairs; the last added
 weight would win. The model is instructed to emit each connection once.

### 8.3 `export_graph(graph, width, height)`

Serializes the in-memory graph to JSON:

1. Build payload dict with `image`, `nodes`, `edges` arrays.
2. Node iteration uses `graph.nodes(data=True)` — order follows networkx internal ordering
   (not guaranteed stable across runs; clients should index by `id`).
3. Edge iteration uses `graph.edges(data=True)`.
4. Create `./output/` if missing.
5. Write pretty-printed JSON (`indent=2`, UTF-8).

The export is a **lossless projection** of the compiled graph plus image dimensions. No
layout algorithm runs at export time.

---

## 9. Stage 4 — Visualization (matplotlib + OpenCV)

**Function:** `render_overlay(image_bgr, graph, out_path)`

**Libraries:** matplotlib (Agg backend), OpenCV

This stage is for **human verification**, not client consumption. It answers: *Does the
extracted graph align with what a human sees on the floor plan?*

### 9.1 Rendering pipeline

1. Convert BGR → RGB with `cv2.cvtColor` (matplotlib expects RGB).
2. Create figure sized to image dimensions: `figsize=(width/100, height/100)`, `dpi=100`
   → output matches native pixel dimensions.
3. `ax.imshow(image_rgb)` — floor plan as background.
4. **Edges:** for each `(u, v)`, draw a cyan line (`#00E5FF`, 2px) between node
   coordinates.
5. **Nodes:** red circle markers (`#FF1744`, size 10) at each `(x, y)`.
6. **Labels:** white bold text with semi-transparent black rounded bbox, offset from marker.
7. Set axis limits: `xlim(0, width)`, `ylim(height, 0)` — **inverted Y** so matplotlib's
   display matches image coordinates (origin top-left).
8. Hide axes, save PNG with `bbox_inches="tight", pad_inches=0`, close figure.

### 9.2 Headless server support

`matplotlib.use("Agg")` is set before importing `pyplot`, enabling rendering on servers
without a display.

---

## 10. End-to-end flow for one image

The following traces `process_image()` for a file like `maps/ucb-mlk-1st-floor.png`:

```
1. load_image
   → cv2.imread → 1052×1314 BGR matrix
   → log: [ingest] Loaded 'ucb-mlk-1st-floor.png' (1052x1314 px)

2. Read raw bytes + MIME type from path

3. analyze_floorplan
   → POST to Gemini with image + schema
   → FloorPlan with ~20 nodes, ~20 edges (example run)
   → log: [vision] Received 20 nodes, 20 edges

4. build_graph
   → nx.Graph with node attrs {name, x, y}
   → edges weighted by hypot
   → dangling/duplicate handling
   → log: [compile] Graph compiled: 20 nodes, 20 edges

5. export_graph
   → ./output/building_graph.json
   → log: [export] Wrote graph topology -> ...

6. render_overlay
   → ./output/vector_map_overlay.png
   → log: [render] Wrote overlay -> ...
```

---

## 11. Error handling matrix

| Condition | Behavior | Exit impact |
|-----------|----------|-------------|
| `./maps/` missing or empty | Log, exit 0 | No work done |
| Missing API key | Log, exit 1 | Before any image |
| Corrupt / unreadable image | Skip file, log `[error]` | Other files continue |
| Invalid dimensions (0×0) | Skip file | Other files continue |
| Vision API timeout / transient error | Retry with backoff, then skip file | Other files continue |
| Pydantic validation failure on response | Propagates as exception in analyze path | File skipped if uncaught |
| Duplicate node id from model | Keep first, warn | Graph still built |
| Dangling edge (unknown node ref) | Drop edge, warn | Graph still built |
| All files fail | exit 1 | |
| At least one success | exit 0 | |

The philosophy is **per-file isolation**: one bad floor plan or one failed API call must
not abort the entire batch.

---

## 12. Configuration reference

All tunables live as module-level constants in `process_maps.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `MAPS_DIR` | `./maps` | Input directory |
| `OUTPUT_DIR` | `./output` | Output directory |
| `GRAPH_JSON_PATH` | `./output/building_graph.json` | Client payload path |
| `OVERLAY_PATH` | `./output/vector_map_overlay.png` | QA visualization path |
| `MODEL_NAME` | `gemini-2.5-flash` | Vision model id |
| `SUPPORTED_EXTENSIONS` | `.png`, `.jpg`, `.jpeg` | Allowed inputs |
| `API_TIMEOUT_MS` | `60000` | HTTP timeout per vision request |
| `MAX_RETRIES` | `2` | Vision API attempt count |

---

## 13. Hard constraints (product invariants)

These are non-negotiable design rules enforced by architecture and code review:

1. **No mock map data.** Every node and edge must originate from a live Gemini call against
   the uploaded image.
2. **No hardcoded layout arrays or sidecar JSON** as map input.
3. **Real pixel coordinates only** — no normalization to 0–1 or arbitrary map units.
4. **Fail fast on missing API key** — never call the API without credentials.
5. **Stage independence** — each stage function can run given only its declared inputs.

---

## 14. Client consumption model (iOS)

The iOS app is expected to:

1. Download `building_graph.json` and the floor plan image (same pixel dimensions as
   `image.width` × `image.height`).
2. Build an adjacency structure from `edges` (undirected: each edge connects both ways).
3. Run shortest-path (e.g. Dijkstra) using `weight` as edge cost.
4. Render the route as a polyline through node coordinates overlaid on the floor plan
   bitmap.
5. Use `nodes[].name` for turn-by-turn instructions ("Head toward Restrooms", etc.).

The server pipeline **does not** compute routes, geofence rooms, or track user position.
It only produces the static topology.

---

## 15. Limitations and design tradeoffs

| Topic | Limitation | Mitigation / future work |
|-------|------------|--------------------------|
| Model accuracy | Gemini may miss rooms, misplace nodes, or hallucinate edges | QA via `vector_map_overlay.png`; prompt tuning; human review |
| Straight-line weights | Edge weight ≠ actual walking distance along walls | Junction nodes in hallways; future: polyline waypoints per edge |
| Single output slot | Last processed image overwrites `./output/` | Per-image output paths keyed by filename |
| 2D only | No multi-floor / vertical connectivity | Separate graphs per floor; stairs as annotated POIs only |
| No scale metadata | Pixel distance ≠ meters | Client calibration or floor plan scale bar parsing |
| Undirected edges | No one-way doors or escalator direction | Extend schema with directed graph if needed |
| Batch ordering | Alphabetical, not user-controlled | CLI flag for single-file selection |

---

## 16. Dependency roles summary

| Package | Stage(s) | Responsibility |
|---------|----------|----------------|
| **opencv-python** | 1, 4 | Image decode, dimension probe, color space conversion |
| **google-genai** | 2 | Gemini API client, structured JSON generation |
| **pydantic** | 2, 3 | Schema validation for LLM output |
| **networkx** | 3 | In-memory graph, compilation, export iteration |
| **matplotlib** | 4 | Overlay rendering |
| **numpy** | (transitive) | Array backing for OpenCV/matplotlib |
| **python-dotenv** | 0 | Optional `.env` loading for API key |

---

## 17. Source file map

```
process_maps.py
├── Configuration (paths, model, prompts, timeouts)
├── Schema (Node, Edge, FloorPlan)
├── Stage 1: find_maps, load_image
├── Stage 2: get_client, analyze_floorplan
├── Stage 3: build_graph, export_graph
├── Stage 4: render_overlay
└── Orchestration: process_image, main
```

All pipeline logic currently lives in this single module by design — minimal surface area
for a hackathon demo with clear stage boundaries.
