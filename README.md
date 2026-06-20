# Indoor Map Processor

Server-side ingestion pipeline for an **Indoor Google Maps** platform. Upload a standard
2D floor plan image (PNG/JPEG) and the pipeline digitizes it into a navigable graph —
**nodes** = rooms/POIs, **edges** = walkable pathways — that an iOS client downloads for
GPS-free, turn-by-turn indoor navigation.

Map data is extracted by a **live Gemini 2.5 Flash vision call** against the real image.
No mock data, no hardcoded layouts, no sidecar JSON.

## Pipeline

```
maps/*.{png,jpg,jpeg}
  → load_image        cv2 → native width × height
  → analyze_floorplan Gemini 2.5 Flash, Pydantic structured output (response_schema)
  → build_graph       networkx; edge weight = Euclidean pixel distance
  → export_graph      output/building_graph.json   (lightweight iOS payload)
  + render_overlay    output/vector_map_overlay.png (graph drawn over the plan)
```

Each stage is an independent function in `process_maps.py`, so the pipeline can be demoed
stage-by-stage.

For a deep dive into every stage, data contract, coordinate system, and error-handling
logic, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Setup

Requires Python 3.11 and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
# Create and activate a virtualenv
python -m venv .venv

source .venv/Scripts/activate          # Git Bash on Windows
.\.venv\Scripts\Activate.ps1           # PowerShell on Windows
source .venv/bin/activate              # macOS / Linux

pip install -r requirements.txt
```

Set your Gemini API key — create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

The pipeline exits with a clear message if the key is missing.

## Running the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser. You'll see the test console where you
can upload a floor plan, inspect the AI-generated overlay, and test pathfinding — all
without needing the iOS app.

### Expose to the internet (required for iOS)

During development, use [ngrok](https://ngrok.com) to give the server a public HTTPS URL
so an iPhone can reach it over the venue WiFi:

```bash
ngrok http 8000
```

ngrok prints a URL like `https://abc123.ngrok.io`. Share that with the iOS team and paste
it into the "API base URL" field in the test console.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Browser test console |
| `POST` | `/buildings/upload` | Upload a floor plan image, run the full pipeline |
| `GET` | `/buildings/{id}/graph` | Full navigation graph JSON |
| `GET` | `/buildings/{id}/nodes` | All nodes (for destination search / autocomplete) |
| `GET` | `/buildings/{id}/route?from=X&to=Y` | Shortest path between two node IDs |
| `GET` | `/buildings/{id}/overlay` | QA overlay PNG |

Auto-generated interactive docs: **http://localhost:8000/docs**

### Example: upload a floor plan

```bash
curl -X POST http://localhost:8000/buildings/upload \
  -F "file=@maps/floor_plan.png"
```

Response:
```json
{
  "building_id": "3f2a1b...",
  "node_count": 18,
  "edge_count": 21,
  "image": { "width": 737, "height": 830 },
  "processing_time_s": 13.1,
  "gemini_time_s": 12.3
}
```

### Example: get a route

```bash
curl "http://localhost:8000/buildings/3f2a1b.../route?from=lobby&to=room_101"
```

Response:
```json
{
  "steps": [
    { "id": "lobby",              "name": "Lobby",              "x": 80,  "y": 80 },
    { "id": "hallway_junction_1", "name": "Hallway Junction",   "x": 180, "y": 80 },
    { "id": "room_101",           "name": "Room 101",           "x": 300, "y": 120 }
  ],
  "total_distance_px": 232.5,
  "step_count": 3
}
```

## Running the CLI pipeline (without the server)

Drop floor plan images into `./maps/` and run:

```bash
python process_maps.py
```

Results are written to `./output/`:
- **`building_graph.json`** — graph topology for the iOS client
- **`vector_map_overlay.png`** — graph drawn over the original floor plan

## Output format

`building_graph.json`:

```json
{
  "image": { "width": 600, "height": 400 },
  "nodes": [
    { "id": "a", "name": "Lobby",    "x": 80,  "y": 80 },
    { "id": "b", "name": "Room 101", "x": 300, "y": 120 }
  ],
  "edges": [
    { "source": "a", "target": "b", "weight": 223.61 }
  ]
}
```

Coordinates are pixels in the source image (origin top-left). `weight` is the Euclidean
pixel distance between connected nodes.

## Error handling

- **Missing API key** — exits cleanly before any API call with a clear message
- **API timeout / transient failure** — 60s request timeout with retry and exponential backoff
- **Corrupt or invalid image** — that file is skipped; remaining images still process
- **Malformed model output** — Pydantic validates the schema; edges referencing unknown nodes are dropped with a warning

## Project layout

```
api.py               # FastAPI server (5 endpoints + browser test console)
process_maps.py      # pipeline (schema + stage functions + CLI orchestrator)
static/index.html    # browser test console UI
docs/ARCHITECTURE.md # in-depth architecture and stage-by-stage logic
requirements.txt     # dependencies
.env                 # API key (create this, not committed)
maps/                # input floor plan images (you provide)
output/              # generated graph JSON + overlay PNG
output/buildings/    # per-building folders created by the API server
```

## Tech stack

FastAPI · uvicorn · OpenCV · Google GenAI SDK (Gemini 2.5 Flash) · Pydantic · networkx · matplotlib
