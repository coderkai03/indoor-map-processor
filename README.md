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
# Create / activate a virtualenv
python -m venv .venv
source .venv/Scripts/activate          # Git Bash on Windows
# .\.venv\Scripts\Activate.ps1         # PowerShell
# source .venv/bin/activate            # macOS / Linux

pip install -r requirements.txt

# Configure the API key
cp .env.example .env                   # then edit .env and set GEMINI_API_KEY
```

The key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment or a `.env`
file. The pipeline exits with a clear message if it is missing.

## Usage

1. Drop one or more floor plan images (PNG/JPEG) into `./maps/`.
2. Run the pipeline:

   ```bash
   python process_maps.py
   ```

3. Find the results in `./output/`:
   - **`building_graph.json`** — the graph topology for the iOS client.
   - **`vector_map_overlay.png`** — the graph drawn over the original floor plan.

All images in `./maps/` are processed in one run; the latest results overwrite `./output/`.

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
pixel distance between the connected nodes.

## Error handling

- **Missing API key** — exits cleanly before any API call.
- **API timeout / transient failure** — 60s request timeout, with retry and backoff.
- **Corrupt or invalid image** — that file is skipped; remaining images still process.
- **Malformed model output** — Pydantic validates the schema; edges referencing unknown
  nodes are dropped with a warning rather than crashing the run.

## Project layout

```
process_maps.py     # the pipeline (schema + stage functions + orchestrator)
docs/ARCHITECTURE.md # in-depth architecture and stage-by-stage logic
requirements.txt    # dependencies
.env.example        # API key template
maps/               # input floor plan images (you provide)
output/             # generated graph JSON + overlay PNG
```

## Tech stack

OpenCV · Google GenAI SDK (Gemini 2.5 Flash) · Pydantic · networkx · matplotlib
