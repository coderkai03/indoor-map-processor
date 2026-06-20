# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Greenfield. The repo currently contains only a `.venv` (Python 3.11.5) with no project
dependencies installed and no source code. The architecture below is the agreed build
spec — treat it as the contract to implement, not as existing code to read.

## What this is

Server-side ingestion pipeline for an "Indoor Google Maps" platform. A user uploads a 2D
floor plan image (PNG/JPEG); the pipeline digitizes it into a navigable graph (nodes =
rooms/POIs, edges = walkable pathways) that an iOS client downloads for GPS-free indoor
turn-by-turn navigation.

## Pipeline architecture (one directional flow)

```
./maps/*.{png,jpg,jpeg}  →  cv2 load (native W×H)  →  base64 + Vision LLM (structured output)
   →  Pydantic-validated nodes/edges  →  networkx graph (Euclidean edge weights)
   →  ./output/building_graph.json   +   ./output/vector_map_overlay.png
```

Keep these as separate, independently testable modules/functions so the flow can be
demoed stage-by-stage live during the hackathon. Each stage's output is the next stage's
input — no stage reaches back.

### Stage contracts

1. **Ingestion** — Scan `./maps/` for incoming images. Load with OpenCV (`cv2`) to read
   *native* pixel dimensions. All coordinates downstream are in this pixel space.
2. **Vision LLM (structured output)** — Send the base64-encoded image to a live vision
   model (Gemini-2.5-flash via Google GenAI SDK, or GPT-4o via OpenAI SDK) using the
   `response_format` structured-outputs feature bound to the Pydantic schema. The system
   prompt instructs the model to locate room centers / hallway junctions and the paths
   connecting them, returning pixel coordinates.
3. **Compilation** — Feed validated JSON into `networkx`. Edge `weight` = Euclidean
   distance between the connected nodes' `(x, y)` coordinates, computed here (not from the
   model). Export topology to `./output/building_graph.json`.
4. **Visualization** — Use `matplotlib` + OpenCV to overlay nodes (bright markers) and
   edges (routing lines) on the original floor plan; save `./output/vector_map_overlay.png`.

### Pydantic schema (the data contract)

- `nodes`: each has `id`, textual room `name`, and exact `x` / `y` pixel coordinates.
- `edges`: each maps a `source` node id to a `target` node id.

This schema is shared by the LLM `response_format` and the networkx compiler — they must
stay in sync. Edge weights are derived, not part of the model's output.

## Hard constraints (do not violate)

- **No mock data, no hardcoded layout arrays, no sidecar JSON files.** Map data must come
  from a live Vision LLM call against the real uploaded image. This is the core demo point.
- Coordinates are real pixel coordinates from the analyzed image — do not normalize or
  fabricate them.
- Must handle: API timeouts, missing/invalid API keys, and invalid/unreadable image
  dimensions — surfaced clearly enough to diagnose during a live presentation.

## Commands

```bash
# Activate the existing venv (Git Bash / PowerShell respectively)
source .venv/Scripts/activate          # bash
.\.venv\Scripts\Activate.ps1           # PowerShell

# Dependencies are NOT yet installed. Core deps to install:
#   opencv-python networkx matplotlib pydantic numpy
#   google-genai   (Gemini path)   OR   openai   (GPT-4o path)
pip install opencv-python networkx matplotlib pydantic numpy google-genai
```

The Vision LLM API key is read from the environment (e.g. `GOOGLE_API_KEY` / `GEMINI_API_KEY`
for Gemini, `OPENAI_API_KEY` for GPT-4o). The pipeline must fail fast with a clear message
when it is missing rather than calling the API.
