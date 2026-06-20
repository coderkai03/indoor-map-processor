"""Indoor Map Processor — FastAPI server.

Exposes the processing pipeline over HTTP so iOS clients and any frontend
can upload floor plans and query navigation graphs.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /buildings/upload          Upload a floor plan image; runs the full pipeline.
    GET  /buildings/{id}/graph      Return the navigation graph JSON.
    GET  /buildings/{id}/nodes      Return all nodes (for search / autocomplete).
    GET  /buildings/{id}/route      Shortest path between two nodes.
    GET  /buildings/{id}/overlay    Return the QA overlay image.
"""

from __future__ import annotations

import datetime
import json
import shutil
import time as _time
import uuid
from pathlib import Path

import networkx as nx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from process_maps import (
    analyze_floorplan,
    build_graph,
    export_graph,
    get_client,
    load_image,
    render_overlay,
)

app = FastAPI(
    title="Indoor Map Processor API",
    description="Upload a floor plan image and get a navigable graph back.",
    version="0.1.0",
)

# Allow any origin so the iOS app / web frontend can call this freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BUILDINGS_DIR = Path("./output/buildings")
BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("./static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SUPPORTED_MIME = {"image/png", "image/jpeg"}


def _log(tag: str, msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


def _building_dir(building_id: str) -> Path:
    return BUILDINGS_DIR / building_id


def _graph_path(building_id: str) -> Path:
    return _building_dir(building_id) / "building_graph.json"


def _overlay_path(building_id: str) -> Path:
    return _building_dir(building_id) / "vector_map_overlay.png"


def _load_graph(building_id: str) -> nx.Graph:
    path = _graph_path(building_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node in data["nodes"]:
        graph.add_node(node["id"], name=node["name"], x=node["x"], y=node["y"])
    for edge in data["edges"]:
        graph.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    return graph


# --------------------------------------------------------------------------- #
# POST /buildings/upload
# --------------------------------------------------------------------------- #

@app.post("/buildings/upload", summary="Upload a floor plan and process it")
async def upload_building(file: UploadFile = File(...)):
    """
    Upload a PNG or JPEG floor plan image. The pipeline runs synchronously
    (Gemini vision call + graph compilation) and returns the building ID
    plus a summary of what was extracted.
    """
    if file.content_type not in SUPPORTED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload PNG or JPEG.",
        )

    t_start = _time.time()
    building_id = str(uuid.uuid4())
    bdir = _building_dir(building_id)
    bdir.mkdir(parents=True, exist_ok=True)

    _log("upload", f"Received '{file.filename}' ({file.content_type}) → building_id={building_id}")

    ext = ".png" if file.content_type == "image/png" else ".jpg"
    image_path = bdir / f"floor_plan{ext}"

    with image_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size_kb = image_path.stat().st_size // 1024
    _log("upload", f"Saved to disk ({file_size_kb} KB)")

    try:
        image_bgr, width, height = load_image(image_path)
    except ValueError as exc:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc))

    _log("upload", f"Image loaded: {width}×{height} px")

    image_bytes = image_path.read_bytes()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    try:
        client = get_client()
        _log("gemini", "Sending image to Gemini 2.5 Flash — waiting for structured output...")
        t_gemini = _time.time()
        floorplan = analyze_floorplan(client, image_bytes, mime_type, width, height)
        gemini_elapsed = _time.time() - t_gemini
        _log("gemini", f"Response received in {gemini_elapsed:.1f}s — {len(floorplan.nodes)} nodes, {len(floorplan.edges)} edges")
    except EnvironmentError as exc:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"Vision API error: {exc}")

    _log("graph", "Building networkx graph with Euclidean edge weights...")
    graph = build_graph(floorplan)
    _log("graph", f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    export_graph(graph, width, height, out_path=_graph_path(building_id))
    _log("export", f"Graph JSON saved → {_graph_path(building_id)}")

    render_overlay(image_bgr, graph, out_path=_overlay_path(building_id))
    _log("render", f"Overlay PNG saved → {_overlay_path(building_id)}")

    total_elapsed = _time.time() - t_start
    _log("done", f"Pipeline complete in {total_elapsed:.1f}s  |  building_id={building_id}")

    return {
        "building_id": building_id,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "image": {"width": width, "height": height},
        "processing_time_s": round(total_elapsed, 1),
        "gemini_time_s": round(gemini_elapsed, 1),
    }


# --------------------------------------------------------------------------- #
# GET /buildings/{building_id}/graph
# --------------------------------------------------------------------------- #

@app.get("/buildings/{building_id}/graph", summary="Get the navigation graph")
def get_graph(building_id: str):
    """
    Returns the full navigation graph JSON for the given building.
    The iOS client uses this to build its local routing structure.
    """
    path = _graph_path(building_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# GET /buildings/{building_id}/nodes
# --------------------------------------------------------------------------- #

@app.get("/buildings/{building_id}/nodes", summary="List all nodes (rooms/POIs)")
def get_nodes(building_id: str):
    """
    Returns all nodes with id, name, and coordinates.
    Useful for populating a destination search / autocomplete on the client.
    """
    path = _graph_path(building_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"building_id": building_id, "nodes": data["nodes"]}


# --------------------------------------------------------------------------- #
# GET /buildings/{building_id}/route
# --------------------------------------------------------------------------- #

@app.get("/buildings/{building_id}/route", summary="Get shortest path between two nodes")
def get_route(
    building_id: str,
    from_node: str = Query(..., alias="from", description="Source node id"),
    to_node: str = Query(..., alias="to", description="Target node id"),
):
    """
    Returns the shortest walking path from `from` to `to` as an ordered list
    of nodes with names and pixel coordinates, plus the total distance in pixels.
    """
    graph = _load_graph(building_id)

    if from_node not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{from_node}' not found.")
    if to_node not in graph.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{to_node}' not found.")
    if not nx.has_path(graph, from_node, to_node):
        raise HTTPException(
            status_code=422,
            detail=f"No walkable path exists between '{from_node}' and '{to_node}'.",
        )

    node_ids = nx.shortest_path(graph, source=from_node, target=to_node, weight="weight")
    total_distance = nx.shortest_path_length(
        graph, source=from_node, target=to_node, weight="weight"
    )

    steps = [
        {
            "id": nid,
            "name": graph.nodes[nid]["name"],
            "x": graph.nodes[nid]["x"],
            "y": graph.nodes[nid]["y"],
        }
        for nid in node_ids
    ]

    return {
        "building_id": building_id,
        "from": from_node,
        "to": to_node,
        "total_distance_px": round(total_distance, 2),
        "step_count": len(steps),
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# GET /buildings/{building_id}/overlay
# --------------------------------------------------------------------------- #

@app.get("/buildings/{building_id}/overlay", summary="Get the QA overlay image")
def get_overlay(building_id: str):
    """
    Returns the PNG visualization of the floor plan with the graph drawn on top.
    Useful for verifying that Gemini extracted the map correctly.
    """
    path = _overlay_path(building_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Overlay for building '{building_id}' not found.")
    return FileResponse(path, media_type="image/png")
