"""Indoor Map Processor — server-side ingestion pipeline.

Turns an uploaded 2D floor plan image into a navigable graph (nodes = rooms/POIs,
edges = walkable pathways) using a live Gemini 2.5 Flash vision call.

Pipeline (each stage is an independent, separately-callable function for live demos):

    maps/*.{png,jpg,jpeg}
        -> load_image           (cv2, native width x height)
        -> analyze_floorplan    (Gemini, Pydantic structured output)
        -> build_graph          (networkx + Euclidean edge weights)
        -> export_graph         (output/building_graph.json)
        +  render_overlay       (output/vector_map_overlay.png)

Run:  python process_maps.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import networkx as nx
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; env vars may be set externally.
    pass

# Use a non-interactive backend so overlays render on headless servers.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MAPS_DIR = Path("./maps")
OUTPUT_DIR = Path("./output")
GRAPH_JSON_PATH = OUTPUT_DIR / "building_graph.json"
OVERLAY_PATH = OUTPUT_DIR / "vector_map_overlay.png"

MODEL_NAME = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

API_TIMEOUT_MS = 60_000  # Per-request timeout for the vision call.
MAX_RETRIES = 2  # Total attempts on transient/timeout failures.

SYSTEM_PROMPT = (
    "You are a computer-vision system that digitizes 2D indoor floor plans into "
    "navigation graphs for turn-by-turn indoor navigation. Study the floor plan carefully and follow these rules exactly.\n\n"
    "NODES — what to create:\n"
    "- One node at the VISUAL CENTER of every named room, office, restroom, stairwell, elevator, and entrance/exit.\n"
    "- One junction node at EVERY hallway intersection, T-junction, and corridor bend. "
    "These are critical: without them, routes will cut through walls.\n"
    "- For long straight corridors (longer than 200 pixels), place intermediate junction nodes "
    "every ~150 pixels along the corridor centerline.\n"
    "- Node coordinates must be inside the open walkable area of the room or corridor — "
    "NEVER on a wall, door frame, or outside the building boundary.\n\n"
    "NODE NAMING — use real-world names:\n"
    "- Rooms: use their number or label as printed (e.g. 'Room 101', 'Conference Room B').\n"
    "- Common areas: 'Lobby', 'Restroom', 'Men's Restroom', 'Women's Restroom', 'Elevator', "
    "'Stairwell', 'Emergency Exit', 'Main Entrance', 'Reception'.\n"
    "- Junctions: 'Hallway Junction', 'North Corridor Junction' — only use generic names "
    "when no real-world label exists.\n\n"
    "EDGES — connectivity rules:\n"
    "- Draw an edge ONLY when a person can walk directly between two nodes without passing through a wall.\n"
    "- A valid edge requires a visible doorway, open corridor, or unobstructed opening between the two nodes.\n"
    "- NEVER connect two rooms through a wall, even if they are close together.\n"
    "- Connect rooms to the nearest hallway junction node, not directly to other rooms "
    "unless there is a direct door between them.\n"
    "- Every node must be reachable (no isolated nodes).\n\n"
    "Coordinates are in PIXELS, origin (0,0) at TOP-LEFT, x right, y down. "
    "Every x must be in [0, width) and every y in [0, height). "
    "Node ids must be unique strings (use snake_case, e.g. 'room_101', 'lobby', 'hallway_junction_1'). "
    "Every edge source and target must reference an existing node id."
)


# --------------------------------------------------------------------------- #
# Schema — shared by the Gemini response_schema and the graph compiler
# --------------------------------------------------------------------------- #


class Node(BaseModel):
    """A room, POI, or hallway junction located at a pixel coordinate."""

    id: str = Field(description="Unique identifier for the node.")
    name: str = Field(description="Human-readable room or POI name.")
    x: int = Field(description="X pixel coordinate of the node center.")
    y: int = Field(description="Y pixel coordinate of the node center.")


class Edge(BaseModel):
    """A walkable connection from one node to another."""

    source: str = Field(description="id of the source node.")
    target: str = Field(description="id of the target node.")


class FloorPlan(BaseModel):
    """Structured map extracted from a floor plan image."""

    nodes: list[Node] = Field(description="All rooms, POIs, and junctions.")
    edges: list[Edge] = Field(description="Walkable connections between nodes.")


# --------------------------------------------------------------------------- #
# Stage 1 — Dynamic map ingestion
# --------------------------------------------------------------------------- #


def find_maps(maps_dir: Path = MAPS_DIR) -> list[Path]:
    """Return all floor plan images in ``maps_dir`` (PNG/JPEG)."""
    if not maps_dir.exists():
        print(f"[ingest] Maps directory '{maps_dir}' does not exist — nothing to process.")
        return []
    images = sorted(
        p for p in maps_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"[ingest] No PNG/JPEG floor plans found in '{maps_dir}'.")
    return images


def load_image(path: Path) -> tuple["cv2.Mat", int, int]:
    """Load an image with OpenCV and return ``(image_bgr, width, height)``.

    Raises ``ValueError`` if the file is unreadable or has invalid dimensions.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image (corrupt or unsupported): {path}")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions {width}x{height}: {path}")
    print(f"[ingest] Loaded '{path.name}' ({width}x{height} px).")
    return image, width, height


# --------------------------------------------------------------------------- #
# Stage 2 — Live Vision API with structured outputs
# --------------------------------------------------------------------------- #


def get_client() -> genai.Client:
    """Construct a Gemini client, failing fast if the API key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
            "environment or a .env file before running."
        )
    return genai.Client(api_key=api_key)


def analyze_floorplan(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    width: int,
    height: int,
) -> FloorPlan:
    """Send the image to Gemini and return a validated ``FloorPlan``.

    The image bytes are handed to the SDK directly (it base64-encodes them on the
    wire). Structured output is enforced via ``response_schema=FloorPlan``.
    """
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    user_prompt = (
        f"This floor plan image is {width} pixels wide and {height} pixels tall. "
        "Extract the navigation graph as structured JSON matching the schema."
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=FloorPlan,
        http_options=types.HttpOptions(timeout=API_TIMEOUT_MS),
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[vision] Calling {MODEL_NAME} (attempt {attempt}/{MAX_RETRIES})...")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[user_prompt, image_part],
                config=config,
            )
            floorplan = response.parsed
            if not isinstance(floorplan, FloorPlan):
                # Fall back to parsing raw text if the SDK didn't auto-parse.
                floorplan = FloorPlan.model_validate_json(response.text)
            print(
                f"[vision] Received {len(floorplan.nodes)} nodes, "
                f"{len(floorplan.edges)} edges."
            )
            return floorplan
        except (genai_errors.APIError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            print(f"[vision] API call failed: {exc}")
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt
                print(f"[vision] Retrying in {backoff}s...")
                time.sleep(backoff)

    raise RuntimeError(
        f"Vision API failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------- #
# Stage 3 — Network compilation
# --------------------------------------------------------------------------- #


def build_graph(floorplan: FloorPlan) -> nx.Graph:
    """Compile the structured map into a weighted networkx graph.

    Edge weights are the Euclidean pixel distance between endpoints. Edges that
    reference unknown nodes are dropped with a warning so one bad edge can't
    break compilation.
    """
    graph = nx.Graph()
    nodes_by_id: dict[str, Node] = {}
    for node in floorplan.nodes:
        if node.id in nodes_by_id:
            print(f"[compile] Duplicate node id '{node.id}' — keeping the first.")
            continue
        nodes_by_id[node.id] = node
        graph.add_node(node.id, name=node.name, x=node.x, y=node.y)

    for edge in floorplan.edges:
        src, tgt = nodes_by_id.get(edge.source), nodes_by_id.get(edge.target)
        if src is None or tgt is None:
            print(
                f"[compile] Dropping dangling edge {edge.source} -> {edge.target} "
                "(unknown node)."
            )
            continue
        weight = math.hypot(tgt.x - src.x, tgt.y - src.y)
        graph.add_edge(edge.source, edge.target, weight=round(weight, 2))

    print(
        f"[compile] Graph compiled: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges."
    )
    return graph


def export_graph(graph: nx.Graph, width: int, height: int, out_path: Path = GRAPH_JSON_PATH) -> None:
    """Write the lightweight graph payload for the iOS client."""
    payload = {
        "image": {"width": width, "height": height},
        "nodes": [
            {"id": nid, "name": data["name"], "x": data["x"], "y": data["y"]}
            for nid, data in graph.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "weight": data["weight"]}
            for u, v, data in graph.edges(data=True)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[export] Wrote graph topology -> {out_path}")


# --------------------------------------------------------------------------- #
# Stage 4 — Consumer visualization
# --------------------------------------------------------------------------- #


def render_overlay(image_bgr: "cv2.Mat", graph: nx.Graph, out_path: Path = OVERLAY_PATH) -> None:
    """Overlay the graph (markers + routing lines) on the floor plan and save it."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image_rgb)

    # Edges as routing lines.
    for u, v in graph.edges():
        x = [graph.nodes[u]["x"], graph.nodes[v]["x"]]
        y = [graph.nodes[u]["y"], graph.nodes[v]["y"]]
        ax.plot(x, y, color="#00E5FF", linewidth=2, alpha=0.9, zorder=2)

    # Nodes as bright markers with labels.
    for nid, data in graph.nodes(data=True):
        ax.plot(data["x"], data["y"], "o", color="#FF1744", markersize=10, zorder=3)
        ax.annotate(
            data["name"],
            (data["x"], data["y"]),
            color="white",
            fontsize=8,
            fontweight="bold",
            xytext=(6, -6),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec="none", alpha=0.6),
            zorder=4,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Match image coordinates (origin top-left).
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[render] Wrote overlay -> {out_path}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def process_image(client: genai.Client, path: Path) -> bool:
    """Run the full pipeline for a single image. Returns True on success."""
    print(f"\n=== Processing '{path.name}' ===")
    try:
        image_bgr, width, height = load_image(path)
    except ValueError as exc:
        print(f"[error] Skipping '{path.name}': {exc}")
        return False

    image_bytes = path.read_bytes()
    mime_type = MIME_TYPES[path.suffix.lower()]

    try:
        floorplan = analyze_floorplan(client, image_bytes, mime_type, width, height)
    except RuntimeError as exc:
        print(f"[error] Vision analysis failed for '{path.name}': {exc}")
        return False

    graph = build_graph(floorplan)
    export_graph(graph, width, height)
    render_overlay(image_bgr, graph)
    return True


def main() -> int:
    images = find_maps()
    if not images:
        return 0

    try:
        client = get_client()
    except EnvironmentError as exc:
        print(f"[error] {exc}")
        return 1

    succeeded = sum(process_image(client, path) for path in images)
    print(f"\nDone. Processed {succeeded}/{len(images)} image(s) successfully.")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
