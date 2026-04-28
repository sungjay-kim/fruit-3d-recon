#!/usr/bin/env python3
"""
Compute the volume of a watertight triangular mesh stored in a binary
little-endian PLY file and write the result to one or more JSON files.

Usage:
    mesh_volume.py <mesh.ply> <workspace_json|-> <label>
    mesh_volume.py <mesh.ply> <workspace_json|-> <label> <sample_id> <dense_id> <global_json>

Pass "-" as the workspace_json to skip writing a per-workspace file.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

SCALAR_TYPES: Dict[str, Tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


class PlyError(RuntimeError):
    pass


def parse_header(handle) -> List[Dict]:
    first = handle.readline().decode("ascii", errors="ignore").strip()
    if first != "ply":
        raise PlyError("Not a PLY file")
    fmt = None
    elements: List[Dict] = []
    current = None
    while True:
        line = handle.readline()
        if not line:
            raise PlyError("Unexpected EOF while parsing header")
        text = line.decode("ascii", errors="ignore").strip()
        if text == "end_header":
            break
        if not text or text.startswith("comment"):
            continue
        parts = text.split()
        head = parts[0]
        if head == "format":
            fmt = parts[1]
        elif head == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif head == "property" and current is not None:
            if parts[1] == "list":
                current["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "item_type": parts[3],
                        "name": parts[4],
                    }
                )
            else:
                current["properties"].append(
                    {"kind": "scalar", "data_type": parts[1], "name": parts[2]}
                )
    if fmt != "binary_little_endian":
        raise PlyError(f"Unsupported PLY format: {fmt}")
    return elements


def read_vertices(handle, elem) -> List[Tuple[float, float, float]]:
    props = elem["properties"]
    if not props:
        raise PlyError("Vertex element has no properties")
    fmt = "<" + "".join(SCALAR_TYPES[p["data_type"]][0] for p in props)
    size = struct.calcsize(fmt)
    idx = {p["name"]: i for i, p in enumerate(props)}
    if not {"x", "y", "z"}.issubset(idx):
        raise PlyError("Vertex properties missing x/y/z")
    verts: List[Tuple[float, float, float]] = []
    for _ in range(elem["count"]):
        chunk = handle.read(size)
        if len(chunk) != size:
            raise PlyError("Unexpected EOF while reading vertices")
        values = struct.unpack(fmt, chunk)
        verts.append(
            (
                float(values[idx["x"]]),
                float(values[idx["y"]]),
                float(values[idx["z"]]),
            )
        )
    return verts


def read_faces(handle, elem) -> List[Sequence[int]]:
    list_prop = next((p for p in elem["properties"] if p["kind"] == "list"), None)
    if list_prop is None:
        raise PlyError("Face element lacks list property")
    count_code, count_size = SCALAR_TYPES[list_prop["count_type"]]
    item_code, item_size = SCALAR_TYPES[list_prop["item_type"]]
    faces: List[Sequence[int]] = []
    for _ in range(elem["count"]):
        count_bytes = handle.read(count_size)
        if len(count_bytes) != count_size:
            raise PlyError("Unexpected EOF while reading face counts")
        num = struct.unpack("<" + count_code, count_bytes)[0]
        if num <= 0:
            faces.append(tuple())
            continue
        data = handle.read(item_size * num)
        if len(data) != item_size * num:
            raise PlyError("Unexpected EOF while reading face indices")
        indices = struct.unpack("<" + (item_code * num), data)
        faces.append(indices)
    return faces


def triangle_volume(a, b, c) -> float:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    ) / 6.0


def compute_volume(verts, faces) -> float:
    total = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        v0 = verts[face[0]]
        for i in range(1, len(face) - 1):
            v1 = verts[face[i]]
            v2 = verts[face[i + 1]]
            total += triangle_volume(v0, v1, v2)
    return abs(total)


def main() -> None:
    if len(sys.argv) not in (4, 7):
        raise SystemExit(
            "Usage: mesh_volume.py <mesh.ply> <volumes.json> <label> "
            "[<sample_id> <dense_id> <global.json>]"
        )
    mesh_path, volume_path, label = sys.argv[1:4]
    global_args = sys.argv[4:]
    with open(mesh_path, "rb") as handle:
        elements = parse_header(handle)
        vert_elem = next((e for e in elements if e["name"] == "vertex"), None)
        face_elem = next((e for e in elements if e["name"] == "face"), None)
        if vert_elem is None or face_elem is None:
            raise PlyError("PLY missing vertex or face elements")
        vertices = read_vertices(handle, vert_elem)
        faces = read_faces(handle, face_elem)
    volume = compute_volume(vertices, faces)
    data = {}
    if os.path.exists(volume_path):
        try:
            with open(volume_path, "r", encoding="utf-8") as existing:
                data = json.load(existing)
        except Exception:
            data = {}
    local_path = None if volume_path in ("", "-") else volume_path
    if local_path:
        data = {}
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as existing:
                    data = json.load(existing)
            except Exception:
                data = {}
        data[label] = {"mesh": mesh_path, "volume": volume}
        with open(local_path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
    if global_args:
        sample_id, dense_id, global_path = global_args
        full = {}
        if os.path.exists(global_path):
            try:
                with open(global_path, "r", encoding="utf-8") as existing:
                    full = json.load(existing)
            except Exception:
                full = {}
        sample_entry = full.setdefault(sample_id, {})
        dense_entry = sample_entry.setdefault(dense_id, {})
        dense_entry[label] = {"mesh": mesh_path, "volume": volume}
        with open(global_path, "w", encoding="utf-8") as gout:
            json.dump(full, gout, indent=2)


if __name__ == "__main__":
    main()
