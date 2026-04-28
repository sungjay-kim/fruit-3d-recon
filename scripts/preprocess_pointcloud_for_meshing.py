#!/usr/bin/env python3
"""Filter out near-black points from a binary little-endian PLY.

This is a conservative cleaner to strip the dark holder/noise from the bottom
before meshing. It does not add synthetic geometry or alter the remaining
points beyond the color-based filter.
"""

import os
import shutil
import struct
import sys
from typing import List, Tuple

SCALAR_TYPES = {
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


def parse_header(handle) -> Tuple[List[bytes], List[Tuple[str, str]], int]:
    header_lines: List[bytes] = []
    vertex_props: List[Tuple[str, str]] = []
    vertex_count = None
    current_element = None
    while True:
        line = handle.readline()
        if not line:
            raise RuntimeError("Unexpected EOF while reading header")
        header_lines.append(line)
        stripped = line.strip()
        if stripped == b"end_header":
            break
        if not stripped or stripped.startswith(b"comment"):
            continue
        parts = stripped.split()
        head = parts[0]
        if head == b"format":
            if parts[1] != b"binary_little_endian":
                raise RuntimeError("Only binary_little_endian PLY supported")
        elif head == b"element":
            current_element = parts[1].decode()
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif head == b"property" and current_element == "vertex":
            if parts[1] == b"list":
                raise RuntimeError("Vertex properties cannot be lists")
            vertex_props.append((parts[2].decode(), parts[1].decode()))
    if vertex_count is None:
        raise RuntimeError("Header missing 'element vertex' definition")
    return header_lines, vertex_props, vertex_count


def rewrite_vertex_line(header_lines: List[bytes], new_count: int) -> List[bytes]:
    updated = []
    for line in header_lines:
        if line.strip().startswith(b"element vertex"):
            updated.append(f"element vertex {new_count}\n".encode("ascii"))
        else:
            updated.append(line)
    return updated


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: preprocess_pointcloud_for_meshing.py <input.ply> <output.ply> "
            "[black_threshold]"
        )
    input_path, output_path = sys.argv[1:3]
    threshold = int(sys.argv[3]) if len(sys.argv) == 4 else 10
    threshold = max(-1, min(255, threshold))

    with open(input_path, "rb") as src:
        header_lines, vertex_props, vertex_count = parse_header(src)
        prop_names = [name for name, _ in vertex_props]
        try:
            r_idx = prop_names.index("red")
            g_idx = prop_names.index("green")
            b_idx = prop_names.index("blue")
        except ValueError:
            r_idx = g_idx = b_idx = None

        # If no RGB or filtering disabled, just copy through.
        if threshold < 0 or r_idx is None:
            tmp_path = output_path + ".tmp"
            src.seek(0)
            with open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp_path, output_path)
            print(
                f"No RGB available or threshold disabled; copied input to output unchanged "
                f"({vertex_count} vertices)."
            )
            return

        fmt = "<" + "".join(SCALAR_TYPES[dtype][0] for _, dtype in vertex_props)
        stride = struct.calcsize(fmt)

        kept = bytearray()
        kept_count = 0
        removed = 0
        for _ in range(vertex_count):
            chunk = src.read(stride)
            if len(chunk) != stride:
                raise RuntimeError("Unexpected EOF while reading vertex data")
            values = struct.unpack(fmt, chunk)
            if all(values[idx] <= threshold for idx in (r_idx, g_idx, b_idx)):
                removed += 1
                continue
            kept.extend(chunk)
            kept_count += 1
        tail_data = src.read()

    updated_header = rewrite_vertex_line(header_lines, kept_count)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as dst:
        dst.writelines(updated_header)
        dst.write(kept)
        dst.write(tail_data)
    os.replace(tmp_path, output_path)

    print(f"Kept {kept_count} / {vertex_count} vertices (removed {removed}).")


if __name__ == "__main__":
    main()
