"""Build the simplified Africa geometry used by the application map.

Reads a Natural Earth admin-0 GeoJSON (110m resolution is enough for a page
graphic), keeps the African features, simplifies each ring with the
Ramer-Douglas-Peucker algorithm and writes a compact JSON file to
``data/reference/africa_geometry.json``.

The output is deliberately small (page graphic, not analysis geometry) and is
committed so the application has no runtime download and no extra geospatial
dependency.

Usage:
    python scripts/build_geo_reference.py --source path/to/ne_110m_admin_0_countries.geojson
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    """Distance from a point to the segment through start and end.

    Args:
        point: The point to measure, as (x, y).
        start: Segment start, as (x, y).
        end: Segment end, as (x, y).

    Returns:
        Perpendicular distance in coordinate units (degrees here).
    """
    (x, y), (x1, y1), (x2, y2) = point, start, end
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px, py = x1 + t * dx, y1 + t * dy
    return ((x - px) ** 2 + (y - py) ** 2) ** 0.5


def simplify(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Simplify a polyline with the Ramer-Douglas-Peucker algorithm.

    Args:
        points: Ordered coordinates.
        tolerance: Maximum allowed deviation, in coordinate units.

    Returns:
        Simplified list of coordinates, endpoints preserved.
    """
    if len(points) < 3:
        return list(points)
    first, last = points[0], points[-1]
    index, max_dist = -1, 0.0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], first, last)
        if dist > max_dist:
            index, max_dist = i, dist
    if max_dist <= tolerance:
        return [first, last]
    left = simplify(points[: index + 1], tolerance)
    right = simplify(points[index:], tolerance)
    return left[:-1] + right


def ring_area(points: Sequence[Point]) -> float:
    """Absolute shoelace area of a ring, in squared degrees.

    Args:
        points: Ring coordinates.

    Returns:
        Absolute polygon area.
    """
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygons_of(geometry: Dict[str, Any]) -> List[List[Point]]:
    """Extract exterior rings from a Polygon or MultiPolygon geometry.

    Args:
        geometry: GeoJSON geometry dict.

    Returns:
        List of exterior rings; holes are dropped because the map is a page
        graphic rather than an analytical surface.
    """
    kind = geometry.get("type")
    if kind == "Polygon":
        return [[(float(x), float(y)) for x, y in geometry["coordinates"][0]]]
    if kind == "MultiPolygon":
        return [
            [(float(x), float(y)) for x, y in polygon[0]]
            for polygon in geometry["coordinates"]
        ]
    raise ValueError(f"unsupported geometry type: {kind!r}")


def build(
    source: Path,
    output: Path,
    tolerance: float = 0.12,
    min_area: float = 0.6,
    precision: int = 2,
) -> Dict[str, Any]:
    """Build the simplified Africa geometry file.

    Args:
        source: Natural Earth admin-0 GeoJSON path.
        output: Destination JSON path.
        tolerance: Simplification tolerance in degrees.
        min_area: Drop rings smaller than this, in squared degrees, except
            each country's largest ring.
        precision: Decimal places kept per coordinate.

    Returns:
        The written payload dict.

    Raises:
        FileNotFoundError: If the source file is missing.
    """
    if not source.exists():
        raise FileNotFoundError(f"source geojson not found: {source}")

    raw = json.loads(source.read_text(encoding="utf-8"))
    countries: List[Dict[str, Any]] = []

    for feature in raw["features"]:
        props = feature.get("properties", {})
        if props.get("CONTINENT") != "Africa":
            continue
        iso3 = props.get("ADM0_A3") or props.get("ISO_A3")
        rings = polygons_of(feature["geometry"])
        simplified = []
        for ring in rings:
            reduced = simplify(ring, tolerance)
            if len(reduced) < 4:
                continue
            simplified.append((ring_area(reduced), [
                [round(x, precision), round(y, precision)] for x, y in reduced
            ]))
        if not simplified:
            continue
        simplified.sort(key=lambda pair: -pair[0])
        kept = [simplified[0][1]] + [
            ring for area, ring in simplified[1:] if area >= min_area
        ]
        countries.append({
            "iso3": iso3,
            "name": props.get("NAME_LONG") or props.get("NAME"),
            "rings": kept,
        })

    countries.sort(key=lambda c: c["iso3"])
    payload = {
        "source": "Natural Earth 1:110m admin 0 countries (public domain)",
        "simplification_tolerance_degrees": tolerance,
        "countries": countries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    n_points = sum(len(r) for c in countries for r in c["rings"])
    logger.info(
        "Wrote %s: %d countries, %d rings, %d points, %.0f kB",
        output, len(countries),
        sum(len(c["rings"]) for c in countries), n_points,
        output.stat().st_size / 1024,
    )
    return payload


def main() -> None:
    """Command line entry point.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Natural Earth admin-0 GeoJSON file")
    parser.add_argument("--output", type=Path,
                        default=Path("data/reference/africa_geometry.json"))
    parser.add_argument("--tolerance", type=float, default=0.12)
    args = parser.parse_args()
    build(args.source, args.output, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
