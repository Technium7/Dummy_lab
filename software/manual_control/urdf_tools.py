from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IGNORED_DIR_NAMES = {
    ".git",
    ".conda",
    ".venv",
    "build",
    "cmake-build-debug",
    "install",
    "log",
    "node_modules",
}


def discover_descriptions(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in {".urdf", ".xacro"} and path.name.lower() != "materials.xacro":
            candidates.append(path)
    return sorted(candidates)


def choose_description(candidates: Sequence[Path], robot_hint: str, root: Path) -> Path:
    if not candidates:
        raise SystemExit(f"No .urdf or .xacro files found under {root}")
    if robot_hint:
        hint_path = (root / robot_hint).resolve() if not Path(robot_hint).is_absolute() else Path(robot_hint)
        if hint_path.exists() and hint_path.suffix.lower() in {".urdf", ".xacro"}:
            return hint_path
        filtered = [p for p in candidates if robot_hint.lower() in str(p).lower()]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            candidates = filtered
        else:
            raise SystemExit(f"No robot description matched --robot={robot_hint!r}")
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit("Multiple robot descriptions found; pass --robot to disambiguate.")


def read_package_name(package_xml: Path) -> Optional[str]:
    try:
        root = ET.parse(package_xml).getroot()
    except ET.ParseError:
        return None
    node = root.find("name")
    return (node.text.strip() if node is not None and node.text else None) or None


def build_package_map(search_root: Path) -> Dict[str, Path]:
    package_map: Dict[str, Path] = {}
    for pkg_xml in search_root.rglob("package.xml"):
        if any(part in IGNORED_DIR_NAMES for part in pkg_xml.parts):
            continue
        pkg_name = read_package_name(pkg_xml)
        if pkg_name:
            package_map[pkg_name] = pkg_xml.parent
    return package_map


def rewrite_ros_paths(xml_text: str, package_map: Dict[str, Path]) -> str:
    xml_text = re.sub(r"\$\(find\s+([^)\s]+)\)", lambda m: str(package_map.get(m.group(1), m.group(0))), xml_text)
    xml_text = re.sub(
        r"package://([^/]+)/",
        lambda m: f"{package_map[m.group(1)].as_posix()}/" if m.group(1) in package_map else m.group(0),
        xml_text,
    )
    return xml_text


def prepare_xacro_without_ament(xacro_path: Path, package_map: Dict[str, Path]) -> Path:
    matching_roots = [root for root in package_map.values() if root in xacro_path.parents]
    package_root = sorted(matching_roots, key=lambda p: len(str(p)), reverse=True)[0] if matching_roots else xacro_path.parent
    temp_root = Path(tempfile.mkdtemp(prefix="xacro_pre_"))
    temp_pkg_root = temp_root / package_root.name
    shutil.copytree(package_root, temp_pkg_root)
    for path in temp_pkg_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".xacro", ".urdf", ".gazebo", ".ros2control", ".xml"}:
            path.write_text(rewrite_ros_paths(path.read_text(encoding="utf-8"), package_map), encoding="utf-8")
    return temp_pkg_root / xacro_path.relative_to(package_root)


def xacro_to_urdf_text(xacro_path: Path, package_map: Dict[str, Path]) -> str:
    try:
        import xacro  # type: ignore

        try:
            return xacro.process_file(str(xacro_path)).toxml()
        except Exception as exc:
            if "ament_index_python" not in str(exc):
                raise
            patched_xacro = prepare_xacro_without_ament(xacro_path, package_map)
            try:
                return xacro.process_file(str(patched_xacro)).toxml()
            finally:
                shutil.rmtree(patched_xacro.parent.parent, ignore_errors=True)
    except ImportError:
        try:
            return subprocess.run(["xacro", str(xacro_path)], check=True, text=True, capture_output=True).stdout
        except Exception as exc:
            raise SystemExit(f"xacro failed for {xacro_path}: {exc}") from exc


def materialize_urdf(description_path: Path, package_map: Dict[str, Path]) -> Tuple[Path, Optional[Path]]:
    if description_path.suffix.lower() == ".urdf":
        return description_path, None
    xml_text = rewrite_ros_paths(xacro_to_urdf_text(description_path, package_map), package_map)
    temp_file = tempfile.NamedTemporaryFile(prefix="urdf_view_", suffix=".urdf", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.write(xml_text.encode("utf-8"))
    temp_file.flush()
    temp_file.close()
    return temp_path, temp_path

