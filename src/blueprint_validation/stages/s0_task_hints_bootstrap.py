"""Stage 0: Bootstrap synthetic task hints for facilities without source metadata."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np

from ..common import StageResult, get_logger, read_json, write_json
from ..config import CameraPathSpec, FacilityConfig, ValidationConfig
from ..rendering.scene_geometry import (
    compute_scene_transform,
    detect_up_axis,
    is_identity_transform,
    transform_means,
)
from ..rendering.vlm_scene_detector import (
    DetectedRegion,
    SceneDetectionResult,
    detect_and_generate_specs,
)
from ..warmup import load_ply_means_and_colors_numpy
from .base import PipelineStage

logger = get_logger("stages.s0_task_hints_bootstrap")

_DEFAULT_HINTS_FILE = "task_targets.synthetic.json"
_MANUAL_ANALYSIS_DETAIL = (
    "VLM bootstrap produced no usable detections/specs. Manual analysis is required: "
    "generate task_targets.json (or task_targets.synthetic.json) with object/task hints "
    "and set facility.task_hints_path before rerunning."
)


def _default_output_path(work_dir: Path) -> Path:
    return work_dir / "bootstrap" / _DEFAULT_HINTS_FILE


# ---------------------------------------------------------------------------
# InteriorGS direct ingestion
# ---------------------------------------------------------------------------

# Semantic label keywords that imply an articulated (openable) object.
_ARTICULATION_KEYWORDS = frozenset(
    {
        "door",
        "cabinet",
        "drawer",
        "refrigerator",
        "dishwasher",
        "oven",
        "microwave",
        "washer",
        "dryer",
        "window",
        "shutter",
        "wardrobe",
        "closet",
        "hatch",
        "flap",
        "lid",
        "gate",
        "cupboard",
    }
)

# Semantic label keywords that imply a navigable region rather than an object.
_NAVIGATION_KEYWORDS = frozenset(
    {
        "room",
        "hallway",
        "corridor",
        "aisle",
        "dock",
        "zone",
        "area",
        "region",
        "floor",
        "waypoint",
        "entrance",
        "exit",
        "stairway",
        "staircase",
        "elevator",
        "lobby",
    }
)

_PICKABLE_OBJECT_KEYWORDS = frozenset(
    {
        "bottle",
        "bowl",
        "box",
        "can",
        "cup",
        "dish",
        "glass",
        "kettle",
        "knife",
        "mug",
        "pan",
        "plate",
        "pot",
        "rice",
        "remote",
        "spoon",
    }
)

_NON_PICKABLE_OBJECT_KEYWORDS = frozenset(
    {
        "bed",
        "cabinet",
        "ceiling",
        "chair",
        "counter",
        "dishwasher",
        "door",
        "drawer",
        "dryer",
        "fan",
        "floor",
        "fridge",
        "lamp",
        "microwave",
        "oven",
        "refrigerator",
        "shelf",
        "sink",
        "sofa",
        "stove",
        "table",
        "tv",
        "wall",
        "wardrobe",
        "washer",
        "window",
    }
)

_TOGGLEABLE_KEYWORDS = frozenset(
    {
        "air",
        "burner",
        "fan",
        "heater",
        "hood",
        "kettle",
        "lamp",
        "light",
        "microwave",
        "oven",
        "stove",
        "switch",
    }
)


def _infer_category_from_label(label: str) -> str:
    """Map a semantic label string to manipulation/articulation/navigation."""
    tokens = _label_tokens(label)
    token_set = set(tokens)
    for kw in _ARTICULATION_KEYWORDS:
        if kw in token_set:
            return "articulation"
    for kw in _NAVIGATION_KEYWORDS:
        if kw in token_set:
            return "navigation"
    return "manipulation"


def _label_tokens(label: str) -> List[str]:
    """Tokenize a semantic label into lowercase words."""
    return [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]


def _semantic_class_key(label: str) -> str:
    """Return a coarse semantic class key for de-duplication across sources."""
    tokens = _label_tokens(label)
    if not tokens:
        return ""
    # Use the head noun heuristic (last token) to collapse variants like
    # "office_chair" and "chair" into the same semantic class.
    return tokens[-1]


def _obb_from_corners(corners: list) -> Optional[dict]:
    """Derive center + AABB extents from a list of 3D corner vertices.

    Accepts two formats:
      - list of ``[x, y, z]`` sequences (normalized / test format)
      - list of ``{"x": ..., "y": ..., "z": ...}`` dicts (actual InteriorGS format)

    Returns None if the input is malformed.
    """
    try:
        if not isinstance(corners, (list, tuple)):
            return None
        # Normalise dict points to [x, y, z] sequences.
        normalised = []
        for pt in corners:
            if isinstance(pt, dict):
                normalised.append([pt["x"], pt["y"], pt["z"]])
            else:
                normalised.append(pt)
        pts = np.array(normalised, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            return None
        center = pts.mean(axis=0).tolist()
        extents = (pts.max(axis=0) - pts.min(axis=0)).tolist()
        return {
            "center": center,
            "extents": extents,
            "axes": np.eye(3, dtype=np.float64).tolist(),
        }
    except (ValueError, TypeError, KeyError):
        return None


def _obb_from_position_size(position: list, size: list) -> Optional[dict]:
    """Build a bounding box dict from a center position + [w, l, h] size.

    Used for structure.json ``ins`` entries.
    """
    try:
        center = [float(v) for v in position]
        extents = [float(v) for v in size]
        if len(center) != 3 or len(extents) != 3:
            return None
        return {
            "center": center,
            "extents": extents,
            "axes": np.eye(3, dtype=np.float64).tolist(),
        }
    except (ValueError, TypeError):
        return None


def _interiorgs_scene_type(structure_data: dict) -> str:
    """Derive a scene_type string from structure.json room definitions."""
    rooms = structure_data.get("rooms", [])
    room_types = [r.get("room_type", "") for r in rooms if r.get("room_type")]
    if not room_types:
        return "indoor"
    counts = Counter(room_types)
    return counts.most_common(1)[0][0].lower().replace(" ", "_")


def _prompt_token_from_entry(entry: dict) -> str:
    """Build a stable prompt token, e.g. ``bowl_101``."""
    label = str(entry.get("label") or "object")
    label_token = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"
    instance_id = str(entry.get("instance_id") or "").strip()
    iid_token = re.sub(r"[^a-zA-Z0-9]+", "", instance_id)
    if iid_token:
        return f"{label_token}_{iid_token}"
    return label_token


def _entry_extents(entry: dict) -> Optional[np.ndarray]:
    bbox = entry.get("boundingBox")
    if not isinstance(bbox, dict):
        return None
    extents = bbox.get("extents")
    if not isinstance(extents, list) or len(extents) != 3:
        return None
    try:
        arr = np.asarray([float(v) for v in extents], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(arr)):
        return None
    return arr


def _is_plausible_pick_candidate(entry: dict) -> bool:
    """Heuristic filter for object-level pick/place prompts."""
    label = str(entry.get("label") or "")
    tokens = set(_label_tokens(label))
    if not tokens:
        return False
    if tokens & _NON_PICKABLE_OBJECT_KEYWORDS:
        return False

    extents = _entry_extents(entry)
    if extents is not None:
        max_extent = float(extents.max())
        volume = float(np.prod(extents))
        # Keep prompts focused on reasonably graspable objects.
        if max_extent > 0.60 or volume > 0.08:
            return False

    return True


def _entry_family(entry: dict) -> str:
    label = str(entry.get("label") or "")
    family = _semantic_class_key(label)
    if family:
        return family
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "object"


def _select_prompt_entries(
    entries: List[dict],
    limit: int,
    *,
    pick_filter: bool = False,
    max_per_family: int = 2,
) -> List[dict]:
    selected: List[dict] = []
    seen: set = set()
    family_counts: Dict[str, int] = {}

    def _add(entry: dict) -> None:
        key = (
            str(entry.get("instance_id") or "").strip(),
            str(entry.get("label") or "").strip().lower(),
        )
        if key in seen:
            return
        family = _entry_family(entry)
        if family and family_counts.get(family, 0) >= max(1, max_per_family):
            return
        seen.add(key)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        selected.append(entry)

    if pick_filter:
        # First pass: prioritize common graspable object classes.
        for entry in entries:
            label_tokens = set(_label_tokens(str(entry.get("label") or "")))
            if not _is_plausible_pick_candidate(entry):
                continue
            if not (label_tokens & _PICKABLE_OBJECT_KEYWORDS):
                continue
            _add(entry)
            if len(selected) >= limit:
                return selected
        # Second pass: broaden to other plausible pick candidates.
        for entry in entries:
            if not _is_plausible_pick_candidate(entry):
                continue
            _add(entry)
            if len(selected) >= limit:
                return selected

    for entry in entries:
        _add(entry)
        if len(selected) >= limit:
            return selected

    return selected


def _build_interiorgs_prompt_tasks(
    manipulation_candidates: List[dict],
    articulation_hints: List[dict],
    navigation_hints: List[dict],
    scene_type: str,
) -> List[dict]:
    """Build task list with both category IDs and explicit natural-language prompts."""
    tasks: List[dict] = []

    if manipulation_candidates:
        tasks.append(
            {"task_id": "pick_place_manipulation", "source": "interiorgs", "scene_type": scene_type}
        )
    if articulation_hints:
        tasks.append(
            {"task_id": "open_close_access_points", "source": "interiorgs", "scene_type": scene_type}
        )
    if not tasks:
        tasks.append(
            {"task_id": "pick_place_manipulation", "source": "fallback", "scene_type": scene_type}
        )

    for entry in _select_prompt_entries(
        manipulation_candidates,
        limit=16,
        pick_filter=True,
        max_per_family=2,
    ):
        token = _prompt_token_from_entry(entry)
        tasks.append(
            {
                "task_id": f"Pick up {token} and place it in the target zone",
                "source": "interiorgs_prompt",
                "scene_type": scene_type,
            }
        )

    for entry in _select_prompt_entries(
        articulation_hints,
        limit=10,
        max_per_family=2,
    ):
        token = _prompt_token_from_entry(entry)
        tasks.append(
            {
                "task_id": f"Open and close {token}",
                "source": "interiorgs_prompt",
                "scene_type": scene_type,
            }
        )

    # Add explicit "toggle" tasks for appliances/switches where appropriate.
    toggle_pool = list(manipulation_candidates) + list(articulation_hints)
    for entry in _select_prompt_entries(
        toggle_pool,
        limit=6,
        max_per_family=1,
    ):
        tokens = set(_label_tokens(str(entry.get("label") or "")))
        if not (tokens & _TOGGLEABLE_KEYWORDS):
            continue
        token = _prompt_token_from_entry(entry)
        tasks.append(
            {
                "task_id": f"Turn on {token} and then turn it off",
                "source": "interiorgs_prompt",
                "scene_type": scene_type,
            }
        )

    for entry in _select_prompt_entries(
        navigation_hints,
        limit=4,
        max_per_family=1,
    ):
        label = str(entry.get("label") or "target area").strip().lower().replace("_", " ")
        tasks.append(
            {
                "task_id": f"Navigate to the {label}",
                "source": "interiorgs_prompt",
                "scene_type": scene_type,
            }
        )

    deduped: List[dict] = []
    seen_prompts: set = set()
    for task in tasks:
        prompt_key = str(task.get("task_id") or "").strip().lower()
        if not prompt_key or prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        deduped.append(task)
    return deduped


def ingest_interiorgs(
    labels_path: Path,
    structure_path: Optional[Path],
    facility: FacilityConfig,
) -> dict:
    """Parse InteriorGS labels.json (+ optionally structure.json) into task hints.

    InteriorGS coordinate system: XYZ = (Right, Back, Up), Z-up, metres.
    No scene transform is required — the pipeline already assumes Z-up.

    Priority of object sources:
      1. objects in labels.json  (ground-truth per-instance OBBs)
      2. holes (DOOR / WINDOW) in structure.json  (architectural features)
      3. ins entries in structure.json  (coarser fallback object list)

    ins entries are only added when their label is not already represented
    by a labels.json object of the same semantic class.
    """
    labels_data = read_json(labels_path)
    structure_data = read_json(structure_path) if structure_path else {}

    manipulation_candidates: List[dict] = []
    articulation_hints: List[dict] = []
    navigation_hints: List[dict] = []

    # Track which semantic classes are already covered by labels.json so we
    # don't double-count coarser structure.json ins entries.
    covered_classes: set = set()

    # ------------------------------------------------------------------
    # 1. labels.json objects (ground-truth OBBs)
    #
    # Two formats are supported:
    #   A) Actual InteriorGS on-disk format — labels_data is a *list* where
    #      each item has keys: ins_id, label, bounding_box (list of {x,y,z} dicts)
    #   B) Normalised / test format — labels_data is a *dict* with an "objects"
    #      key; each item has: instance_id, semantic_label, bounding_box (list
    #      of [x,y,z] lists)
    # ------------------------------------------------------------------
    if isinstance(labels_data, list):
        # Format A: actual InteriorGS file
        raw_objects: List[dict] = labels_data
        _get_iid = lambda o: str(o.get("ins_id", "")).strip()
        _get_label = lambda o: str(o.get("label", "unknown")).strip()
    else:
        # Format B: normalised / test dict
        raw_objects = labels_data.get("objects", [])
        _get_iid = lambda o: str(o.get("instance_id", "")).strip()
        _get_label = lambda o: str(o.get("semantic_label", "unknown")).strip()

    for obj in raw_objects:
        iid = _get_iid(obj)
        label = _get_label(obj)
        corners = obj.get("bounding_box")
        if not corners:
            continue
        bbox = _obb_from_corners(corners)
        if bbox is None:
            logger.debug("Skipping malformed bounding_box for instance %s", iid)
            continue

        category = _infer_category_from_label(label)
        entry = {
            "instance_id": iid or f"igs_{label}",
            "label": label,
            "category": category,
            "confidence": 1.0,
            "source": "interiorgs_labels",
            "boundingBox": bbox,
        }
        class_key = _semantic_class_key(label)
        if class_key:
            covered_classes.add(class_key)

        if category == "articulation":
            articulation_hints.append(entry)
        elif category == "navigation":
            navigation_hints.append(entry)
        else:
            manipulation_candidates.append(entry)

    # ------------------------------------------------------------------
    # 2. structure.json holes (architectural doors / windows)
    # ------------------------------------------------------------------
    for hole in structure_data.get("holes", []):
        hole_type = str(hole.get("type", "")).upper()
        # OPENING is a passthrough arch feature — treat as door for task purposes.
        if hole_type not in {"DOOR", "WINDOW", "OPENING"}:
            continue
        profile = hole.get("profile")
        if not profile:
            continue
        bbox = _obb_from_corners(profile)
        if bbox is None:
            continue

        label = "window" if hole_type == "WINDOW" else "door"
        center_key = [round(v, 3) for v in bbox["center"]]
        extent_key = [round(v, 3) for v in bbox["extents"]]
        stable_key = f"{label}|{center_key}|{extent_key}".encode("utf-8")
        hole_id = hashlib.sha1(stable_key).hexdigest()[:8]
        entry = {
            "instance_id": f"igs_hole_{label}_{hole_id}",
            "label": label,
            "category": "articulation",
            "confidence": 0.9,
            "source": "interiorgs_structure",
            "boundingBox": bbox,
        }
        articulation_hints.append(entry)

    # ------------------------------------------------------------------
    # 3. structure.json ins (coarser fallback — skip if label already seen)
    # ------------------------------------------------------------------
    for ins in structure_data.get("ins", []):
        label = str(ins.get("label", "unknown")).strip()
        class_key = _semantic_class_key(label)
        if class_key and class_key in covered_classes:
            continue
        position = ins.get("position")
        size = ins.get("size")
        if not position or not size:
            continue
        bbox = _obb_from_position_size(position, size)
        if bbox is None:
            continue

        category = _infer_category_from_label(label)
        entry = {
            "instance_id": f"igs_ins_{label}",
            "label": label,
            "category": category,
            "confidence": 0.8,
            "source": "interiorgs_structure",
            "boundingBox": bbox,
        }
        if class_key:
            covered_classes.add(class_key)

        if category == "articulation":
            articulation_hints.append(entry)
        elif category == "navigation":
            navigation_hints.append(entry)
        else:
            manipulation_candidates.append(entry)

    # ------------------------------------------------------------------
    # Derive tasks from what we found
    # ------------------------------------------------------------------
    scene_type = _interiorgs_scene_type(structure_data)
    tasks = _build_interiorgs_prompt_tasks(
        manipulation_candidates=manipulation_candidates,
        articulation_hints=articulation_hints,
        navigation_hints=navigation_hints,
        scene_type=scene_type,
    )

    num_m = len(manipulation_candidates)
    num_a = len(articulation_hints)
    num_n = len(navigation_hints)
    logger.info(
        "InteriorGS ingestion: %d manipulation, %d articulation, %d navigation (scene_type=%s)",
        num_m,
        num_a,
        num_n,
        scene_type,
    )

    if num_m + num_a + num_n == 0:
        raise ValueError(
            "InteriorGS metadata present but produced no usable object hints"
        )

    # InteriorGS is Z-up — scene_transform is identity.
    return {
        "bootstrap_generated": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facility_name": facility.name,
        "source": "interiorgs",
        "scene_type": scene_type,
        "tasks": tasks,
        "manipulation_candidates": manipulation_candidates,
        "articulation_hints": articulation_hints,
        "navigation_hints": navigation_hints,
        "scene_transform": np.eye(4, dtype=np.float64).tolist(),
        "resolved_up_axis": "z",
        "interiorgs_labels_path": str(labels_path),
    }


# ---------------------------------------------------------------------------
# Task derivation (scene-agnostic)
# ---------------------------------------------------------------------------

_CATEGORY_TO_TASK = {
    "manipulation": "pick_place_manipulation",
    "articulation": "open_close_access_points",
    # "navigation" has no downstream prompt mapping — intentionally omitted so
    # navigation detections don't emit unrecognized task IDs.
}


def _derive_tasks_from_detections(
    detections: List[DetectedRegion],
    suggested_tasks: List[dict],
    scene_type: str,
) -> List[dict]:
    """Derive task list from VLM detections and suggestions.

    Priority:
      1. VLM-suggested tasks (deduped by task description)
      2. Tasks inferred from detection categories
      3. Fallback ``pick_place_manipulation``
    """
    tasks: List[dict] = []
    seen: set = set()

    # 1. VLM-suggested tasks
    for st in suggested_tasks:
        desc = st.get("suggested_task") or st.get("task")
        if desc and desc not in seen:
            tasks.append({"task_id": desc, "source": "vlm_suggestion", "scene_type": scene_type})
            seen.add(desc)

    # 2. Category-inferred tasks
    categories_present = {d.category for d in detections}
    for cat in sorted(categories_present):
        task_id = _CATEGORY_TO_TASK.get(cat)
        if not task_id:
            continue
        if task_id not in seen:
            tasks.append({"task_id": task_id, "source": "category_inference", "scene_type": scene_type})
            seen.add(task_id)

    # 3. Fallback
    if not tasks:
        tasks.append({"task_id": "pick_place_manipulation", "source": "fallback", "scene_type": scene_type})

    return tasks


# ---------------------------------------------------------------------------
# Inverse-frame helper
# ---------------------------------------------------------------------------


def _to_original_frame(points_corrected: np.ndarray, scene_T: np.ndarray) -> np.ndarray:
    """Map points from corrected (Z-up) frame back to native PLY frame."""
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = scene_T[:3, :3].T
    return transform_means(points_corrected, T_inv)


def _extents_to_original_frame(extents_corrected: np.ndarray, scene_T: np.ndarray) -> np.ndarray:
    """Map axis-aligned extents from corrected frame back to native PLY frame."""
    if extents_corrected.size == 0:
        return extents_corrected
    # Convert lengths under rotation by projecting through |R|.
    r_inv = scene_T[:3, :3].T
    return (np.abs(r_inv) @ extents_corrected.T).T


# ---------------------------------------------------------------------------
# Synthetic task hints builder
# ---------------------------------------------------------------------------


def _build_synthetic_task_hints(
    centers_original: np.ndarray,
    source: str,
    facility: FacilityConfig,
    scene_T: np.ndarray,
    tasks: List[dict],
    scene_type: str = "unknown",
    labels: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    extents: Optional[np.ndarray] = None,
) -> Dict:
    """Build the task_targets.synthetic.json payload.

    Accepts per-object labels, categories, and extents from VLM detections
    or cluster analysis. Falls back to generic values when not provided.
    """
    n = len(centers_original)
    if labels is None:
        labels = [f"target_{i + 1:02d}" for i in range(n)]
    if categories is None:
        categories = ["manipulation"] * n
    if extents is None:
        extents = np.full((n, 3), 0.35, dtype=np.float64)

    manipulation_candidates = []
    articulation_hints = []
    navigation_hints = []

    for idx, (center, label, cat, ext) in enumerate(
        zip(centers_original, labels, categories, extents)
    ):
        entry = {
            "instance_id": f"bootstrap_{label}",
            "label": label,
            "category": cat,
            "confidence": 0.6,
            "source": source,
            "boundingBox": {
                "center": np.asarray(center, dtype=float).tolist(),
                "extents": np.asarray(ext, dtype=float).tolist(),
                "axes": np.eye(3, dtype=np.float64).tolist(),
            },
        }

        if cat == "articulation":
            articulation_hints.append(entry)
        elif cat == "navigation":
            navigation_hints.append(entry)
        else:
            manipulation_candidates.append(entry)

    return {
        "bootstrap_generated": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "facility_name": facility.name,
        "source": source,
        "scene_type": scene_type,
        "tasks": tasks,
        "manipulation_candidates": manipulation_candidates,
        "articulation_hints": articulation_hints,
        "navigation_hints": navigation_hints,
        "scene_transform": scene_T.tolist(),
    }


# ---------------------------------------------------------------------------
# Stage implementation
# ---------------------------------------------------------------------------


class TaskHintsBootstrapStage(PipelineStage):
    @property
    def name(self) -> str:
        return "s0_task_hints_bootstrap"

    @property
    def description(self) -> str:
        return "Bootstrap synthetic task_targets.json from first-pass VLM detections"

    def run(
        self,
        config: ValidationConfig,
        facility: FacilityConfig,
        work_dir: Path,
        previous_results: Dict[str, StageResult],
    ) -> StageResult:
        del previous_results

        if facility.task_hints_path is not None and facility.task_hints_path.exists():
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                outputs={"task_hints_path": str(facility.task_hints_path)},
                detail="facility.task_hints_path already exists",
            )

        output_path = facility.task_hints_path or _default_output_path(work_dir)
        if output_path.exists():
            facility.task_hints_path = output_path
            existing = read_json(output_path)
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0,
                outputs={"task_hints_path": str(output_path)},
                metrics={
                    "bootstrap_reused": True,
                    "num_candidates": len(existing.get("manipulation_candidates", [])),
                },
            )

        # --- Priority 0: InteriorGS direct metadata ingestion ---
        # labels.json and structure.json sit alongside the PLY file.
        # When present, skip the VLM entirely — the ground truth is already there.
        labels_path = facility.ply_path.parent / "labels.json"
        structure_path = facility.ply_path.parent / "structure.json"
        if labels_path.exists():
            logger.info(
                "InteriorGS labels.json found at %s — using direct ingestion", labels_path
            )
            try:
                payload = ingest_interiorgs(
                    labels_path=labels_path,
                    structure_path=structure_path if structure_path.exists() else None,
                    facility=facility,
                )
            except Exception:
                logger.warning("InteriorGS ingestion failed", exc_info=True)
                return StageResult(
                    stage_name=self.name,
                    status="failed",
                    elapsed_seconds=0,
                    detail=(
                        "InteriorGS metadata was found but no usable hints could be derived. "
                        "Manual analysis is required: generate task_targets.json (or "
                        "task_targets.synthetic.json) and set facility.task_hints_path."
                    ),
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(payload, output_path)
            facility.task_hints_path = output_path
            return StageResult(
                stage_name=self.name,
                status="success",
                elapsed_seconds=0,
                outputs={"task_hints_path": str(output_path)},
                metrics={
                    "bootstrap_reused": False,
                    "num_candidates": len(payload["manipulation_candidates"]),
                    "num_articulation": len(payload["articulation_hints"]),
                    "num_navigation": len(payload.get("navigation_hints", [])),
                    "source": "interiorgs",
                    "scene_type": payload["scene_type"],
                    "resolved_up_axis": "z",
                },
            )

        if not config.render.vlm_fallback:
            return StageResult(
                stage_name=self.name,
                status="skipped",
                elapsed_seconds=0,
                detail="render.vlm_fallback=false and no task hints present",
            )

        if not facility.ply_path.exists():
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"PLY file not found: {facility.ply_path}",
            )

        # --- Load PLY with colors ---
        means_raw, colors = load_ply_means_and_colors_numpy(facility.ply_path)
        if len(means_raw) == 0:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=f"No points loaded from PLY: {facility.ply_path}",
            )

        # --- Resolve up-axis and compute scene transform ---
        resolved_facility = facility
        if facility.up_axis.lower().strip() == "auto":
            detected = detect_up_axis(means_raw)
            resolved_facility = replace(facility, up_axis=detected)
        else:
            detected = facility.up_axis

        scene_T = compute_scene_transform(resolved_facility)
        has_transform = not is_identity_transform(scene_T)
        if not has_transform:
            means_corrected = means_raw
        else:
            means_corrected = transform_means(means_raw, scene_T)
        scene_center = means_corrected.mean(axis=0)

        # --- VLM detection (with colors) ---
        detections: List[DetectedRegion] = []
        scene_type = "unknown"
        suggested_tasks: List[dict] = []
        source = "vlm"

        try:
            vlm_result = detect_and_generate_specs(
                splat_means_np=means_corrected,
                scene_center=scene_center,
                num_views=config.render.vlm_fallback_num_views,
                model=config.render.vlm_fallback_model,
                resolution=config.render.resolution,
                splat_colors=colors,
            )
            detections = vlm_result.detections
            scene_type = vlm_result.scene_type
            suggested_tasks = vlm_result.suggested_tasks
        except Exception:
            logger.warning("VLM bootstrap detection failed", exc_info=True)
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=_MANUAL_ANALYSIS_DETAIL,
            )

        if not vlm_result.specs:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=_MANUAL_ANALYSIS_DETAIL,
            )
        specs = vlm_result.specs

        approach_points_corr: List[List[float]] = [
            list(spec.approach_point)
            for spec in specs
            if spec.approach_point is not None
        ]
        if not approach_points_corr:
            return StageResult(
                stage_name=self.name,
                status="failed",
                elapsed_seconds=0,
                detail=_MANUAL_ANALYSIS_DETAIL,
            )

        # --- Build per-object metadata ---
        n = len(approach_points_corr)
        if source == "vlm" and detections:
            labels = [d.label for d in detections[:n]]
            categories = [d.category for d in detections[:n]]
            obj_extents = np.array([d.extents_3d for d in detections[:n]])
            # Pad if fewer detections than specs (shouldn't happen, but defensive)
            while len(labels) < n:
                labels.append(f"target_{len(labels) + 1:02d}")
                categories.append("manipulation")
            if len(obj_extents) < n:
                pad = np.full((n - len(obj_extents), 3), 0.35)
                obj_extents = np.vstack([obj_extents, pad])
        else:
            labels = None
            categories = None
            obj_extents = None

        # --- Derive tasks ---
        tasks = _derive_tasks_from_detections(detections, suggested_tasks, scene_type)

        # --- Convert centers back to original PLY frame ---
        centers_corrected = np.asarray(approach_points_corr, dtype=np.float64)
        if not has_transform:
            centers_original = centers_corrected
        else:
            centers_original = _to_original_frame(centers_corrected, scene_T)

        if obj_extents is not None and has_transform:
            obj_extents = _extents_to_original_frame(np.asarray(obj_extents, dtype=np.float64), scene_T)

        payload = _build_synthetic_task_hints(
            centers_original=centers_original,
            source=source,
            facility=facility,
            scene_T=scene_T,
            tasks=tasks,
            scene_type=scene_type,
            labels=labels,
            categories=categories,
            extents=obj_extents,
        )
        payload["resolved_up_axis"] = resolved_facility.up_axis
        if facility.up_axis.lower().strip() == "auto":
            payload["detected_up_axis"] = detected
        write_json(payload, output_path)
        facility.task_hints_path = output_path

        num_manip = len(payload["manipulation_candidates"])
        num_artic = len(payload["articulation_hints"])
        num_nav = len(payload.get("navigation_hints", []))

        return StageResult(
            stage_name=self.name,
            status="success",
            elapsed_seconds=0,
            outputs={
                "task_hints_path": str(output_path),
            },
            metrics={
                "bootstrap_reused": False,
                "num_candidates": num_manip,
                "num_articulation": num_artic,
                "num_navigation": num_nav,
                "source": source,
                "scene_type": scene_type,
                "resolved_up_axis": resolved_facility.up_axis,
            },
        )
