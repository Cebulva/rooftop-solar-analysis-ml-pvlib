"""
Inquiry Manager for Solar Analysis Pipeline.
Handles persistent storage of analysis sessions with unique IDs.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import cv2


def get_inquiries_dir() -> Path:
    """Get the inquiries storage directory, creating if needed."""
    # Get the project root (parent of src/)
    project_root = Path(__file__).parent.parent
    inquiries_dir = project_root / "data" / "inquiries"
    inquiries_dir.mkdir(parents=True, exist_ok=True)
    return inquiries_dir


def get_index_path() -> Path:
    """Get path to the master index file."""
    return get_inquiries_dir() / "index.json"


def load_index() -> Dict[str, Any]:
    """Load the master index file."""
    index_path = get_index_path()
    if index_path.exists():
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # Return default structure if file doesn't exist or is corrupted
    return {"last_id": 0, "inquiries": {}}


def save_index(index: Dict[str, Any]) -> None:
    """Save the master index file."""
    index_path = get_index_path()
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def generate_new_id() -> str:
    """Generate the next inquiry ID (INQ-001, INQ-002, etc.)."""
    index = load_index()
    new_num = index["last_id"] + 1
    return f"INQ-{new_num:03d}"


def create_inquiry(lat: Optional[float] = None, lon: Optional[float] = None,
                   address: Optional[str] = None) -> str:
    """
    Create a new inquiry and return its ID.

    Args:
        lat: Optional latitude for initial summary
        lon: Optional longitude for initial summary
        address: Optional address string for summary

    Returns:
        New inquiry ID (e.g., "INQ-001")
    """
    index = load_index()
    new_num = index["last_id"] + 1
    inquiry_id = f"INQ-{new_num:03d}"

    # Create inquiry folder
    inquiry_dir = get_inquiries_dir() / inquiry_id
    inquiry_dir.mkdir(parents=True, exist_ok=True)

    # Add to index
    now = datetime.now().isoformat()
    index["last_id"] = new_num
    index["inquiries"][inquiry_id] = {
        "created": now,
        "modified": now,
        "address": address or "",
        "lat": lat,
        "lon": lon,
        "current_step": 1,
        "system_kwp": None
    }
    save_index(index)

    return inquiry_id


def inquiry_exists(inquiry_id: str) -> bool:
    """Check if an inquiry ID exists."""
    index = load_index()
    return inquiry_id in index.get("inquiries", {})


def get_inquiry_dir(inquiry_id: str) -> Path:
    """Get the directory for a specific inquiry."""
    return get_inquiries_dir() / inquiry_id


def _save_image(path: Path, image: np.ndarray) -> bool:
    """
    Save a BGR numpy array as PNG file.

    Args:
        path: Path to save the image
        image: BGR numpy array (OpenCV format)

    Returns:
        True if successful, False otherwise
    """
    try:
        if image is None:
            return False
        if not isinstance(image, np.ndarray):
            return False
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        cv2.imwrite(str(path), image)
        return True
    except Exception:
        return False


def _load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load a PNG file as BGR numpy array.

    Args:
        path: Path to the image file

    Returns:
        BGR numpy array or None if failed
    """
    try:
        if not path.exists():
            return None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return image
    except Exception:
        return None


def _is_numpy_array(obj: Any) -> bool:
    """Check if object is a numpy array."""
    return isinstance(obj, np.ndarray)


def _serialize_data(data: Dict[str, Any], inquiry_dir: Path) -> Dict[str, Any]:
    """
    Serialize session data for JSON storage.
    Saves numpy arrays as image files and returns JSON-safe dict.

    Args:
        data: The session_state.data dictionary
        inquiry_dir: Directory to save images

    Returns:
        JSON-serializable dictionary with image paths
    """
    serialized = {}
    images_saved = {}

    # Image fields to extract and save
    image_fields = {
        "full_img": "full_img.png",
        "pdf_panel_image": "panel_image.png",
    }

    for key, value in data.items():
        # Handle direct image fields
        if key in image_fields:
            if _is_numpy_array(value):
                img_path = inquiry_dir / image_fields[key]
                if _save_image(img_path, value):
                    images_saved[key] = image_fields[key]
            continue

        # Handle 'res' dict which contains zoom_img
        if key == "res" and isinstance(value, dict):
            res_serialized = {}
            for res_key, res_value in value.items():
                if res_key == "zoom_img" and _is_numpy_array(res_value):
                    img_path = inquiry_dir / "zoom_img.png"
                    if _save_image(img_path, res_value):
                        images_saved["zoom_img"] = "zoom_img.png"
                elif _is_numpy_array(res_value):
                    # Skip other numpy arrays in res
                    continue
                else:
                    # Convert lists/tuples to ensure JSON serializable
                    res_serialized[res_key] = _convert_to_serializable(res_value)
            serialized[key] = res_serialized
            continue

        # Handle solar_results which might have numpy arrays
        if key == "solar_results" and isinstance(value, dict):
            solar_serialized = {}
            for sr_key, sr_value in value.items():
                if _is_numpy_array(sr_value):
                    continue  # Skip numpy arrays in solar_results
                else:
                    solar_serialized[sr_key] = _convert_to_serializable(sr_value)
            serialized[key] = solar_serialized
            continue

        # Regular fields - convert to JSON serializable
        serialized[key] = _convert_to_serializable(value)

    # Add images mapping
    serialized["_images"] = images_saved

    return serialized


def _convert_to_serializable(obj: Any) -> Any:
    """Convert an object to JSON-serializable format."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    # For other types, try to convert to string
    try:
        return str(obj)
    except Exception:
        return None


def _deserialize_data(metadata: Dict[str, Any], inquiry_dir: Path) -> Dict[str, Any]:
    """
    Deserialize saved data back to session format.
    Loads image files back as numpy arrays.

    Args:
        metadata: The saved metadata dictionary
        inquiry_dir: Directory containing saved images

    Returns:
        Dictionary suitable for session_state.data
    """
    data = {}
    images_mapping = metadata.pop("_images", {})

    # Image field mapping (metadata key -> session key)
    image_restore = {
        "full_img": "full_img",
        "pdf_panel_image": "pdf_panel_image",
        "zoom_img": None,  # Handled in 'res' dict
    }

    for key, value in metadata.items():
        if key == "res" and isinstance(value, dict):
            # Restore zoom_img into res dict
            res_restored = dict(value)
            if "zoom_img" in images_mapping:
                img_path = inquiry_dir / images_mapping["zoom_img"]
                zoom_img = _load_image(img_path)
                if zoom_img is not None:
                    res_restored["zoom_img"] = zoom_img
            data[key] = res_restored
        else:
            data[key] = value

    # Restore top-level images
    for img_key, filename in images_mapping.items():
        if img_key in image_restore and image_restore[img_key] is not None:
            img_path = inquiry_dir / filename
            img = _load_image(img_path)
            if img is not None:
                data[image_restore[img_key]] = img

    return data


def save_inquiry(inquiry_id: str, data: Dict[str, Any], step: int,
                 sub_step: str = "verify", address: Optional[str] = None) -> bool:
    """
    Save inquiry data to disk.

    Args:
        inquiry_id: The inquiry ID (e.g., "INQ-001")
        data: The session_state.data dictionary
        step: Current step number
        sub_step: Current sub-step string
        address: Optional address to store in index

    Returns:
        True if successful, False otherwise
    """
    try:
        inquiry_dir = get_inquiry_dir(inquiry_id)
        inquiry_dir.mkdir(parents=True, exist_ok=True)

        # Serialize data (saves images, returns JSON-safe dict)
        serialized = _serialize_data(data, inquiry_dir)

        # Add metadata
        serialized["inquiry_id"] = inquiry_id
        serialized["step"] = step
        serialized["sub_step"] = sub_step
        serialized["saved_at"] = datetime.now().isoformat()

        # Save metadata.json
        metadata_path = inquiry_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        # Update index
        index = load_index()
        if inquiry_id in index["inquiries"]:
            index["inquiries"][inquiry_id]["modified"] = datetime.now().isoformat()
            index["inquiries"][inquiry_id]["current_step"] = step

            # Update location if available
            if "confirmed_lat" in data:
                index["inquiries"][inquiry_id]["lat"] = data["confirmed_lat"]
            if "confirmed_lon" in data:
                index["inquiries"][inquiry_id]["lon"] = data["confirmed_lon"]
            if address:
                index["inquiries"][inquiry_id]["address"] = address

            # Update system size if available
            if "solar_results" in data and isinstance(data["solar_results"], dict):
                system_kwp = data["solar_results"].get("system_kwp")
                if system_kwp is not None:
                    index["inquiries"][inquiry_id]["system_kwp"] = system_kwp

            save_index(index)

        return True
    except Exception as e:
        print(f"Error saving inquiry {inquiry_id}: {e}")
        return False


def load_inquiry(inquiry_id: str) -> Optional[Dict[str, Any]]:
    """
    Load inquiry data from disk.

    Args:
        inquiry_id: The inquiry ID to load

    Returns:
        Dictionary with data, step, and sub_step, or None if failed
    """
    try:
        inquiry_dir = get_inquiry_dir(inquiry_id)
        metadata_path = inquiry_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Extract step info
        step = metadata.pop("step", 1)
        sub_step = metadata.pop("sub_step", "verify")
        metadata.pop("inquiry_id", None)
        metadata.pop("saved_at", None)

        # Deserialize (loads images back as numpy arrays)
        data = _deserialize_data(metadata, inquiry_dir)

        return {
            "data": data,
            "step": step,
            "sub_step": sub_step
        }
    except Exception as e:
        print(f"Error loading inquiry {inquiry_id}: {e}")
        return None


def list_inquiries() -> List[Dict[str, Any]]:
    """
    List all saved inquiries with summary info.

    Returns:
        List of inquiry summaries, sorted by modification date (newest first)
    """
    index = load_index()
    inquiries = []

    for inquiry_id, info in index.get("inquiries", {}).items():
        summary = {
            "id": inquiry_id,
            "created": info.get("created", ""),
            "modified": info.get("modified", ""),
            "address": info.get("address", ""),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "current_step": info.get("current_step", 1),
            "system_kwp": info.get("system_kwp"),
        }
        inquiries.append(summary)

    # Sort by modified date, newest first
    inquiries.sort(key=lambda x: x.get("modified", ""), reverse=True)

    return inquiries


def get_inquiry_summary(inquiry_id: str) -> Optional[Dict[str, Any]]:
    """
    Get summary info for a specific inquiry.

    Args:
        inquiry_id: The inquiry ID

    Returns:
        Summary dictionary or None if not found
    """
    index = load_index()
    if inquiry_id in index.get("inquiries", {}):
        info = index["inquiries"][inquiry_id]
        return {
            "id": inquiry_id,
            "created": info.get("created", ""),
            "modified": info.get("modified", ""),
            "address": info.get("address", ""),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "current_step": info.get("current_step", 1),
            "system_kwp": info.get("system_kwp"),
        }
    return None


def delete_inquiry(inquiry_id: str) -> bool:
    """
    Delete an inquiry and its data.

    Args:
        inquiry_id: The inquiry ID to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        import shutil

        # Remove folder
        inquiry_dir = get_inquiry_dir(inquiry_id)
        if inquiry_dir.exists():
            shutil.rmtree(inquiry_dir)

        # Remove from index
        index = load_index()
        if inquiry_id in index.get("inquiries", {}):
            del index["inquiries"][inquiry_id]
            save_index(index)

        return True
    except Exception as e:
        print(f"Error deleting inquiry {inquiry_id}: {e}")
        return False