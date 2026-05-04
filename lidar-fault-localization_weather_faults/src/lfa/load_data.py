import os
import numpy as np
from pathlib import Path

from lfi.apply_fault_model import apply_fault_model


# Utility function to read point cloud data
def get_dir(sequence: str, oc_type: str, oc_level: str = "cover") -> str:
    """
    Retrieves the file paths for a given LIDAROC occlusion and reference.
    Picks a random subdirectory from a given occlusion and severity type, if multiple exist.
    """
    mes_dir = os.path.join("data/lidaroc", sequence)

    if oc_type == "cover":
        mes_dir = os.path.join(mes_dir, f"2_cover")
    elif oc_type == "mudDrop":
        mes_dir = os.path.join(mes_dir, f"4_mudDrop")
    elif oc_type == "mudUniform":
        mes_dir = os.path.join(mes_dir, f"5_mudUniform")
    elif oc_type == "dust":
        mes_dir = os.path.join(mes_dir, f"6_dust")
    else:
        raise ValueError(f"Unknown occlusion type: {oc_type}")
    
    if oc_level == "cover":
        mes_dir = os.path.join(mes_dir, "1_low")
    elif oc_level == "low":
        mes_dir = os.path.join(mes_dir, "1_low")
    elif oc_level == "mid":
        mes_dir = os.path.join(mes_dir, "2_mid")
    elif oc_level == "high":
        mes_dir = os.path.join(mes_dir, "3_high")
    else:
        raise ValueError(f"Unknown occlusion level: {oc_level}")
    
    return mes_dir

def get_subdirectories(folder_path: Path) -> list[Path]:
    return sorted(d for d in folder_path.iterdir() if d.is_dir())


def get_bin_files(subdir: Path) -> list[Path]:
    if not subdir.is_dir():
        raise ValueError(f"{subdir} ist kein Ordner")

    return sorted(
        subdir.glob("*.bin"),
        key=lambda p: int(p.stem)
    )



# Utility function to load point cloud data
def load_point_cloud(filepath: str, fault_model: str = "None") -> np.ndarray:
    """
    Robuster LiDAR-Loader:
    - Header / Null-Padding am Anfang überspringen
    - Ab dem ersten Nicht-Null-Byte die Punkte lesen
    - 4 float32 pro Punkt (x, y, z, intensity)
    - Unvollständige Punkte am Ende automatisch abschneiden
    """
    values_per_row = 4
    bytes_per_row = values_per_row * 4  # 16 Bytes pro Punkt
    dtype = np.dtype("<f4")             # little-endian float32

    # 1) Dateigröße bestimmen
    with open(filepath, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()

    # 2) Erster Nicht-Null-Byte finden
    with open(filepath, "rb") as f:
        offset = 0
        chunk_size = 4096
        found = False

        while True:
            f.seek(offset)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for i, b in enumerate(chunk):
                if b != 0:
                    offset += i
                    found = True
                    break
            if found:
                break
            offset += len(chunk)

    if not found or offset >= file_size:
        # keine Daten
        return np.empty((0, values_per_row), dtype=np.float32)

    # 3) Nutzdatenlänge bestimmen und Tail auf Vielfaches von 16 Bytes kürzen
    usable_bytes = file_size - offset
    usable_bytes -= usable_bytes % bytes_per_row

    if usable_bytes <= 0:
        return np.empty((0, values_per_row), dtype=np.float32)

    # 4) Punkte laden
    with open(filepath, "rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype, count=usable_bytes // 4)

    data = data.reshape(-1, values_per_row)

    # 5) Optional Fault Model anwenden
    if fault_model != "None":
        data = apply_fault_model(data, fault_model)

    return data

