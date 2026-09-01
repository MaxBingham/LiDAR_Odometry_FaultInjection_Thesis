# Lidar Fault Localization — Weather Faults Pipeline

Dieses Repository enthält die vollständige Pipeline zur Evaluation von LiDAR-basierter Lokalisierung unter simulierten Wetterfehlern.
Die Pipeline verbindet Fehlerinjektionsmodelle (Nebel, Regen) mit dem KISS-ICP Odometrie-Algorithmus und der EVO Pose-Error-Evaluation auf KITTI-Odometrie-Sequenzen. Der Intensitätskanal wird verworfen — es werden ausschließlich XYZ-Koordinaten verwendet.

## Projektstruktur

```
pcmod_pipeline/
├── src/
│   ├── lfl/                         # Lokalisierungspipeline — CLI, KISS-ICP Runner, Batch-Sweeps, EVO-Metriken
│   ├── lfi/                         # Fehlerinjektion — Dispatcher und modellspezifische Simulatoren
│   │   ├── apply_fault_model.py     # Zentraler Dispatcher: leitet an das korrekte Fehlermodell weiter
│   │   └── fault_models/            # Einzelne Fehlermodell-Implementierungen
│   │       ├── FOG_Injector.py      # Nebelsimulator (Teufel et al.)
│   │       └── RAIN_Injection.py    # Regensimulator
│   └── lfa/                         # Analyse-Tooling — Chamfer/Hausdorff-Distanzen, Histogramme, SDASN
│
├── kiss_icp_modifications/          # Lokale KISS-ICP-Modifikationen (werden beim Setup angewendet)
├── data/kitti/                      # KITTI-Datensatz (nicht versioniert — separat herunterladen)
├── results/                         # Pipeline-Ausgabe (nicht versioniert)
│
├── setup.sh                         # Automatisiertes Setup-Skript (venv, Abhängigkeiten, KISS-ICP Build)
├── Makefile                         # Entwickler-Shortcuts (install, clean, rebuild, run)
├── pyproject.toml                   # Python-Projektkonfiguration und Entry Points
├── requirements.txt                 # Gepinnte Abhängigkeiten
├── README.md                        # Diese Dokumentation
└── .gitignore
```

## Kurzanleitung

### 1. Repository klonen

```bash
git clone https://github.com/MaxBingham/LiDAR_Odometry_FaultInjection_Thesis.git
cd LiDAR_Odometry_FaultInjection_Thesis/lidar-fault-localization_weather_faults
```

### 2. Setup-Skript ausführen

Das Setup-Skript übernimmt alles: Systemabhängigkeiten, Python 3.10+ Virtual Environment, pip-Pakete, KISS-ICP klonen, Modifikationen anwenden und bauen.

```bash
chmod +x setup.sh
./setup.sh
```

Unterstützte Distributionen: Ubuntu/Debian, Fedora, RHEL/CentOS, Arch Linux, openSUSE.
Falls das System nicht erkannt wird, listet das Skript die benötigten Pakete für eine manuelle Installation auf.

### 3. KITTI-Datensatz herunterladen

Den KITTI-Odometrie-Datensatz herunterladen und nach `data/kitti/` entpacken:

- Webseite: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
- `data_odometry_velodyne.zip` und `data_odometry_poses.zip` herunterladen
- Beide nach `data/kitti/` entpacken

Erwartete Verzeichnisstruktur:

```
data/kitti/
└── sequences/
    ├── 00/
    │   ├── velodyne/        # LiDAR .bin Scans
    │   └── calib.txt
    ├── 01/
    │   ...
    └── 10/
```

### 4. Umgebung aktivieren und testen

```bash
source venv/bin/activate
lfl_pipeline --help
```

## Modulbeschreibungen

| Modul | Funktion |
|-------|----------|
| `lfl` | Hauptpipeline für die Lokalisierung. Führt KISS-ICP Odometrie auf KITTI-Sequenzen aus, berechnet EVO APE/RPE-Metriken und unterstützt Batch-Sweeps über Fehlerparameter. Entry Point: `lfl_pipeline`. |
| `lfi` | Fehlerinjektionsschicht. `apply_fault_model.py` leitet an den korrekten Simulator in `fault_models/` weiter. Jeder Simulator erhält eine `(N, 3)` Punktwolke und gibt die modifizierte Version zurück. |
| `lfa` | Eigenständiges Analyse-Tooling. Vergleicht Punktwolken mittels Chamfer-Distanz, Hausdorff-Distanz, Verteilungshistogrammen und SDASN-Metriken. Entry Point: `lfa`. |
| `kiss_icp_modifications` | Lokale Modifikationen an KISS-ICP, die in die Fehlerinjektionsschicht einhaken. Werden automatisch von `setup.sh` nach `external/kiss-icp/` angewendet. |

## Nutzung

### Einzelner Lauf

```bash
# Baseline (kein Fehlermodell)
lfl_pipeline --sequence 07 --data_root data/kitti --fault_model none

# Nebel mit 50 m Sichtweite
lfl_pipeline --sequence 07 --data_root data/kitti --fault_model fog --visibility 50

# Regen mit 25 mm/h
lfl_pipeline --sequence 07 --data_root data/kitti --fault_model rain --rain_rate 25
```

| Parameter | Beschreibung |
|-----------|--------------|
| `--sequence` | KITTI-Sequenz (z. B. `00` bis `10` für Sequenzen mit Ground Truth) |
| `--data_root` | Wurzelverzeichnis der KITTI-Daten (Standard: `data/kitti`) |
| `--fault_model` | Anzuwendendes Fehlermodell: `none`, `fog` oder `rain` |
| `--visibility` | Sichtweite in Metern (erforderlich bei `fault_model=fog`) |
| `--rain_rate` | Regenrate in mm/h (erforderlich bei `fault_model=rain`) |
| `--fog_metric` | Nebel-Parametrisierung: `distance` (Standard) oder `chamfer` |
| `--visualize` | Trajektorien-Visualisierung während des Laufs aktivieren |

### Batch-Sweep

Der Runner durchläuft automatisch Parameterbereiche und protokolliert alle Metriken als CSV.

```bash
# Nebel-Sweep — Sichtweite von 30 bis 250 m in 20er-Schritten (Standardbereich)
python -m lfl.runner --fault_model fog --sequence 07 --data_root data/kitti

# Regen-Sweep — Regenrate von 10 bis 100 mm/h in 10er-Schritten (Standardbereich)
python -m lfl.runner --fault_model rain --sequence 07 --data_root data/kitti

# Benutzerdefinierter Nebelbereich (Start Ende Schritt)
python -m lfl.runner --fault_model fog --fog_range 10 200 10 --sequence 07

# Benutzerdefinierter Regenbereich
python -m lfl.runner --fault_model rain --rain_range 5 50 5 --sequence 07

# Mehrere Sequenzen
python -m lfl.runner --fault_model fog --sequences 00 02 05 07 08 --data_root data/kitti
```

### Makefile-Shortcuts

`make help` zeigt alle verfügbaren Targets:

| Target | Beschreibung |
|--------|--------------|
| `make install` | Vollständige Installation (Abhängigkeiten + KISS-ICP) |
| `make clean` | Build-Artefakte und Cache entfernen |
| `make rebuild` | Vollständiger Neuaufbau |
| `make sync` | Schnelle Synchronisation nach Branch-Wechsel |
| `make run-fog` | Nebelmodell auf Sequenz 03 ausführen (Beispiel) |
| `make run-rain` | Regenmodell auf Sequenz 03 ausführen (Beispiel) |

## Ergebnisse & Ausgabe

Jeder Pipeline-Lauf schreibt die Ergebnisse in ein Verzeichnis mit Zeitstempel unter `results/`.
Ein `latest`-Symlink zeigt immer auf den aktuellsten Lauf.

```
results/
├── latest -> 2026-02-23_13-28-33/
├── 2026-02-23_13-28-33/
│   ├── <seq>_gt_kitti.txt              # Ground-Truth-Posen (KITTI-Format)
│   ├── <seq>_poses_kitti.txt           # Geschätzte Posen (KITTI-Format)
│   ├── config.yml                      # KISS-ICP Konfigurationsschnappschuss
│   ├── result_metrics.log              # KISS-ICP Zusammenfassung
│   └── evo/
│       ├── ape.zip                     # EVO Absolute Pose Error
│       └── rpe.zip                     # EVO Relative Pose Error
│
├── localization_metrics_<run>.csv      # APE/RPE-Statistiken über einen Batch-Sweep
├── point_cloud_metrics_<run>.csv       # Punktlöschung/Rückstreuung pro Konfiguration
└── distance_metrics_<run>.csv          # Distanz-basierte Fehlerstatistiken (nur Nebel)
```

EVO-Ergebnisse visualisieren:

```bash
evo_res results/latest/evo/ape.zip -p
evo_res results/latest/evo/rpe.zip -p
```

## Neues Fehlermodell hinzufügen

Die Fehlerinjektionsschicht ist so konzipiert, dass sie einfach um neue Wetter- oder Sensorfehlermodelle erweitert werden kann.
Alle Fehlermodelle befinden sich in `src/lfi/fault_models/` und werden über den zentralen Dispatcher in `src/lfi/apply_fault_model.py` eingebunden.

### Schritt 1 — Fehlermodell-Modul erstellen

Eine neue Python-Datei in `src/lfi/fault_models/` anlegen, z. B. `snow.py`.
Das Modell muss ein `(N, 3)` NumPy-Array mit XYZ-Koordinaten entgegennehmen und ein modifiziertes `(M, 3)` Array zurückgeben:

```python
# src/lfi/fault_models/snow.py
import numpy as np

class SnowSimulator:
    def __init__(self, intensity: float):
        self.intensity = float(intensity)
        self.stats = {"backscattered": 0, "modified": 0}

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Schnee-Effekt auf eine Punktwolke anwenden.

        Args:
            points: (N, 3) Array mit XYZ-Koordinaten

        Returns:
            Modifizierte (M, 3) Punktwolke (M <= N)
        """
        # Implementierung hier
        # Statistiken in self.stats für das Logging tracken
        return modified_points
```

### Schritt 2 — Im Dispatcher registrieren

`src/lfi/apply_fault_model.py` bearbeiten:

1. Den Simulator am Anfang der Datei importieren:

```python
from lfi.fault_models.snow import SnowSimulator
```

2. Einen neuen Branch in der `apply_fault_model()` Funktion hinzufügen, analog zu den bestehenden Nebel/Regen-Blöcken:

```python
if faultmodel == "snow":
    sim = _SIM_CACHE["snow"].get(intensity)
    if sim is None:
        sim = SnowSimulator(intensity)
        _SIM_CACHE["snow"][intensity] = sim

    result = sim.apply(points)

    _FAULT_STATS["total_frames"] += 1
    _FAULT_STATS["total_points"] += n_in
    _FAULT_STATS["total_deleted"] += n_in - result.shape[0]
    _FAULT_STATS["total_backscattered"] += sim.stats.get("backscattered", 0)
    _FAULT_STATS["total_modified"] += sim.stats.get("modified", 0)
    save_fault_stats()

    return result
```

3. Einen Cache-Eintrag für das neue Modell in `_SIM_CACHE` hinzufügen:

```python
_SIM_CACHE = {
    "fog": {},
    "rain": {},
    "snow": {},
}
```

### Schritt 3 — In der CLI verfügbar machen

Folgende Dateien bearbeiten, um das Modell über die Kommandozeile auswählbar zu machen:

- **`src/lfl/cli.py`** — Modellnamen zu den `--fault_model` Choices hinzufügen und ggf. modellspezifische CLI-Argumente ergänzen (z. B. `--snow_intensity`).
- **`src/lfl/runner.py`** — Modell zu den Runner-Choices hinzufügen und einen Standard-Parameterbereich für den Sweep definieren.
- **`src/lfl/lfl_pipeline.py`** — Die neuen CLI-Argumente an `run_kiss_icp()` durchreichen.

### Schritt 4 — Testen

```bash
# Einzelner Lauf mit dem neuen Modell
lfl_pipeline --sequence 07 --data_root data/kitti --fault_model snow --snow_intensity 0.5

# Prüfen, ob die Ergebnisse korrekt geschrieben wurden
ls results/latest/
```

## Zusammenarbeit

Bei der Entwicklung neuer Fehler- oder Sensormodelle sollte Folgendes beachtet werden:

1. Modell-Implementierung in `src/lfi/fault_models/` anlegen.
2. Im Dispatcher (`apply_fault_model.py`) und in der CLI (`cli.py`, `runner.py`) registrieren.
3. Kurze Beschreibung in dieser README unter der Modulbeschreibungen-Tabelle hinzufügen.
4. Änderungen per Pull-Request einreichen.
