# LiDAR Fog Validation – MCAP Architecture

> **Experimental, non-portable path:** this subtree requires real-fog MCAP
> recordings that are not public. Use `lidar-fault-localization_weather_faults/`
> for the canonical KITTI odometry pipeline.

Dieses Repository enthält die **Python-basierte Validierungs-Pipeline** für das synthetische Nebelmodell.  
Die Pipeline lädt **MCAP-Aufnahmen** (sauber vs. realer Nebel), modifiziert die sauberen Scans zu **Nebel Scans** und vergleicht **Verteilungen** sowie optional **geometrische und Intensitätsmetriken**.

**Wichtig:** Auf diesem Branch wird **keine Odometrie** (z. B. KISS-ICP) ausgeführt. Dafür existiert ein **separater Branch**; hier geht es um das laden von MCAP Dateien, den Vergleich von sauberen und verschmutzten LiDAR Scans **inkl. Intensität**, die in der KISS-ICP-Pipeline nicht ausgewertet wird.

Ergebnisgrößen u. a.: Extinktionskoeffizient, Survival-/Deletion-Kurven, Intensitätsverhältnisse nach Distanz, Chamfer/Hausdorff (optional, mit Keyframes).

---

## Projektstruktur

```
lidar-fault-localization_feature-mcap/
├── matched1.yaml … matched5.yaml    # Ausgewählte Sequenzen & Pfade pro Fahrszenario
├── keyframes.yaml                   # (optional) Kurzform für ein Match
├── data/mcap/                       # MCAP-Dateien hier ablegen oder verlinken
├── src/
│   ├── lfa/mcap_validation/         # Validierung: Loader, Metriken, CLI
│   ├── lfi/                         # Fehlerinjektion (Nebel, Regen)
│   └── lfl/                         # Experiment-Runner (KITTI/Odometrie; hier nicht genutzt)
├── kiss_icp_modifications/          # angepasstes KISS-ICP (für lfl / andere Branches)
├── results/                         # Ausgabe (git-ignoriert)
├── setup.sh                         # Linux: System-Deps + venv + pip (optional)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Kurzanleitung zur Installation

### 1. Repository klonen und Branch auschecken

```bash
git clone https://github.com/MaxBingham/LiDAR_Odometry_FaultInjection_Thesis.git
cd LiDAR_Odometry_FaultInjection_Thesis/lidar-fault-localization_feature-mcap
```

### 2. Python-Umgebung einrichten

**Option A – automatisiert (Linux, inkl. vieler System-Pakete):**

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Option B – manuell:**

```bash
python3 -m venv venv
source venv/bin/activate   # unter Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Die MCAP-relevanten Pakete (`mcap`, `mcap-ros2-support` usw.) stehen in `requirements.txt`.

### 3. Daten und Pfade anpassen

Die MCAP-Aufnahmen liegen **nicht** im Repository (zu groß, via `.gitignore` ausgeschlossen).
Auf einem neuen Rechner müssen folgende Pfade angepasst werden:

1. **`matched1.yaml` … `matched5.yaml`** — jeweils unter `paths:` die beiden absoluten Pfade zu den MCAP-Dateien (`clean` und `real_fog`). Die aktuellen Pfade zeigen auf ein gemountetes Laufwerk des Entwicklungsrechners:

```yaml
# Beispiel (matched1.yaml) — ANPASSEN:
paths:
  clean:    /media/ubuntu/…/Match1/2026_02_03-15-33-44_perception_raw_4.mcap
  real_fog: /media/ubuntu/…/Match1/2026_01_30-15-14-00_perception_raw_3.mcap
```

2. **MCAP-Dateien** auf die Maschine kopieren oder per Symlink einbinden, z. B. unter `data/mcap/`, und die Pfade in den YAMLs entsprechend setzen.

Alle anderen Pfade (Ausgabe, Timestamps usw.) werden zur Laufzeit erzeugt und müssen **nicht** manuell geändert werden.

### 4. Validierung ausführen

Einstiegspunkt ist das CLI-Modul `lfa.mcap_validation.cli`. Beispiel (beide Stufen, Keyframes aus YAML):

```bash
PYTHONPATH=src python -m lfa.mcap_validation.cli \
  --keyframes matched1.yaml \
  --visibility 100 \
  --output results/validation
```

Weitere Beispiele und Flags siehe Abschnitt **Nutzung (CLI)**.

---

## Externe / optionale Komponenten


| Pfad                      | Rolle auf diesem Branch                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `kiss_icp_modifications/` | Enthält das für **lfl** bzw. KITTI/Odometrie vorgesehene KISS-ICP. Für die **MCAP-Validierung nicht erforderlich**. |


Es gibt **keine** ROS-2-`colcon`-Builds und **keine** zentrale Launch-Datei; die Pipeline ist rein **Python**.

---

## Beschreibung der Hauptmodule


| Modul                 | Funktion                                                                                                                                                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lfa.mcap_validation` | Zentrale **Validierungs-Pipeline**: MCAP laden, synthetischen Nebel erzeugen (über `lfi`), Verteilungs- und Keyframe-Metriken, CSV-Ausgabe; optional Sanity-Overlays (`--sanity-checks`). Visualisierung der CSVs erfolgt in einer separaten Plotting-Pipeline. |
| `lfi`                 | **Fehlerinjektion**: physikalisches Nebelmodell (`FogSimulator`) und Regen; wird von der Validierung für synthetischen Nebel genutzt.                                                                                                                           |
| `lfl`                 | **Experiment-Runner** und Anbindung an KITTI/KISS-ICP-Pfade; für die reine MCAP-Nebelvalidierung hier **nicht** zwingend.                                                                                                                                       |


---

## Nutzung (CLI)

Die Pipeline wird typischerweise **einmal pro Match** (`matched1.yaml` … `matched5.yaml`) ausgeführt.

### Analyse-Stufen


| Flag               | Stufe   | Ausgabe (Kurz)                                                        |
| ------------------ | ------- | --------------------------------------------------------------------- |
| `--distributional` | Stufe 1 | Extinktion, Survival/Deletion, Intensitätsverhältnisse                |
| `--geometric`      | Stufe 2 | Chamfer, Hausdorff, Intensität pro Keyframe (erfordert `--keyframes`) |


Wenn **keines** der beiden Flags gesetzt ist, laufen **beide** Stufen (abwärtskompatibel).

### Beispiele

Nur Verteilungen (Stufe 1):

```bash
PYTHONPATH=src python -m lfa.mcap_validation.cli \
  --distributional \
  --keyframes matched1.yaml \
  --visibility 100 \
  --output results/validation
```

Nur Geometrie (Stufe 2):

```bash
PYTHONPATH=src python -m lfa.mcap_validation.cli \
  --geometric \
  --keyframes matched1.yaml \
  --visibility 100 \
  --output results/validation
```

Zeitliches Trimmen (z. B. `matched2.yaml`):

```bash
PYTHONPATH=src python -m lfa.mcap_validation.cli \
  --keyframes matched2.yaml \
  --visibility 140 \
  --skip_seconds_clean 16.629 --max_frames_clean 338 \
  --skip_seconds_fog 8.818 --max_frames_fog 283 \
  --output results/validation
```

### Alle fünf Matches (Parameterübersicht)


| Match                 | YAML            | Sichtweite (Beispiel) | Zusätzliche Trim-Flags                                                                             |
| --------------------- | --------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| 1 Autobahn            | `matched1.yaml` | `100`                 | —                                                                                                  |
| 2 Autobahnauffahrt    | `matched2.yaml` | `140`                 | `--skip_seconds_clean 16.629 --max_frames_clean 338 --skip_seconds_fog 8.818 --max_frames_fog 283` |
| 3 Landstraße          | `matched3.yaml` | `100`                 | —                                                                                                  |
| 4 Innerstädtisch      | `matched4.yaml` | `100`                 | —                                                                                                  |
| 5 Garching Marktplatz | `matched5.yaml` | `100`                 | —                                                                                                  |


Gleiches Kommando wie oben, nur `--keyframes` und ggf. `--visibility` / Trim-Spalte setzen.

### CLI-Referenz (wichtigste Flags)


| Flag                                                                                   | Beschreibung                                                                          |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `--output`                                                                             | Zielverzeichnis (**Pflicht**); darunter wird ein Unterordner mit Zeitstempel angelegt |
| `--keyframes`                                                                          | YAML mit Keyframes und optional `paths:` zu den MCAPs.                                |
| `--clean` / `--real_fog`                                                               | MCAP-Pfade (falls nicht in der YAML unter `paths:`)                                   |
| `--visibility`                                                                         | Sichtweite in Metern für das Nebelmodell (**Pflicht**)                                |
| `--distributional` / `--geometric`                                                     | Nur eine Stufe; ohne beide → beide Stufen                                             |
| `--fog_metric`                                                                         | `distance` oder `chamfer` (Standard: `distance`)                                      |
| `--topic`                                                                              | LiDAR-Topic (Standard: `/sensing/lidar/concatenated/pointcloud`)                      |
| `--voxel_size`                                                                         | Voxelgröße in m für geometrische Metriken (Standard: `0.2`)                           |
| `--sanity-checks`                                                                      | BEV-Overlay-Bilder (`sanity_overlays/`) in Stufe 2 erzeugen (Standard: aus)           |
| `--skip_seconds_clean`, `--skip_seconds_fog`, `--max_frames_clean`, `--max_frames_fog` | Aufnahmen zeitlich begrenzen (z. B. Match 2)                                          |


Weitere Schalter siehe `python -m lfa.mcap_validation.cli --help`.

---

## Ausgabe

Pro Lauf entsteht ein zeitgestempeltes Unterverzeichnis, z. B.:

```
results/validation/match1_data_<zeitstempel>/
├── aggregate_metrics.csv
├── range_binned_metrics.csv
├── intensity_ratio_by_distance.csv
├── keyframe_metrics_detailed.csv   # Stufe 2
├── keyframe_metrics_summary.csv    # Stufe 2
└── sanity_overlays/                # optional: --sanity-checks (Stufe 2)
```

Die CSV-Dateien sind die Eingabe für eine **separate Plotting-Pipeline** (nicht Teil dieses CLI).

---

## Matched-Sequence-YAML-Format

Jede `matchedN.yaml` enthält die Pfade zu den beiden MCAPs und Keyframes für die geometrische Stufe:

```yaml
paths:
  clean: /absolut/pfad/zu/clean.mcap
  real_fog: /absolut/pfad/zu/fog.mcap

keyframes:   
  - clean_time: [<epoch_start>, <dauer_sekunden>]
    fog_time:   [<epoch_start>, <dauer_sekunden>]
    label: beschreibung
```

---

## Zusammenarbeit

- **Neue Nebel- oder Sensormodelle:** primär unter `src/lfi/` (z. B. neue Injektoren oder Erweiterung von `apply_fault_model.py`), dann die MCAP-Seite unter `src/lfa/mcap_validation/analysis/` anbinden, falls neue Metriken nötig sind.
- **Nur Validierungslogik / Metriken:** `src/lfa/mcap_validation/`. Hier kann die Validierungslogik angepasst werden. 
- Änderungen idealerweise mit kurzer Beschreibung im Commit / PR; README bei neuen CLI-Flags oder Datenformaten anpassen.

