import pyvista as pv
import random

from pathlib import Path
from lfa.load_data import get_subdirectories, get_bin_files, load_point_cloud

#Utilitiy function to visualize lidar point clouds
def plot_lidar(data_list: dict, cmap: str ="coolwarm", point_size: int =4) -> None:
    """
    data_list: Dictionary mit 3 Arrays der Form (N, 4) -> x, y, z, intensity
    cmap: Colormap für die Intensitätswerte
    point_size: Größe der Punkte in der Visualisierung

    Visualisiert drei Punktwolken nebeneinander mit Farbcodierung basierend auf Intensitätswerten.
    """

    # Prüfen, ob genau 3 Datensätze übergeben wurden
    assert len(data_list) == 3, "Es müssen genau 3 Datensätze übergeben werden."

    plotter = pv.Plotter(shape=(1, 3))

    # Jede Punktwolke in einem eigenen Subplot darstellen
    for i, (label, data) in enumerate(data_list.items()):
        points = data[:, :3]
        intensity = data[:, 3]

        cloud = pv.PolyData(points)
        cloud["intensity"] = intensity

        plotter.subplot(0, i)
        plotter.add_points(
            cloud,
            scalars="intensity",
            cmap=cmap,
            point_size=point_size
        )
        plotter.add_text(label, font_size=10)

    # Synchrones Drehen/Zoomen zwischen den Subplots 
    plotter.link_views()  

    # Kameraposition für alle Subplots festlegen
    plotter.camera_position = [
        (-1, -1, 1),  
        (0, 0, 0),    
        (0, 0, 1)     
    ]

    plotter.reset_camera()

    plotter.show()


def visualize_lidar_point_clouds(ref_folder: Path, oc_gt_folder: Path, args) -> None:


    # Pfade zu den Messungen holen und zufällige auswählen   
    ref_measuremnts = get_subdirectories(ref_folder)
    oc_gt_measuremnts = get_subdirectories(oc_gt_folder)

    ref_measurement = random.choice(ref_measuremnts)
    oc_gt_measurement = random.choice(oc_gt_measuremnts)


    # Alle .bin Dateien in den Messungsordnern holen und zufällige auswählen
    ref_bins = get_bin_files(ref_measurement)
    oc_gt_bins = get_bin_files(oc_gt_measurement)

    ref_data = random.choice(ref_bins)
    oc_gt_data = random.choice(oc_gt_bins)
    
    # Punktwolken einlesen
    ref_points = load_point_cloud(str(ref_data))
    oc_gt_points = load_point_cloud(str(oc_gt_data))
    oc_points = load_point_cloud(str(ref_data), args.fault_model)

    data_list = {"Referenz": ref_points, "Occlusion - Ground Truth": oc_gt_points, "Occlusion - Model": oc_points}

    # Daten filtern
    for key, data in data_list.items():
        mask = (data[:, 2] >= -0.5) & (data[:, 0] <= 36.5)
        data_list[key] = data[mask]

    # Daten visualisieren
    plot_lidar(data_list, point_size=4)