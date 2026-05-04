import numpy as np
import random
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

from lfa.load_data import get_dir, get_bin_files, get_subdirectories, load_point_cloud
from lfa.metrics.distribution_histograms import compute_distribution_histograms
from lfa.metrics.hausdorff_distance import compare_point_clouds_hausdorff
from lfa.metrics.chamfer_distance import compare_point_clouds_chamfer
from lfa.metrics.sdasn import compute_point_clouds_sdasn
from lfa.metrics.visualize import visualize_lidar_point_clouds



def compare_pointclouds(ref_folder: Path, oc_gt_folder: Path, args):
    """
    Compares point clouds from reference and occluded ground truth folders.
    
    :param ref_folder: Path to the reference point clouds
    :type ref_folder: Path
    :param oc_gt_folder: Path to the occluded ground truth point clouds
    :type oc_gt_folder: Path
    :param fault_model: Fault model to apply to the reference point clouds
    :type fault_model: str
    """

    # Holt sich alle Unterordner (Messungen) in den zu vergleichenden Ordnern
    ref_measurements = get_subdirectories(ref_folder)
    oc_gt_measurements = get_subdirectories(oc_gt_folder)

    # Sammelt alle Histogramme und Metriken
    all_histograms = []
    hausdorff_gt_values = []        # Hausdorff Distanzen für Referenz vs Ground Truth
    hausdorff_model_values = []     # Hausdorff Distanzen für Referenz vs Model
    chamfer_gt_values = []      # Chamfer Distanzen für Referenz vs Ground Truth
    chamfer_model_values = []   # Chamfer Distanzen für Referenz vs Model
    sdasn_ref_values = []       # SDASN für Referenz
    sdasn_gt_values = []        # SDASN für Ground Truth
    sdasn_model_values = []     # SDASN für Model

    bins = None
    labels = None
    num_iterations = 0

    # Vergleicht alle Messungen (Alle möglichen Kombinationen der einzelnen Messungen) paarweise
    for ref_measurement, oc_gt_measurement in product(ref_measurements, oc_gt_measurements):

        # Holt sich alle .bin Dateien in den Messungsordnern und bestimmt den Datensatz mit der geringeren Anzahl
        ref_bins = get_bin_files(ref_measurement)
        oc_gt_bins = get_bin_files(oc_gt_measurement)

        min_len = min(len(ref_bins), len(oc_gt_bins))

        # Vergleicht die Punktwolken paarweise
        for i in range(min_len):

            # Lädt die Punktwolken
            ref_data_file = ref_bins[i]
            oc_gt_data_file = oc_gt_bins[i]

            ref_data = load_point_cloud(str(ref_data_file))
            oc_gt_data = load_point_cloud(str(oc_gt_data_file))
            oc_model_data = load_point_cloud(str(ref_data_file), args.fault_model)

            data_list = {"Referenz": ref_data, "Occlusion - Ground Truth": oc_gt_data, "Occlusion - Model": oc_model_data}

            # Preprocessing
            for key, data in data_list.items():
                mask = (data[:, 2] >= -0.5) & (data[:, 0] <= 36.5)
                data_list[key] = data[mask]

            # Punkteverteilung auswerten und Histogramme sammeln
            if "distribution" in args.metric:
                bins, labels, histograms = compute_distribution_histograms(data_list, bin_width=1.0)
                all_histograms.append(histograms)
            
            # Hausdorff Distance berechnen und sammeln
            if "hausdorff" in args.metric:
                ref_pc = data_list["Referenz"]
                gt_pc = data_list["Occlusion - Ground Truth"]
                model_pc = data_list["Occlusion - Model"]
                
                hausdorff_gt = compare_point_clouds_hausdorff({"Referenz": ref_pc, "Occlusion - Ground Truth": gt_pc})
                hausdorff_model = compare_point_clouds_hausdorff({"Referenz": ref_pc, "Occlusion - Model": model_pc})
                
                hausdorff_gt_values.append(hausdorff_gt["Referenz vs Occlusion - Ground Truth"])
                hausdorff_model_values.append(hausdorff_model["Referenz vs Occlusion - Model"])
            
            # Chamfer Distance berechnen und sammeln
            if "chamfer" in args.metric:
                ref_pc = data_list["Referenz"]
                gt_pc = data_list["Occlusion - Ground Truth"]
                model_pc = data_list["Occlusion - Model"]
                
                chamfer_gt = compare_point_clouds_chamfer({"Referenz": ref_pc, "Occlusion - Ground Truth": gt_pc})
                chamfer_model = compare_point_clouds_chamfer({"Referenz": ref_pc, "Occlusion - Model": model_pc})
                
                chamfer_gt_values.append(chamfer_gt["Referenz vs Occlusion - Ground Truth"])
                chamfer_model_values.append(chamfer_model["Referenz vs Occlusion - Model"])
            
            # SDASN berechnen und sammeln
            if "sdasn" in args.metric:
                sdasn_values = compute_point_clouds_sdasn(data_list)
                sdasn_ref_values.append(sdasn_values["Referenz"])
                sdasn_gt_values.append(sdasn_values["Occlusion - Ground Truth"])
                sdasn_model_values.append(sdasn_values["Occlusion - Model"])


            num_iterations += 1

    # Mittelt alle distribution histograms und plottet das Ergebnis
    if "distribution" in args.metric and all_histograms and bins is not None and labels is not None:
        # Konvertiert Liste von Listen zu Array für einfachere Verarbeitung
        # Form: (num_iterations, num_labels, num_bins)
        all_histograms_array = np.array(all_histograms)
        
        # Mittelt über alle Iterationen
        averaged_histograms = np.mean(all_histograms_array, axis=0)
        
        # Plot
        plt.figure(figsize=(10, 6))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for hist, label in zip(averaged_histograms, labels):
            plt.plot(bin_centers, hist, marker='o', label=label)

        plt.xlabel("Entfernung der Punkte in m", fontsize=12)
        plt.ylabel("Anzahl der Punkte (Normiert auf saubere Referenz) in Prozent", fontsize=12)
        plt.title(f"Punktverteilung der LiDAR-Punktwolken (Mittel über {num_iterations} Frames) \n\n"
                  f"Sequenz: {args.sequence}, Verschmutzung: {args.oc_type}, Verschmutzungsgrad: {args.oc_level}, Fehlermodell: {args.fault_model}", 
                  fontsize=14)
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.ylim(bottom=0)
        plt.show()
    
    # Plottet Histogramm der Hausdorff Distance Metriken
    if "hausdorff" in args.metric and hausdorff_gt_values and hausdorff_model_values:
        plt.figure(figsize=(12, 6))
        
        # Histogramm für beide Verteilungen
        all_values = np.concatenate([hausdorff_gt_values, hausdorff_model_values])

        num_bins = 100
        bin_min = np.min(all_values)
        bin_max = np.max(all_values)
        bins = np.linspace(bin_min, bin_max, num_bins + 1)

        plt.hist(
            hausdorff_gt_values,
            bins=bins,
            alpha=0.6,
            label="Referenz vs Ground Truth",
            edgecolor="black"
        )

        plt.hist(
            hausdorff_model_values,
            bins=bins,
            alpha=0.6,
            label="Referenz vs Model",
            edgecolor="black"
        )
        
        plt.xlabel("Modified Hausdorff Distance (m)", fontsize=12)
        plt.ylabel("Häufigkeit", fontsize=12)
        plt.title(f"Verteilung der Modified Hausdorff Distance Metriken ({num_iterations} Frames) \n\n"
                  f"Sequenz: {args.sequence}, Verschmutzung: {args.oc_type}, Verschmutzungsgrad: {args.oc_level}, Fehlermodell: {args.fault_model}", 
                  fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiken ausgeben
        print("\n" + "="*60)
        print("Modified Hausdorff Distance Statistiken")
        print("="*60)
        print(f"\nReferenz vs Ground Truth:")
        print(f"  Mittelwert: {np.mean(hausdorff_gt_values):.4f} m")
        print(f"  Std Abw.:   {np.std(hausdorff_gt_values):.4f} m")
        print(f"  Min:        {np.min(hausdorff_gt_values):.4f} m")
        print(f"  Max:        {np.max(hausdorff_gt_values):.4f} m")
        print(f"\nReferenz vs Model:")
        print(f"  Mittelwert: {np.mean(hausdorff_model_values):.4f} m")
        print(f"  Std Abw.:   {np.std(hausdorff_model_values):.4f} m")
        print(f"  Min:        {np.min(hausdorff_model_values):.4f} m")
        print(f"  Max:        {np.max(hausdorff_model_values):.4f} m")
        print("="*60)
    
    # Plottet Histogramm der Chamfer Distance Metriken
    if "chamfer" in args.metric and chamfer_gt_values and chamfer_model_values:
        plt.figure(figsize=(12, 6))
        
        # Histogramm für beide Verteilungen
        all_values = np.concatenate([chamfer_gt_values, chamfer_model_values])

        num_bins = 100
        bin_min = np.min(all_values)
        bin_max = np.max(all_values)
        bins = np.linspace(bin_min, bin_max, num_bins + 1)

        plt.hist(
            chamfer_gt_values,
            bins=bins,
            alpha=0.6,
            label="Referenz vs Ground Truth",
            edgecolor="black"
        )

        plt.hist(
            chamfer_model_values,
            bins=bins,
            alpha=0.6,
            label="Referenz vs Model",
            edgecolor="black"
        )
        
        plt.xlabel("Chamfer Distance (m)", fontsize=12)
        plt.ylabel("Häufigkeit", fontsize=12)
        plt.title(f"Verteilung der Chamfer Distance Metriken ({num_iterations} Frames) \n\n"
                  f"Sequenz: {args.sequence}, Verschmutzung: {args.oc_type}, Verschmutzungsgrad: {args.oc_level}, Fehlermodell: {args.fault_model}", 
                  fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiken ausgeben
        print("\n" + "="*60)
        print("Chamfer Distance Statistiken")
        print("="*60)
        print(f"\nReferenz vs Ground Truth:")
        print(f"  Mittelwert: {np.mean(chamfer_gt_values):.4f} m")
        print(f"  Std Abw.:   {np.std(chamfer_gt_values):.4f} m")
        print(f"  Min:        {np.min(chamfer_gt_values):.4f} m")
        print(f"  Max:        {np.max(chamfer_gt_values):.4f} m")
        print(f"\nReferenz vs Model:")
        print(f"  Mittelwert: {np.mean(chamfer_model_values):.4f} m")
        print(f"  Std Abw.:   {np.std(chamfer_model_values):.4f} m")
        print(f"  Min:        {np.min(chamfer_model_values):.4f} m")
        print(f"  Max:        {np.max(chamfer_model_values):.4f} m")
        print("="*60)
    
    # Plottet Histogramm der SDASN Metriken
    if "sdasn" in args.metric and sdasn_ref_values and sdasn_gt_values and sdasn_model_values:
        plt.figure(figsize=(12, 6))
        
        # Histogramm für alle drei Verteilungen
        all_values = np.concatenate([sdasn_ref_values, sdasn_gt_values, sdasn_model_values])

        num_bins = 300
        bin_min = np.min(all_values)
        bin_max = np.max(all_values)
        bins = np.linspace(bin_min, bin_max, num_bins + 1)

        plt.hist(sdasn_ref_values, bins=bins, alpha=0.6, label="Referenz", edgecolor='black')
        plt.hist(sdasn_gt_values, bins=bins, alpha=0.6, label="Occlusion - Ground Truth", edgecolor='black')
        plt.hist(sdasn_model_values, bins=bins, alpha=0.6, label="Occlusion - Model", edgecolor='black')
        
        plt.xlabel("SDASN (m)", fontsize=12)
        plt.ylabel("Häufigkeit", fontsize=12)
        plt.title(f"Verteilung der SDASN Metriken ({num_iterations} Frames) \n\n"
                  f"Sequenz: {args.sequence}, Verschmutzung: {args.oc_type}, Verschmutzungsgrad: {args.oc_level}, Fehlermodell: {args.fault_model}", 
                  fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiken ausgeben
        print("\n" + "="*60)
        print("SDASN Statistiken")
        print("="*60)
        print(f"\nReferenz:")
        print(f"  Mittelwert: {np.mean(sdasn_ref_values):.4f} m")
        print(f"  Std Abw.:   {np.std(sdasn_ref_values):.4f} m")
        print(f"  Min:        {np.min(sdasn_ref_values):.4f} m")
        print(f"  Max:        {np.max(sdasn_ref_values):.4f} m")
        print(f"\nOcclusion - Ground Truth:")
        print(f"  Mittelwert: {np.mean(sdasn_gt_values):.4f} m")
        print(f"  Std Abw.:   {np.std(sdasn_gt_values):.4f} m")
        print(f"  Min:        {np.min(sdasn_gt_values):.4f} m")
        print(f"  Max:        {np.max(sdasn_gt_values):.4f} m")
        print(f"\nOcclusion - Model:")
        print(f"  Mittelwert: {np.mean(sdasn_model_values):.4f} m")
        print(f"  Std Abw.:   {np.std(sdasn_model_values):.4f} m")
        print(f"  Min:        {np.min(sdasn_model_values):.4f} m")
        print(f"  Max:        {np.max(sdasn_model_values):.4f} m")
        print("="*60)






def run_lfa(args):

    ref_dir = get_dir(args.sequence, "cover")
    oc_gt_dir = get_dir(args.sequence, args.oc_type, args.oc_level)

    # ref_dir = "data/lidaroc/20m/10_test/1_low"
    # oc_gt_dir = "data/lidaroc/20m/11_test/3_high"

    compare_pointclouds(ref_folder=Path(ref_dir), oc_gt_folder=Path(oc_gt_dir), args=args)

    if args.visualize:
        visualize_lidar_point_clouds(ref_folder=Path(ref_dir), oc_gt_folder=Path(oc_gt_dir), args=args)

    




 

