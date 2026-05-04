"""
Main CLI entry point for MCAP validation pipeline.
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import yaml

from .recording.loader import McapLoader
from .analysis.simulator import apply_fog_to_frames
from .recording.keyframes import load_keyframes
from .analysis.metrics import (
    compute_chamfer_distance,
    compute_hausdorff_distance,
    compute_intensity_metrics,
    normalize_intensity_median,
    aggregate_sequence_metrics
)
from .analysis.aggregate import (
    compute_range_binned_statistics,
    fit_extinction_coefficient,
    compute_curve_similarity
)
from .analysis.sanity import generate_overlay_visualization


def save_mcap_output(frames: list, output_path: Path, topic_name: str, reference_mcap: Path):
    """
    Save synthetic fog frames to MCAP file.
    
    Note: MCAP writing requires careful version-specific handling.
    For now, we skip MCAP writing and focus on CSV metrics which are
    the primary scientific output. Geometric metrics can still be computed
    without saving the synthetic MCAP by directly using the in-memory frames.
    """
    print(f"\n⚠ MCAP writing not yet implemented in this version.")
    print(f"  Synthetic fog data exists in memory and will be used for keyframe metrics.")
    print(f"  CSV outputs contain all scientifically relevant data.\n")
    
    # Create placeholder file
    output_path.write_text("# Synthetic MCAP writing not yet implemented\n")
    print(f"✓ Placeholder created: {output_path.name}")



def main():
    parser = argparse.ArgumentParser(
        description="MCAP LiDAR Fog Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Distributional only
  python -m lfa.mcap_validation.cli --distributional \\
      --clean clean.mcap --real_fog fog.mcap \\
      --visibility 100 --output results/

  # Geometric only (requires --keyframes)
  python -m lfa.mcap_validation.cli --geometric \\
      --visibility 100 --keyframes matched1.yaml --output results/

  # Both stages (default when neither flag is given)
  python -m lfa.mcap_validation.cli \\
      --visibility 100 --keyframes matched1.yaml --output results/
        """
    )
    
    parser.add_argument('--clean', type=Path, default=None,
                        help='Clean reference MCAP (optional if paths defined in keyframes YAML)')
    parser.add_argument('--real_fog', type=Path, default=None,
                        help='Real fog MCAP (optional if paths defined in keyframes YAML)')
    parser.add_argument('--visibility', type=float, required=True, help='Visibility in meters')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--topic', type=str, default='/sensing/lidar/concatenated/pointcloud',
                        help='LiDAR topic name')
    parser.add_argument('--fog_metric', type=str, default='distance', choices=['distance', 'chamfer'],
                        help='Fog simulation metric')
    parser.add_argument('--voxel_size', type=float, default=0.2, help='Voxel size in meters')
    parser.add_argument('--distributional', action='store_true',
                        help='Run Stage 1: distributional comparison')
    parser.add_argument('--geometric', action='store_true',
                        help='Run Stage 2: geometric comparison (requires --keyframes)')
    parser.add_argument('--save_synthetic', action='store_true',
                        help='Save synthetic fog MCAP to disk')
    parser.add_argument('--keyframes', type=Path, help='YAML file with keyframe pairs')
    parser.add_argument('--sample_frames', type=int, default=10,
                        help='For sequences: generate overlays every N frames (0=disable)')
    parser.add_argument('--sanity-checks', action='store_true',
                        help='Generate BEV overlay images (sanity_overlays/) in geometric stage')
    parser.add_argument('--skip_seconds_clean', type=float, default=0.0,
                        help='Skip the first N seconds of the clean MCAP (default: 0)')
    parser.add_argument('--skip_seconds_fog', type=float, default=0.0,
                        help='Skip the first N seconds of the real fog MCAP (default: 0)')
    parser.add_argument('--max_frames_clean', type=int, default=None,
                        help='Maximum number of clean frames to load (default: all)')
    parser.add_argument('--max_frames_fog', type=int, default=None,
                        help='Maximum number of real fog frames to load (default: all)')
    
    args = parser.parse_args()

    # If neither stage flag is set, run both
    if not args.distributional and not args.geometric:
        args.distributional = True
        args.geometric = True

    # Geometric requires keyframes
    if args.geometric and not args.keyframes:
        parser.error('--geometric requires --keyframes')

    # Resolve --clean / --real_fog from keyframes YAML if not supplied on CLI
    if args.keyframes and args.keyframes.exists() and (args.clean is None or args.real_fog is None):
        with open(args.keyframes, 'r') as f:
            kf_data = yaml.safe_load(f)
        yaml_paths = kf_data.get('paths', {})
        if args.clean is None:
            if 'clean' not in yaml_paths:
                parser.error('--clean not provided and no paths.clean in keyframes YAML')
            args.clean = Path(yaml_paths['clean'])
        if args.real_fog is None:
            if 'real_fog' not in yaml_paths:
                parser.error('--real_fog not provided and no paths.real_fog in keyframes YAML')
            args.real_fog = Path(yaml_paths['real_fog'])

    if args.clean is None:
        parser.error('--clean is required (provide via CLI or paths.clean in keyframes YAML)')
    if args.real_fog is None:
        parser.error('--real_fog is required (provide via CLI or paths.real_fog in keyframes YAML)')

    # Derive match identifier from keyframes YAML filename (e.g. matched1.yaml → match1)
    match_tag = None
    if args.keyframes:
        m = re.search(r'matched?(\d+)', args.keyframes.stem)
        if m:
            match_tag = f"match{m.group(1)}"

    # Create output directory: <match_tag>_data_<timestamp> or <timestamp>
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{match_tag}_data_{timestamp}" if match_tag else timestamp
    output_dir = args.output / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  MCAP FOG VALIDATION PIPELINE")
    print(f"{'='*70}")
    print(f"Clean MCAP:      {args.clean}")
    print(f"Real Fog MCAP:   {args.real_fog}")
    print(f"Visibility:      {args.visibility} m")
    print(f"Topic:           {args.topic}")
    print(f"Voxel size:      {args.voxel_size} m")
    stages = []
    if args.distributional:
        stages.append("distributional")
    if args.geometric:
        stages.append("geometric")
    print(f"Stages:          {' + '.join(stages)}")
    print(f"Save synthetic:  {args.save_synthetic}")
    print(f"Keyframes:       {args.keyframes if args.keyframes else 'None'}")
    if args.skip_seconds_clean or args.skip_seconds_fog or args.max_frames_clean or args.max_frames_fog:
        print(f"Trim clean:      skip={args.skip_seconds_clean}s, max_frames={args.max_frames_clean}")
        print(f"Trim fog:        skip={args.skip_seconds_fog}s, max_frames={args.max_frames_fog}")
    print(f"Output:          {output_dir}")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # DATA LOADING (shared by both stages)
    # ========================================================================
    print("Loading clean reference MCAP...")
    clean_loader = McapLoader(args.clean, args.topic)
    clean_frames = clean_loader.load_all_frames(
        max_frames=args.max_frames_clean,
        skip_seconds=args.skip_seconds_clean
    )
    
    synth_frames = apply_fog_to_frames(clean_frames, args.visibility, args.fog_metric)
    
    print("\nLoading real fog MCAP...")
    real_loader = McapLoader(args.real_fog, args.topic)
    real_frames = real_loader.load_all_frames(
        max_frames=args.max_frames_fog,
        skip_seconds=args.skip_seconds_fog
    )
    
    # ========================================================================
    # STAGE 1: DISTRIBUTIONAL COMPARISON
    # ========================================================================
    if args.distributional:
        print(f"\n{'='*70}")
        print(f"  STAGE 1: DISTRIBUTIONAL COMPARISON")
        print(f"{'='*70}\n")
        
        print("Computing distributional statistics...")
        range_stats = compute_range_binned_statistics(clean_frames, synth_frames, real_frames)
        
        synth_beta = fit_extinction_coefficient(range_stats['synth_survival'], range_stats['range_centers'])
        real_beta = fit_extinction_coefficient(range_stats['real_survival'], range_stats['range_centers'])
        
        curve_similarity = compute_curve_similarity(range_stats['real_survival'], range_stats['synth_survival'])
        
        aggregate_csv_path = output_dir / 'aggregate_metrics.csv'
        with open(aggregate_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Synthetic_Fog', 'Real_Fog'])
            writer.writerow(['extinction_coefficient_alpha', synth_beta['alpha'], real_beta['alpha']])
            writer.writerow(['visibility_est_m', synth_beta['visibility_est'], real_beta['visibility_est']])
            writer.writerow(['alpha_r_squared', synth_beta['r_squared'], real_beta['r_squared']])
            writer.writerow(['survival_curve_rmse', curve_similarity['survival_rmse'], ''])
            writer.writerow(['survival_curve_mae', curve_similarity['survival_mae'], ''])
            writer.writerow(['survival_curve_correlation', curve_similarity['survival_correlation'], ''])
        print(f"✓ Saved: {aggregate_csv_path.name}")
        
        range_csv_path = output_dir / 'range_binned_metrics.csv'
        with open(range_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Range_Min', 'Range_Max', 'Range_Center',
                            'Clean_Count', 'Synth_Count', 'Real_Count',
                            'Synth_Survival', 'Real_Survival',
                            'Synth_Deletion', 'Real_Deletion'])
            
            for i in range(len(range_stats['range_centers'])):
                writer.writerow([
                    range_stats['range_bins'][i],
                    range_stats['range_bins'][i+1],
                    range_stats['range_centers'][i],
                    range_stats['clean_counts'][i],
                    range_stats['synth_counts'][i],
                    range_stats['real_counts'][i],
                    range_stats['synth_survival'][i],
                    range_stats['real_survival'][i],
                    range_stats['synth_deletion'][i],
                    range_stats['real_deletion'][i],
                ])
        print(f"✓ Saved: {range_csv_path.name}")
        
        intensity_csv_path = output_dir / 'intensity_ratio_by_distance.csv'
        with open(intensity_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Range_Center',
                            'Clean_Intensity_Mean', 'Synth_Intensity_Mean', 'Real_Intensity_Mean',
                            'Synth_Intensity_Ratio', 'Real_Intensity_Ratio'])
            
            for i in range(len(range_stats['range_centers'])):
                writer.writerow([
                    range_stats['range_centers'][i],
                    range_stats['clean_intensity_mean'][i],
                    range_stats['synth_intensity_mean'][i],
                    range_stats['real_intensity_mean'][i],
                    range_stats['synth_mean_I_ratio'][i],
                    range_stats['real_mean_I_ratio'][i],
                ])
        print(f"✓ Saved: {intensity_csv_path.name}")
        
        print("\n✅ STAGE 1 COMPLETE: Distributional metrics saved")
    
    # ========================================================================
    # STAGE 2: GEOMETRIC COMPARISON
    # ========================================================================
    if args.geometric:
        print(f"\n{'='*70}")
        print(f"  STAGE 2: GEOMETRIC COMPARISON")
        print(f"{'='*70}\n")
        
        if args.save_synthetic:
            synthetic_mcap_path = output_dir / 'synthetic_fog.mcap'
            save_mcap_output(synth_frames, synthetic_mcap_path, args.topic, args.clean)
        
        print("Loading keyframes for geometric comparison...")
        keyframe_sequences, alignment_info = load_keyframes(args.keyframes, clean_frames, real_frames)
        
        if alignment_info is not None:
            alignment_path = output_dir / 'time_alignment_report.txt'
            with open(alignment_path, 'w') as f:
                f.write("TIME ALIGNMENT REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Mean time offset: {alignment_info['mean_offset']:.3f}s\n")
                f.write(f"Std deviation: {alignment_info['std_offset']:.3f}s\n")
                f.write(f"Max deviation: {alignment_info['max_deviation']:.3f}s\n\n")
                f.write("Individual landmarks:\n")
                for label, offset, clean_t, fog_t in alignment_info['offsets']:
                    f.write(f"  {label}: clean={clean_t:.3f}s, fog={fog_t:.3f}s, offset={offset:.3f}s\n")
            print(f"✓ Saved: {alignment_path.name}")
        
        all_frame_results = []
        sequence_summary = []
        
        if args.sanity_checks:
            sanity_dir = output_dir / 'sanity_overlays'
            sanity_dir.mkdir(exist_ok=True)
        
        for seq in keyframe_sequences:
            print(f"\nProcessing sequence: {seq.label}")
            
            if seq.is_sequence:
                print(f"  Frame range: {len(seq.frame_pairs)} pairs")
            
            sequence_results = []
            
            for pair_idx, (clean_idx, fog_idx, pair_label) in enumerate(seq.frame_pairs):
                clean_frame = clean_frames[clean_idx]
                synth_frame = synth_frames[clean_idx]
                real_frame = real_frames[fog_idx]
                
                if seq.roi is not None:
                    clean_frame = crop_frame(clean_frame, seq.roi)
                    synth_frame = crop_frame(synth_frame, seq.roi)
                    real_frame = crop_frame(real_frame, seq.roi)
                    if pair_idx == 0:
                        print(f"  ROI crop: clean {len(clean_frames[clean_idx]['xyz'])}→{len(clean_frame['xyz'])} pts, "
                              f"synth {len(synth_frames[clean_idx]['xyz'])}→{len(synth_frame['xyz'])} pts, "
                              f"real {len(real_frames[fog_idx]['xyz'])}→{len(real_frame['xyz'])} pts")
                
                chamfer = compute_chamfer_distance(synth_frame['xyz'], real_frame['xyz'], args.voxel_size)
                hausdorff = compute_hausdorff_distance(synth_frame['xyz'], real_frame['xyz'], args.voxel_size)
                
                synth_intensity_norm = normalize_intensity_median(synth_frame['intensity'])
                real_intensity_norm = normalize_intensity_median(real_frame['intensity'])
                intensity_metrics = compute_intensity_metrics(synth_intensity_norm, real_intensity_norm)
                
                frame_result = {
                    'sequence_label': seq.label,
                    'pair_label': pair_label,
                    'clean_idx': clean_idx,
                    'fog_idx': fog_idx,
                    **chamfer,
                    **hausdorff,
                    **intensity_metrics,
                    'synth_n_points': len(synth_frame['xyz']),
                    'real_n_points': len(real_frame['xyz'])
                }
                
                sequence_results.append(frame_result)
                all_frame_results.append(frame_result)
                
                should_generate_overlay = False
                if args.sanity_checks:
                    if not seq.is_sequence:
                        should_generate_overlay = True
                    elif args.sample_frames > 0:
                        is_first = (pair_idx == 0)
                        is_last = (pair_idx == len(seq.frame_pairs) - 1)
                        is_sampled = (pair_idx % args.sample_frames == 0)
                        should_generate_overlay = is_first or is_last or is_sampled
                
                if should_generate_overlay:
                    if seq.is_sequence:
                        seq_overlay_dir = sanity_dir / seq.label
                        seq_overlay_dir.mkdir(exist_ok=True)
                    else:
                        seq_overlay_dir = sanity_dir
                    
                    overlay_synth = seq_overlay_dir / f'{pair_label}_synth_vs_clean.png'
                    generate_overlay_visualization(
                        clean_frame['xyz'], synth_frame['xyz'],
                        overlay_synth,
                        f'Synthetic vs Clean: {pair_label}',
                        args.voxel_size
                    )
                    
                    overlay_real = seq_overlay_dir / f'{pair_label}_real_vs_clean.png'
                    generate_overlay_visualization(
                        clean_frame['xyz'], real_frame['xyz'],
                        overlay_real,
                        f'Real vs Clean: {pair_label}',
                        args.voxel_size
                    )
            
            if seq.is_sequence:
                aggregated = aggregate_sequence_metrics(sequence_results)
                aggregated['sequence_label'] = seq.label
                aggregated['is_sequence'] = True
                sequence_summary.append(aggregated)
                
                print(f"  ✓ Processed {len(sequence_results)} frame pairs")
                print(f"    Chamfer mean: {aggregated['chamfer_mean_mean']:.3f} ± {aggregated['chamfer_mean_std']:.3f} m")
                print(f"    Hausdorff p95: {aggregated['hausdorff_p95_mean']:.3f} ± {aggregated['hausdorff_p95_std']:.3f} m")
            else:
                result = sequence_results[0]
                summary_entry = {
                    'sequence_label': seq.label,
                    'n_frames': 1,
                    'is_sequence': False,
                    'chamfer_mean_mean': result['chamfer_mean'],
                    'chamfer_p50_mean': result['chamfer_p50'],
                    'chamfer_p95_mean': result['chamfer_p95'],
                    'hausdorff_max_mean': result['hausdorff_max'],
                    'hausdorff_p95_mean': result['hausdorff_p95'],
                    'intensity_rmse_mean': result['intensity_rmse'],
                    'intensity_mae_mean': result['intensity_mae'],
                }
                sequence_summary.append(summary_entry)
                print(f"  ✓ Processed single frame")
        
        detailed_csv_path = output_dir / 'keyframe_metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'sequence_label', 'pair_label', 'clean_idx', 'fog_idx',
                'chamfer_mean', 'chamfer_p50', 'chamfer_p95',
                'hausdorff_max', 'hausdorff_p95',
                'intensity_rmse', 'intensity_mae',
                'synth_n_points', 'real_n_points'
            ])
            writer.writeheader()
            writer.writerows(all_frame_results)
        print(f"\n✓ Saved: {detailed_csv_path.name} ({len(all_frame_results)} frame pairs)")
        
        summary_csv_path = output_dir / 'keyframe_metrics_summary.csv'
        with open(summary_csv_path, 'w', newline='') as f:
            if sequence_summary:
                fieldnames = list(sequence_summary[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sequence_summary)
        print(f"✓ Saved: {summary_csv_path.name} ({len(sequence_summary)} sequences)")
        
        print("\n✅ STAGE 2 COMPLETE: Geometric metrics saved")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"  ✅ VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles generated:")
    if args.distributional:
        print(f"  - aggregate_metrics.csv (extinction coeff, RMSE, correlation)")
        print(f"  - range_binned_metrics.csv (survival/deletion by distance)")
        print(f"  - intensity_ratio_by_distance.csv (I_fog/I_clean ratios)")
    if args.geometric:
        print(f"  - keyframe_metrics_detailed.csv (per-frame Chamfer/Hausdorff)")
        print(f"  - keyframe_metrics_summary.csv (aggregated stats per sequence)")
        if args.sanity_checks:
            print(f"  - sanity_overlays/ (overlay images)")
    if args.save_synthetic:
        print(f"  - synthetic_fog.mcap (~{len(synth_frames)} frames)")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
