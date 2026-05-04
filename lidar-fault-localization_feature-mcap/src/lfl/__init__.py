"""
Lidar Fault Localization (lfl) package.

Main modules:
- cli: Command-line interface (lfl_pipeline entry point)
- runner: Batch experiment runner for multiple sequences
- lfl_pipeline: Core orchestration (KISS-ICP, fault injection, EVO evaluation)
- plot_metrics_comprehensive: Publication-quality metrics visualization (lfl_plot entry point)
- fault_stats_logger: Fault statistics aggregation and logging

Internal modules (use directly when needed):
- lfl_pipeline: run_lfl() function for single experiment runs
"""
