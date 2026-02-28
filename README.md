**Expected file structure**
```plaintext
skyfinder-dir-project/ 
│ 
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ configs/
│  ├─ skyfinder_feature_config.yaml
│  ├─ experiments/
│  │  ├─ stage1_minimal.yaml
│  │  ├─ stage1_full.yaml
│  │  ├─ stage1_ablation.yaml
│  │  └─ stage2_future.yaml
│  └─ splits/
│     ├─ cross_camera_split_seed42.yaml
│     ├─ cross_camera_split_seed43.yaml
│     └─ random_split_seed42.yaml
│
├─ data/
│  ├─ raw/
│  │  └─ skyfinder.csv
│  ├─ processed/
│  │  ├─ skyfinder_cleaned.csv
│  │  ├─ skyfinder_cleaned_missing_stats.csv
│  │  └─ analysis_summary.json
│  └─ interim/
│     ├─ feature_tables/
│     └─ split_tables/
│
├─ notebooks/
│  ├─ 01_data_cleaning_and_qc.ipynb
│  ├─ 02_label_distribution_analysis.ipynb
│  ├─ 03_cross_camera_analysis.ipynb
│  └─ 04_results_summary.ipynb
│
├─ src/
│  ├─ __init__.py
│  │
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ config_loader.py
│  │  ├─ io_utils.py
│  │  ├─ logging_utils.py
│  │  ├─ seed_utils.py
│  │  └─ metrics_utils.py
│  │
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ clean_csv.py
│  │  ├─ feature_builder.py
│  │  ├─ split_builder.py
│  │  ├─ dataset.py
│  │  └─ preprocess.py
│  │
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ mlp_regressor.py
│  │  ├─ fds.py
│  │  └─ losses.py
│  │
│  ├─ training/
│  │  ├─ __init__.py
│  │  ├─ train_one_run.py
│  │  ├─ trainer.py
│  │  ├─ evaluate.py
│  │  └─ inference.py
│  │
│  └─ analysis/
│     ├─ __init__.py
│     ├─ label_distribution.py
│     ├─ camera_analysis.py
│     ├─ summarize_results.py
│     └─ ablation_analysis.py
│
├─ scripts/
│  ├─ run_cleaning.sh
│  ├─ run_stage1_minimal.sh
│  ├─ run_stage1_full.sh
│  ├─ run_stage1_ablation.sh
│  └─ summarize_results.sh
│
├─ runs/
│  ├─ stage1/
│  │  ├─ E01_S1_Plain_seed42/
│  │  │  ├─ config_snapshot.yaml
│  │  │  ├─ split_snapshot.yaml
│  │  │  ├─ train_log.txt
│  │  │  ├─ metrics.json
│  │  │  ├─ predictions_val.csv
│  │  │  ├─ model_best.pt
│  │  │  └─ curves/
│  │  │     ├─ loss_curve.png
│  │  │     └─ pred_vs_true.png
│  │  │
│  │  ├─ E02_S1_LDS_seed42/
│  │  ├─ E03_S2_Plain_seed42/
│  │  └─ E04_S2_LDS_seed42/
│  │
│  └─ stage2/
│     └─ (future)
│
├─ reports/
│  ├─ tables/
│  │  ├─ experiment_registry.csv
│  │  ├─ core_results_stage1.csv
│  │  ├─ multi_seed_summary.csv
│  │  ├─ gain_analysis.csv
│  │  └─ failure_analysis.csv
│  │
│  ├─ figures/
│  │  ├─ temp_distribution.png
│  │  ├─ temp_distribution_log.png
│  │  ├─ temp_by_camera.png
│  │  ├─ temp_by_month.png
│  │  └─ ablation_barplot.png
│  │
│  └─ drafts/
│     ├─ stage1_notes.md
│     ├─ method_outline.md
│     └─ discussion_outline.md
│
└─ tests/
   ├─ test_config_loader.py
   ├─ test_feature_builder.py
   └─ test_split_builder.py
```
