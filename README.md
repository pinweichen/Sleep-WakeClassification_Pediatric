# Pediatric Sleep–Wake Classification (Actigraphy-Only LSTM and classic models)

This repository serves as supplementary of the manuscript published as Chen et al., (2026) The Sleep-Wake Classification Performance of Pediatric-Trained Machine Learning Algorithms for Accelerometer Data, Sleep. The LSTM model built was adapted from [LSTM-Sleep](https://github.com/theweaverlab/lstm-sleep) model (Weaver et al., *Journal of Sleep Research*, 2025) with modification of cross validation. The rest of the classic models were developed from tidyverse and tidymodel R packages from POSIT Inc. T  Please cite both Chen et al., 2026 if you utilize the scripts.  Please cite Weaver et al., 2025 if also used LSTM model.

**[View the project website](https://pinweichen.github.io/Pediatric_Sleep_ML/)**

## Interactive Visualizations
The interactive visualization is the supplementary figures from Chen et al., 2026 Sleep. 
- **[Algorithm Ranking (Sankey Diagram)](https://pinweichen.github.io/Pediatric_Sleep_ML/Sankey/)** — Compare 8 algorithms across 6 performance metrics

## Pipeline for LSTM

| Step | Script | Description |
|---|---|---|
| 1 | `data_ingestion.py` | Convert per-subject CSVs to pipeline format |
| 2 | `preprocessing/preprocessing_feature.py` | Crop signals and extract FFT features (1–30 Hz) |
| 3 | `source/analysis_runner_weighted_split_torch.py` | Train LocalGlobalLSTM with LOSO cross-validation |

## Pipeline for classic models

| Step | Script | Description |
|---|---|---|
| 1 | `data_ingestion.py` | Convert per-subject CSVs to pipeline format |
| 2 | `preprocessing/preprocessing_feature.py` | Crop signals and extract FFT features (1–30 Hz) |
| 3 | `source/analysis_runner_weighted_split_torch.py` | Train LocalGlobalLSTM with LOSO cross-validation |

Your input CSV needs columns: `timestamp` (ISO 8601), `x`, `y`, `z` (accelerometer in g). Ground-truth `label` column is optional.
The script handles all preprocessing internally (resampling, FFT feature extraction, epoching) and outputs one prediction CSV per subject.

## Requirements

```
Python >= 3.9
torch >= 1.12
numpy, pandas, scikit-learn, scipy, tqdm
```

## Citation

If you use this pipeline, please cite the original paper:
> Chen, P-w et al. (2026). The Sleep-Wake Classification Performance of Pediatric-Trained Machine Learning Algorithms for Accelerometer Data. *Journal of Sleep*.
> 
> Weaver, T. et al. (2025). Predicting Sleep and Sleep Stage in Children Using Actigraphy and Heart Rate via a Local-Global LSTM. *Journal of Sleep Research*.

## License

See [LICENSE](LICENSE) for details.
