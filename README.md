# Pediatric Sleep–Wake Classification (Actigraphy-Only LSTM and classic models)

This repository serves as supplementary of the manuscript published as Chen et al., (2026) The Sleep-Wake Classification Performance of Pediatric-Trained Machine Learning Algorithms for Accelerometer Data, Sleep. The LSTM model built was adapted from LSTM-Sleep model (Weaver et al., *Journal of Sleep Research*, 2025) with modification of cross validation. The rest of the classic models were developed from tidyverse and tidymodel R packages. Please cite Chen et al., 2026 if you utilize the scripts.  Please cite Weaver et al., 2025 if also used LSTM model.


## Interactive Visualizations
The interactive visualization is the supplementary figures from Chen et al., 2026 Sleep. 
- **[Algorithm Ranking (Sankey Diagram)](https://pinweichen.github.io/Sleep-WakeClassification_Pediatric/Sankey/index.html)** — Compare 8 algorithms across 6 performance metrics

## Data Structure
For machine learning modeling, your input CSV needs columns: `timestamp` (ISO 8601), `x`, `y`, `z` (accelerometer in g) and Ground-truth `label` column.
The script handles all preprocessing internally (resampling, FFT feature extraction, epoching) and outputs one prediction CSV per subject.

For GGIR, the original ".bin" files were used. 

## Pipeline for LSTM

| Step | Script | Description |
|---|---|---|
| 1 | `data_ingestion.py` | Convert per-subject CSVs to pipeline format |
| 2 | `preprocessing/preprocessing_feature.py` | Crop signals and extract FFT features (1–30 Hz) |
| 3 | `source/analysis_runner_weighted_split_torch.py` | Train LocalGlobalLSTM with LOSO cross-validation |

## Pipeline for classic models

| Step | Script | Description |
|---|---|---|
| 1 | `feature_create.R` | Create features for the classic models |
| 2 | `ML_master_script_all_ML.R` | This is the main script that implement the preprocessing and nested cross validation |

## Requirements

```
Python >= 3.9
torch >= 1.12
numpy, pandas, scikit-learn, scipy, tqdm

R >= 4.4.0
tidyverse >= 2.0.0
pacman >= 0.5.1
data.table >= 1.18.2.1
doParallel >= 1.0.17
futile.logger >= 1.4.9
workflowsets >= 1.1.1
GGIR >= 3.3-4

- pacman for package installation and loading

| Package | Purpose |
|----------|----------|
| tidymodels | Machine learning framework |
| tidyverse | Data manipulation and visualization |
| data.table | High-performance data processing |
| doParallel | Parallel model training |
| parallel | Native parallel computing |
| doFuture | Future-based parallel backend |
| vip | Variable importance analysis |
| themis | SMOTE oversampling |
| discrim | Naive Bayes models |
| ranger | Random forest models |
| kknn | k-nearest neighbors models |
| glmnet | Regularized logistic regression |
| xgboost | Extreme gradient boosting |
| kernlab | Support vector machines |
| baguette | Bagged tree models |
| workflowsets | Workflow orchestration |
| futile.logger | Logging |

```

## Citation

If you use this pipeline, please cite the original paper:
> Chen, P-w et al. (2026). The Sleep-Wake Classification Performance of Pediatric-Trained Machine Learning Algorithms for Accelerometer Data. *Journal of Sleep*.
> 
> Weaver, T. et al. (2025). Predicting Sleep and Sleep Stage in Children Using Actigraphy and Heart Rate via a Local-Global LSTM. *Journal of Sleep Research*.

## License

See [LICENSE](LICENSE) for details.
