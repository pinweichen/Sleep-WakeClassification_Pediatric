# 1. SETUP & LIBRARIES
# -----------------------------------------------------------------------------
# Use pacman to load/install packages
if(!require(pacman)){
  install.packages("pacman", repos = "http://cran.us.r-project.org")
  library(pacman)
} else {
  library(pacman)
}
pacman::p_load(
  tidymodels,       # Core tidymodels framework
  tidyverse,      # Data manipulation and visualization
  data.table,     # Fast data manipulation
  doParallel,     # For parallel processing
  parallel,
  doFuture,
  vip,            # For variable importance plots
  themis,         # For SMOTE recipe step
  discrim,        # For Naive Bayes models
  ranger,         # Fast random forest implementation
  kknn,           # k-Nearest Neighbors
  glmnet,         # Logistic Regression
  xgboost,        # XGBoost engine
  kernlab,        # SVM engine
  baguette       # Bag tree
)
if(!require(workflowsets)){
  install.packages("workflowsets", repos = "http://cran.us.r-project.org")
  library(workflowsets)
} else {
  library(workflowsets)
}
if(!require(futile.logger)){
  install.packages("futile.logger", repos = "http://cran.us.r-project.org")
  library(futile.logger)
} else {
  library(futile.logger)
}


# --- Define Paths and Constants ---
tz = "America/New_York"

# Set model to run
# This will input the model argument. "model_to_run" determine which model to run. 
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("No model name provided! Usage: Rscript ML_master_script.R <model_name>", call. = FALSE)
} else {
  model_to_run <- args[1]
  cat(sprintf("--- RUNNING ANALYSIS FOR MODEL: %s ---\n", model_to_run))
}


# -----------------------------------------------------------------
feature_path <- "path to feature space"

# Output
# The reviewer want us to test if there no correlation implemented
ML_subject_output_path <- "path to output subject folder"
QC_subject_output_path <- "Path to QC output for each subject"
importance_subject_output_path <- "Path to importance"
model_subject_output_path <- "Path to model outputs"
parameters_subject_output_path <- "Path to parameters"
log_file <- "log files path"

# Create the directory where the log file will be saved
dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)
dir.create(ML_subject_output_path, showWarnings = FALSE, recursive = TRUE)
dir.create(QC_subject_output_path, showWarnings = FALSE, recursive = TRUE)
dir.create(importance_subject_output_path, showWarnings = FALSE, recursive = TRUE)
dir.create(model_subject_output_path, showWarnings = FALSE, recursive = TRUE)
dir.create(parameters_subject_output_path, showWarnings = FALSE, recursive = TRUE)

# Configure the logger to write to the console and the log file
flog.appender(appender.tee(log_file))
flog.info("---------------------------------------------------------")
flog.info("Starting new machine learning run at %s", Sys.time())
flog.info("---------------------------------------------------------")

dir.create(ML_output_path,showWarnings = FALSE, recursive = TRUE)
# --- Parallel Processing Setup ---

# 1. Detect Slurm Cores
n_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK"))

# 2. Fallback logic 
if (is.na(n_cores)) {
  flog.info("SLURM_CPUS_PER_TASK not set. Detecting cores locally.")
  # Subtract 1 locally to keep the OS responsive; on Slurm use all assigned.
  n_cores <- parallel::detectCores(logical = FALSE) - 1
}

flog.info(paste("Setting up parallel processing with", n_cores, "cores."))

# 3. Register the Parallel Backend
registerDoFuture()
plan(multisession, workers = n_cores)

# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
feature_files <- list.files(feature_path, pattern = "\\.csv$", full.names = TRUE)
dt_all <- map_dfr(feature_files, ~fread(.x) %>%
                    mutate(ID = str_extract(basename(.x), "^[0-9]+")))

# --- Data Cleaning and Transformation ---
# Keep track of original epoch and ID info for later

# Clean and prepare the dataset for modeling
dt_all <- dt_all %>%
  # Create numeric factor for classification
  mutate(label_n = factor(if_else(label == "Sleep", 1, 0), levels = c(1, 0))) %>%
  # Select only the necessary columns (ID, outcome, and predictors)
  dplyr::select(ID, label_n, everything(), -label, -epoch)

# Get unique participant IDs for LOSO cross-validation
ID_ls <- unique(dt_all$ID)
num_participants <- length(ID_ls)

cat(sprintf("Data loaded for %d participants.\n", num_participants))

#3. MODEL & WORKFLOW DEFINITIONS
# -----------------------------------------------------------------------------
# --- Preprocessing Recipe ---
# This recipe will be applied to all models.

dt_recipe <- recipe(label_n ~ ., data = dt_all %>% dplyr::select(-ID)) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors(), neighbors = 5) %>% 
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  step_smote(label_n, over_ratio = 1, seed = 42)

# --- Model Specifications ---
# Create a list of all models to be tested. This makes it easy to add/remove models.
model_definitions <- list(
  # 1. Decision Tree
  tree = decision_tree(cost_complexity = tune(), tree_depth = tune()) %>%
    set_engine("rpart") %>% set_mode("classification"),

  # 2. Random Forest
  rf = rand_forest(trees = tune(), min_n = tune()) %>%
    set_engine("ranger", importance = "permutation") %>% set_mode("classification"),

  # # 3. XGBoost
  xgb = boost_tree(trees = tune(), min_n = tune(), learn_rate = tune()) %>%
    set_engine("xgboost") %>% set_mode("classification"),

  # 4. k-Nearest Neighbors
  knn = nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
    set_engine("kknn") %>% set_mode("classification"),

  # # 5. SVM (Linear)
  svm_linear = svm_linear(cost = tune()) %>%
    set_engine("kernlab") %>% set_mode("classification"),
  #
  # # 6. SVM (Radial Basis Function Kernel)
  svm_rbf = svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab", control = list(maxit = 2000)) %>% # Increase iterations
    set_mode("classification"),
  #
  # # 7. Logistic Regression (Baseline)
  log_reg = logistic_reg(penalty = tune(), mixture = 1) %>%
    # Pass 'maxit' and 'thresh' directly, not inside a control list
    set_engine("glmnet", maxit = 1e6, thresh = 1e-8) %>% 
    set_mode("classification"),
  #
  # # 8. Naive Bayes
  naive_bayes = naive_Bayes(smoothness = tune(), Laplace = tune()) %>%
    set_engine("klaR") %>% set_mode("classification"),

  # 9. Bagged Tree 
  bag_tree = bag_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>%
    set_engine("rpart", times = 50) %>% # 'times' is the number of bootstrapped models
    set_mode("classification"),
  # 10. Neural Network (MLP)
  nn_mlp = mlp(hidden_units = tune(), penalty = tune()) %>%
    set_engine("nnet") %>%
    set_mode("classification")
  # 
)



# Select the 'model_to_run'
selected_model_list <- model_definitions[model_to_run]

if (!model_to_run %in% names(model_definitions)) {
  stop(sprintf("Error: model '%s' not found in model_definitions. Available models are: %s",
               model_to_run, paste(names(model_definitions), collapse = ", ")))
}

# --- Create Workflow Set ---
# This combines the recipe with all models automatically.
wflow_set <- workflow_set(
  preproc = list(base_recipe = dt_recipe),
  models = selected_model_list,
  cross = TRUE 
)

# 4. LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
# -----------------------------------------------------------------------------
set.seed(42)

# Checkpointing to save progress after each subject.
checkpoint_file <- file.path(ML_output_path, "loso_detailed_metrics_checkpoint.csv")
# Determine which subjects have already been processed.
completed_ids <- c()
if (file.exists(checkpoint_file)) {
  completed_ids <- fread(checkpoint_file, select = "ID") %>% pull(ID) %>% unique()
  cat(sprintf("Checkpoint file found. Resuming. %d of %d subjects already processed.\n",
              length(completed_ids), length(ID_ls)))
}
ids_to_process <- setdiff(ID_ls, completed_ids)

for (current_id in ids_to_process) {
  
  cat(sprintf("Processing fold for ID: %s (%d of %d remaining)...\n",
              current_id, match(current_id, ids_to_process), length(ids_to_process)))
  # Log the start time for this specific subject
  loop_start_time <- Sys.time()
  flog.info("------------------------------------------------")
  flog.info("Processing fold for ID: %s (%d of %d remaining)...",
            current_id, match(current_id, ids_to_process), length(ids_to_process))
  
  tryCatch({
    
    # Split data: one subject for testing, the rest for training
    flog.info("ID %s: Splitting data into training/testing sets.", current_id)
    training_data <- dt_all %>% filter(ID != current_id)
    testing_data  <- dt_all %>% filter(ID == current_id)
    
    # --- QUALITY CONTROL (QC) DATA COLLECTION START ---
    
    tryCatch({
      flog.info("ID %s: Running QC checks (Imputation & Feature Removal)...", current_id)
      
      # 1. Capture Imputation Stats (Raw NAs before processing)
      # -------------------------------------------------------
      na_stats <- training_data %>% 
        dplyr::select(-ID, -label_n) %>% 
        summarise(across(everything(), ~ sum(is.na(.)))) %>%
        tidyr::pivot_longer(everything(), names_to = "feature", values_to = "missing_count") %>%
        filter(missing_count > 0) %>%
        mutate(
          ID = current_id, 
          total_rows = nrow(training_data),
          pct_missing = (missing_count / nrow(training_data)) * 100
        )
      
      # Save Imputation QC
      if (nrow(na_stats) > 0) {
        fwrite(na_stats, file.path(QC_subject_output_path, paste0("QC_imputation_", current_id, ".csv")))
      }
      
      # 2. Capture Removed Features (Zero Variance & Correlation)
      # ---------------------------------------------------------
      # We must 'prep' the recipe on the current training data to see which vars get dropped
      qc_prep <- prep(dt_recipe, training = training_data, verbose = FALSE)
      
      # Extract Zero Variance Removals (Step 1 in your recipe)
      zv_removed <- tidy(qc_prep, number = 1) %>% 
        mutate(reason = "Zero Variance", ID = current_id)
      
      # Based on your previous snippet: 1=zv, 2=dummy, 3=norm, 4=impute, 5=corr
      corr_removed <- tidy(qc_prep, number = 5) %>% 
        mutate(reason = "High Correl ation (>0.9)", ID = current_id)
       
      # Combine and Save
      all_removed <- bind_rows(zv_removed, corr_removed)
      if (nrow(all_removed) > 0) {
        fwrite(all_removed, file.path(QC_subject_output_path, paste0("QC_removed_features_", current_id, ".csv")))
        flog.info("ID %s: QC Saved. Removed %d features due to ZV/Correlation.", current_id, nrow(all_removed))
      }
      
    }, error = function(e) {
      flog.warn("ID %s: QC Data Collection Failed (non-critical). Error: %s", current_id, e$message)
    })
    # ==============================================================================
    # --- QUALITY CONTROL (QC) DATA COLLECTION END ---
    # ==============================================================================
    # Create inner cross-validation folds for hyperparameter tuning
    cv_folds <- vfold_cv(training_data, v = 5, strata = label_n)
    
    # Tune all models defined in the workflow_set
    flog.info("ID %s: Starting hyperparameter tuning with workflow_map...", current_id)
    tuning_start_time <- Sys.time()
    tuned_models <- wflow_set %>%
      workflow_map(
        "tune_grid",
        resamples = cv_folds,
        grid = 10,
        metrics = metric_set(roc_auc),
        control = control_grid(save_pred = TRUE, verbose = TRUE, parallel_over = "everything"),
        seed = 42
      )
    
    tuning_time <- Sys.time() - tuning_start_time
    flog.info("ID %s: Hyperparameter tuning FINISHED. Time taken: %s", current_id, format(tuning_time))
    flog.info("ID %s: Processing results for each model...", current_id)
    
    # Helper function to process one successfully tuned model
    process_one_model <- function(wflow_id, tuned_result, training_data, testing_data) {
      
      # 1. Error Diagnosis
      if (inherits(tuned_result, "try-error") || !inherits(tuned_result, "tbl_df")) {
        error_msg <- "Unknown error"
        if (inherits(tuned_result, "tbl_df") && "notes" %in% names(tuned_result)) {
          notes_df <- tuned_result$notes[[1]]
          if (!is.null(notes_df) && nrow(notes_df) > 0) {
            error_msg <- paste(unique(notes_df$note), collapse = " | ")
          }
        }
        flog.warn("ID %s: Model '%s' FAILED tuning. Error details: %s", current_id, wflow_id, error_msg)
        return(NULL)
      }
      
      # 2. Extract Best Parameters
      best_params <- select_best(tuned_result, metric = "roc_auc")
      if (nrow(best_params) == 0) {
        flog.warn("ID %s: No best parameters found for model '%s'.", current_id, wflow_id)
        return(NULL)
      }
      
      # 3. Save Best Parameters (From previous step)
      tryCatch({
        params_to_save <- best_params %>% 
          mutate(model = wflow_id, ID = current_id, metric_used = "roc_auc") %>%
          dplyr::select(-any_of(c(".config")))
        
        param_file <- file.path(parameters_subject_output_path, paste0("ML_best_params_", current_id, "_", wflow_id, ".csv"))
        fwrite(params_to_save, param_file)
      }, error = function(e) {
        flog.warn("ID %s: Failed to save parameters. Error: %s", current_id, e$message)
      })
      
      # 4. Finalize and Fit the Model
      final_wflow <- extract_workflow(tuned_models, wflow_id) %>%
        finalize_workflow(best_params)
      
      final_fit <- fit(final_wflow, data = training_data)
      
      # ==============================================================================
      # --- SAVE TRAINED MODEL OBJECT START ---
      # ==============================================================================
      tryCatch({
        model_filename <- file.path(model_subject_output_path, paste0("ML_trained_model_", current_id, "_", wflow_id, ".rds"))
        saveRDS(final_fit, model_filename, compress = TRUE)
        flog.info("ID %s: Saved trained model object to %s", current_id, basename(model_filename))
      }, error = function(e) {
        flog.warn("ID %s: Failed to save model object for '%s'. Error: %s", current_id, wflow_id, e$message)
      })
      # ==============================================================================
      # --- SAVE TRAINED MODEL OBJECT END ---
      # ==============================================================================
      
      # 5. Extract Features & Importance
      tryCatch({
        # Save Features
        prepped_recipe <- pull_workflow_prepped_recipe(final_fit)
        final_features <- prepped_recipe$var_info %>%
          filter(role == "predictor") %>%
          pull(variable)
        
        feature_data <- data.table(feature = final_features, model = wflow_id, ID = current_id)
        fwrite(feature_data, file.path(ML_subject_output_path, paste0("ML_features_", current_id, "_", wflow_id, ".csv")))
        
        # Save Importance
        parsnip_fit <- extract_fit_parsnip(final_fit)
        importance_data <- tryCatch({
          vip::vi(parsnip_fit) %>%
            mutate(model = wflow_id, ID = current_id) %>%
            as.data.table()
        }, error = function(e) return(NULL))
        
        if (!is.null(importance_data) && nrow(importance_data) > 0) {
          fwrite(importance_data, file.path(importance_subject_output_path, paste0("ML_importance_", current_id, "_", wflow_id, ".csv")))
        }
      }, error = function(e) {
        flog.error("ID %s: Feature extraction failed. Error: %s", current_id, e$message)
      })
      
      # 6. Predictions & Metrics
      test_predictions <- predict(final_fit, new_data = testing_data, type = "prob") %>%
        bind_cols(predict(final_fit, new_data = testing_data)) %>%
        bind_cols(testing_data %>% select(label_n))
      
      fwrite(test_predictions, file.path(ML_subject_output_path, paste0("ML_pred_",current_id,"_",wflow_id,".csv")))
      
      class_metrics <- metric_set(accuracy, bal_accuracy, sensitivity, specificity, roc_auc)
      metrics <- test_predictions %>%
        class_metrics(truth = label_n, estimate = .pred_class, .pred_1) %>%
        mutate(ID = current_id, model = wflow_id)
      
      return(metrics)
    }
    
    all_metrics_for_subject <- purrr::map2_dfr(
      .x = tuned_models$wflow_id,
      .y = tuned_models$result,
      .f = ~tryCatch({
        process_one_model(.x, .y, training_data = training_data, testing_data = testing_data)
      }, error = function(e){
        flog.error("ID %s: FAILED to process results for model '%s'. Error: %s",
                   current_id, .x, e$message)
        return(NULL)
      })
    )
    
    # Save results for the current subject to the checkpoint file
    if (!is.null(all_metrics_for_subject) && nrow(all_metrics_for_subject) > 0) {
      fwrite(all_metrics_for_subject, file = checkpoint_file, append = file.exists(checkpoint_file))
      flog.info("ID %s: Successfully saved metrics for %d models.", current_id, nrow(all_metrics_for_subject))
    } else {
      flog.warn("ID %s: No model metrics were successfully generated to save.", current_id)
    }

    loop_time <- Sys.time() - loop_start_time
    flog.info(">>>> Successfully processed and saved ID: %s. Total time for this fold: %s <<<<", 
              current_id, format(loop_time))

    
  }, error = function(e) {
    flog.error("! ERROR processing ID %s !", current_id)
    flog.error("The error was: %s", conditionMessage(e))
    flog.info("Skipping to the next ID.")
  }) # End of tryCatch
  
} # End of for loop

# 5. ANALYZE AND SAVE RESULTS
# -----------------------------------------------------------------------------
# Load the complete results from the checkpoint file for final analysis.
loso_results <- fread(checkpoint_file)

# --- Summarize Performance Across All Subjects ---
summary_metrics <- loso_results %>%
  group_by(model, .metric) %>%
  summarise(
    mean_performance = mean(.estimate, na.rm = TRUE),
    std_err = sd(.estimate, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# --- Display and Save Results ---
print("--- Average Performance Across All Subjects (LOSO-CV) ---")
print(summary_metrics, n = 50)

# Save summary results (the detailed results are already in the checkpoint file)
fwrite(summary_metrics, file.path(ML_output_path, "loso_summary_metrics.csv"))

# Visualize the results
summary_metrics %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = mean_performance, y = reorder(model, mean_performance),
             xmin = mean_performance - std_err, xmax = mean_performance + std_err)) +
  geom_point(color = "blue", size = 3) +
  geom_errorbarh(height = 0.2) +
  labs(
    title = "Model Performance (ROC AUC) via LOSO-CV",
    subtitle = "Error bars represent standard error across subjects",
    x = "Mean ROC AUC", y = "Model"
  ) +
  theme_minimal()