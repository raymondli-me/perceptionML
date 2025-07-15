# PerceptionML BASIC MODE - R Wrapper
# 
# This wrapper provides R functions to use PerceptionML BASIC MODE
# Requires: reticulate package
#
# Installation:
#   install.packages("reticulate")
#   # Ensure Python environment has perceptionML installed
#
# Usage:
#   source("basic_mode_wrapper.R")
#   results <- run_basic_analysis(...)

library(reticulate)

# Initialize Python and import modules
init_perceptionml_basic <- function(python_path = NULL) {
  "Initialize PerceptionML BASIC MODE Python modules"
  
  if (!is.null(python_path)) {
    use_python(python_path)
  }
  
  # Import Python modules
  basic_mode <<- import("perceptionML.basic_mode")
  
  message("PerceptionML BASIC MODE initialized successfully")
}

# Main analysis function
run_basic_analysis <- function(
  data_path = NULL,
  embeddings_path = NULL,
  treatment_col,
  outcome_col,
  text_col = NULL,
  id_col = NULL,
  precomputed_embeddings = FALSE,
  embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
  n_pca_components = 200L,
  n_top_features = 6L,
  n_folds = 5L,
  random_seed = 42L,
  output_dir = NULL
) {
  "
  Run PerceptionML BASIC MODE analysis
  
  Args:
    data_path: Path to CSV file with raw data
    embeddings_path: Path to CSV file with precomputed embeddings
    treatment_col: Name of treatment column (X variable)
    outcome_col: Name of outcome column (Y variable)
    text_col: Name of text column (required if using raw data)
    id_col: Name of ID column (optional)
    precomputed_embeddings: Whether embeddings are precomputed
    embedding_model: Embedding model to use
    n_pca_components: Number of PCA components
    n_top_features: Number of top features to select
    n_folds: Number of CV folds
    random_seed: Random seed for reproducibility
    output_dir: Directory to save outputs
  
  Returns:
    Data frame with analysis results
  "
  
  # Ensure initialization
  if (!exists("basic_mode")) {
    init_perceptionml_basic()
  }
  
  # Create analyzer
  analyzer <- basic_mode$BasicAnalysis(
    embedding_model = embedding_model,
    n_pca_components = as.integer(n_pca_components),
    n_top_features = as.integer(n_top_features),
    n_folds = as.integer(n_folds),
    random_seed = as.integer(random_seed),
    output_dir = output_dir
  )
  
  # Run analysis
  if (precomputed_embeddings) {
    results_py <- analyzer$run(
      embeddings_path = embeddings_path,
      treatment_col = treatment_col,
      outcome_col = outcome_col,
      id_col = id_col,
      precomputed_embeddings = TRUE
    )
  } else {
    results_py <- analyzer$run(
      data_path = data_path,
      treatment_col = treatment_col,
      outcome_col = outcome_col,
      text_col = text_col,
      id_col = id_col,
      precomputed_embeddings = FALSE
    )
  }
  
  # Convert to R data frame
  results <- py_to_r(results_py)
  
  return(results)
}

# Generate embeddings only
generate_embeddings <- function(
  texts,
  model_name = "sentence-transformers/all-MiniLM-L6-v2",
  batch_size = 32L,
  normalize = TRUE,
  random_seed = 42L
) {
  "
  Generate embeddings for texts
  
  Args:
    texts: Character vector of texts
    model_name: Embedding model name
    batch_size: Batch size for encoding
    normalize: Whether to L2 normalize embeddings
    random_seed: Random seed
    
  Returns:
    Matrix of embeddings
  "
  
  # Ensure initialization
  if (!exists("basic_mode")) {
    init_perceptionml_basic()
  }
  
  # Create embedding generator
  embedder <- basic_mode$EmbeddingGenerator(
    model_name = model_name,
    batch_size = as.integer(batch_size),
    normalize = normalize,
    random_seed = as.integer(random_seed)
  )
  
  # Generate embeddings
  embeddings_py <- embedder$generate_embeddings(texts)
  
  # Convert to R matrix
  embeddings <- py_to_r(embeddings_py)
  
  return(embeddings)
}

# Example usage functions
example_raw_text <- function() {
  "Example using raw text data"
  
  cat("Example: Analyzing raw text data\n")
  
  # Sample data (replace with your data)
  results <- run_basic_analysis(
    data_path = "your_data.csv",
    treatment_col = "social_class",
    outcome_col = "ai_rating",
    text_col = "essay_text",
    id_col = "essay_id",
    precomputed_embeddings = FALSE
  )
  
  cat(sprintf("Total models analyzed: %d\n", nrow(results)))
  
  return(results)
}

example_precomputed <- function() {
  "Example using precomputed embeddings"
  
  cat("Example: Using precomputed embeddings\n")
  
  results <- run_basic_analysis(
    embeddings_path = "embeddings_minilm.csv",
    treatment_col = "social_class",
    outcome_col = "ai_rating",
    id_col = "id",
    precomputed_embeddings = TRUE
  )
  
  return(results)
}

example_embeddings_only <- function() {
  "Example generating embeddings only"
  
  cat("Example: Generating embeddings\n")
  
  texts <- c(
    "This is a sample text.",
    "Another example sentence.",
    "A third piece of text for analysis."
  )
  
  embeddings <- generate_embeddings(texts)
  
  cat(sprintf("Embeddings shape: %d x %d\n", nrow(embeddings), ncol(embeddings)))
  
  return(embeddings)
}

# Utility functions
get_best_model <- function(results) {
  "Get the best performing model from results"
  
  # Convert reduction percentages to numeric
  results$reduction_numeric <- as.numeric(gsub("%", "", results$`Reduction (vs baseline)`))
  
  # Find best model
  best_idx <- which.max(results$reduction_numeric)
  best_model <- results[best_idx, ]
  
  cat(sprintf("Best model: %s %s with %.2f%% reduction\n",
              best_model$Model,
              best_model$`Learner/Selector`,
              best_model$reduction_numeric))
  
  return(best_model)
}

filter_significant_results <- function(results, alpha = 0.05) {
  "Filter results to show only statistically significant models"
  
  # Convert p-values to numeric (handling scientific notation)
  results$p_numeric <- as.numeric(results$`p-value`)
  
  # Filter significant results
  significant <- results[results$p_numeric < alpha, ]
  
  cat(sprintf("Found %d significant models (p < %.2f)\n", nrow(significant), alpha))
  
  return(significant)
}

# Print formatted results table
print_results_table <- function(results) {
  "Print a nicely formatted results table"
  
  # Select key columns for display
  display_cols <- c("Model", "Learner/Selector", "Embedding", "Features",
                    "Coeff (θ)", "Robust SE", "p-value", "95% CI",
                    "R²(AI) Full", "R²(SC) Full", "Reduction (vs baseline)")
  
  # Create display table
  display_table <- results[, display_cols]
  
  # Print with formatting
  print(display_table, row.names = FALSE)
}

# Initialize on load
message("PerceptionML BASIC MODE R Wrapper loaded")
message("Run init_perceptionml_basic() to initialize Python modules")
message("See ?run_basic_analysis for usage")