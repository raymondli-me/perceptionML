#!/usr/bin/env Rscript
# Reproduce the exact MiniLM analysis from the paper
# This script matches the bash script run_minilm_analysis.sh

# Function to find the embeddings file
find_embeddings_file <- function() {
  possible_paths <- c(
    "fully_anonymized_embeddings_minilm.csv",
    "../fully_anonymized_embeddings_minilm.csv", 
    "../../fully_anonymized_embeddings_minilm.csv"
  )
  
  for (path in possible_paths) {
    if (file.exists(path)) {
      return(normalizePath(path))
    }
  }
  
  return(NULL)
}

# Main analysis function
run_minilm_analysis <- function(embeddings_file = NULL) {
  
  # Find embeddings file if not provided
  if (is.null(embeddings_file)) {
    embeddings_file <- find_embeddings_file()
    if (is.null(embeddings_file)) {
      stop("Could not find fully_anonymized_embeddings_minilm.csv\n",
           "Please provide the path as an argument or place the file in the current directory.")
    }
  }
  
  cat("==========================================\n")
  cat("   Running MiniLM Analysis with BASIC MODE\n")
  cat("==========================================\n\n")
  
  cat("Found embeddings file:", embeddings_file, "\n\n")
  
  cat("Starting MiniLM analysis with:\n")
  cat("  - Treatment: social_class\n")
  cat("  - Outcome: ai_rating\n")
  cat("  - Model: MiniLM (384 dimensions)\n")
  cat("  - Random seed: 42\n")
  cat("  - PCA components: 200\n")
  cat("  - Top features: 6\n")
  cat("  - CV folds: 5\n\n")
  
  # Set environment variables for reproducibility
  Sys.setenv(PYTHONHASHSEED = "42")
  Sys.setenv(OMP_NUM_THREADS = "1")
  Sys.setenv(MKL_NUM_THREADS = "1")
  Sys.setenv(OPENBLAS_NUM_THREADS = "1")
  Sys.setenv(VECLIB_MAXIMUM_THREADS = "1")
  Sys.setenv(NUMEXPR_NUM_THREADS = "1")
  
  # Build the command
  cmd <- paste(
    "python -m perceptionML basic",
    "--embeddings", shQuote(embeddings_file),
    "--treatment social_class",
    "--outcome ai_rating",
    "--precomputed",
    "--random-seed 42",
    "--pca-components 200",
    "--top-features 6",
    "--cv-folds 5",
    "--output-dir minilm_analysis_results"
  )
  
  # Run the analysis
  result <- system(cmd)
  
  if (result == 0) {
    cat("\n==========================================\n")
    cat("   MiniLM Analysis Complete!              \n")
    cat("==========================================\n\n")
    
    cat("Results saved to: minilm_analysis_results/\n\n")
    
    cat("Key output files:\n")
    cat("  - minilm_analysis_results/results_table.csv        # Main results matching paper format\n")
    cat("  - minilm_analysis_results/feature_selections.csv   # Top 6 PC selections for each method\n")
    cat("  - minilm_analysis_results/pca_components.csv       # 200 PCA components\n")
    cat("  - minilm_analysis_results/pca_variance_explained.csv\n")
    cat("  - minilm_analysis_results/analysis_params.json     # Parameters for reproducibility\n\n")
    
    # Read and display results
    if (file.exists("minilm_analysis_results/results_table.csv")) {
      cat("Loading results...\n\n")
      results <- read.csv("minilm_analysis_results/results_table.csv", stringsAsFactors = FALSE)
      
      # Display key results
      cat("=== KEY RESULTS ===\n\n")
      
      # Baseline OLS
      baseline <- results[results$Model == "OLS" & results$Features == "N/A", ]
      cat(sprintf("Baseline OLS (no controls): θ = %.4f (SE = %.4f)\n", 
                  baseline$Coeff..θ., baseline$Robust.SE))
      
      # Full embeddings
      full_ols <- results[results$Model == "OLS" & results$Features == "Full", ]
      cat(sprintf("OLS with full embeddings: θ = %.4f (SE = %.4f), R²(Y) = %.4f\n",
                  full_ols$Coeff..θ., full_ols$Robust.SE, full_ols$R².Y..Full))
      
      # PCA
      pca_ols <- results[results$Model == "OLS" & results$Features == "200 PCs", ]
      cat(sprintf("OLS with 200 PCs: θ = %.4f (SE = %.4f), R²(Y) = %.4f\n",
                  pca_ols$Coeff..θ., pca_ols$Robust.SE, pca_ols$R².Y..Full))
      
      cat("\n")
      
      # Feature selections
      if (file.exists("minilm_analysis_results/feature_selections.csv")) {
        selections <- read.csv("minilm_analysis_results/feature_selections.csv", stringsAsFactors = FALSE)
        cat("=== FEATURE SELECTIONS ===\n\n")
        print(selections)
      }
    }
    
  } else {
    cat("\nError: Analysis failed with code", result, "\n")
  }
  
  return(result)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) > 0) {
  # Use provided embeddings file
  result <- run_minilm_analysis(args[1])
} else {
  # Try to find embeddings file automatically
  result <- run_minilm_analysis()
}

quit(status = result)