#!/usr/bin/env Rscript
# Basic R script to run perceptionML analysis via command line
# This script demonstrates the simplest way to access perceptionML from R

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Function to run perceptionML analysis
run_perception_analysis <- function(data_file, treatment, outcome, text_col = NULL) {
  
  # Build the command
  cmd <- "python -m perceptionML basic"
  cmd <- paste(cmd, "--data", shQuote(data_file))
  cmd <- paste(cmd, "--treatment", shQuote(treatment))
  cmd <- paste(cmd, "--outcome", shQuote(outcome))
  
  if (!is.null(text_col)) {
    cmd <- paste(cmd, "--text", shQuote(text_col))
  }
  
  # Add standard parameters for reproducibility
  cmd <- paste(cmd, "--pca-components 200")
  cmd <- paste(cmd, "--top-features 6")
  cmd <- paste(cmd, "--cv-folds 5")
  cmd <- paste(cmd, "--random-seed 42")
  cmd <- paste(cmd, "--output-dir perception_results")
  
  # Print command for debugging
  cat("Running command:\n")
  cat(cmd, "\n\n")
  
  # Execute the command
  result <- system(cmd, intern = FALSE)
  
  if (result == 0) {
    cat("\nAnalysis completed successfully!\n")
    cat("Results saved in: perception_results/\n")
    
    # Read and display the results table
    if (file.exists("perception_results/results_table.csv")) {
      results <- read.csv("perception_results/results_table.csv")
      cat("\nResults Summary:\n")
      print(head(results, 10))
    }
  } else {
    cat("\nError: Analysis failed with code", result, "\n")
  }
  
  return(result)
}

# Main execution
if (length(args) < 3) {
  cat("Usage: Rscript basic_analysis.R <data.csv> <treatment> <outcome> [text_column]\n")
  cat("\nExample:\n")
  cat("  Rscript basic_analysis.R essays.csv social_class ai_rating essay_text\n")
  cat("\nFor precomputed embeddings:\n")
  cat("  Rscript basic_analysis.R embeddings.csv social_class ai_rating\n")
  quit(status = 1)
}

# Extract arguments
data_file <- args[1]
treatment <- args[2]
outcome <- args[3]
text_col <- if (length(args) >= 4) args[4] else NULL

# Check if file exists
if (!file.exists(data_file)) {
  cat("Error: Data file", data_file, "not found\n")
  quit(status = 1)
}

# Run the analysis
cat("PerceptionML Basic Analysis\n")
cat("===========================\n")
cat("Data file:", data_file, "\n")
cat("Treatment:", treatment, "\n")
cat("Outcome:", outcome, "\n")
if (!is.null(text_col)) {
  cat("Text column:", text_col, "\n")
}
cat("\n")

result <- run_perception_analysis(data_file, treatment, outcome, text_col)

quit(status = result)