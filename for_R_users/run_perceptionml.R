# R script to run perceptionml through reticulate

library(reticulate)

# Option 1: Using the Python wrapper
# Set up the virtual environment
use_virtualenv("perception_env", required = TRUE)

# Source the wrapper
source_python("perceptionml_wrapper.py")

# Run the analysis
cat("Running perceptionml analysis...\n")
result <- analyze_anger_family(output_file = "minilm_reddit_analysis_from_R.html")

if (result[[1]]) {
  cat("Analysis completed successfully!\n")
  cat("Output saved to: minilm_reddit_analysis_from_R.html\n")
} else {
  cat("Analysis failed:\n")
  cat(result[[2]], "\n")
}

# Option 2: Direct system call
# system2("bash", args = "./run_perceptionml.sh", stdout = TRUE, stderr = TRUE)

# Option 3: Using subprocess through reticulate
# subprocess <- import("subprocess")
# result <- subprocess$run(
#   c("perceptionml", "--data", "anger_family.csv", 
#     "--y-var", "gpt_sum_score", 
#     "--x-var", "human_sum_score",
#     "--control-vars", "num_raters",
#     "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
#     "--num-gpus", "4",
#     "--batch-size", "32",
#     "--sample-size", "10000",
#     "--auto-cluster", "descriptions",
#     "--outcome-mode", "zero_presence",
#     "--stratify-by", "human_sum_score",
#     "--output", "minilm_reddit_analysis_R.html"),
#   capture_output = TRUE,
#   text = TRUE
# )