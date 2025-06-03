# Using PerceptionML with R (via reticulate)

This setup allows R data scientists to run perceptionml analyses on the anger_family.csv dataset.

## Requirements
- R with reticulate package
- Python 3.8 or higher
- anger_family.csv file

## Files Needed
1. `perceptionml_wrapper.py` - Python wrapper for the CLI
2. `run_perceptionml.R` - Example R script
3. `setup_for_R_users.R` - One-time setup script
4. `anger_family.csv` - Your data file

## Setup Instructions

1. Download all files to your working directory
2. Run the setup script once:
   ```r
   source("setup_for_R_users.R")
   ```

3. Run the analysis:
   ```r
   library(reticulate)
   use_virtualenv("perception_env", required = TRUE)
   source_python("perceptionml_wrapper.py")
   
   # Run analysis
   result <- analyze_anger_family()
   
   if (result[[1]]) {
     cat("Success! Check minilm_reddit_analysis.html\n")
   }
   ```

## Customizing Parameters

To run with different parameters, use the `run_perceptionml_analysis()` function:

```r
result <- run_perceptionml_analysis(
  data_file = "anger_family.csv",
  y_var = "gpt_sum_score",
  x_var = "human_sum_score",
  control_vars = "num_raters",
  embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
  num_gpus = 4,
  batch_size = 32,
  sample_size = 10000,
  auto_cluster = "descriptions",
  outcome_mode = "zero_presence",
  stratify_by = "human_sum_score",
  output_file = "custom_output.html"
)
```

## Troubleshooting

- If you get a "Python not found" error, ensure Python 3.8+ is installed
- If perceptionml installation fails, try: `py_install("perceptionml", pip = TRUE, pip_options = "--user")`
- For GPU issues, set `num_gpus = 0` to use CPU only
- The analysis may take several minutes depending on your hardware

## Notes

- The virtual environment is created in your current directory as `perception_env/`
- Output HTML files will be created in your working directory
- This wrapper uses subprocess to call the perceptionml CLI, so all CLI options are supported