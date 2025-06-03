# Setup script for R users to run perceptionml
# This script sets up everything needed to run the anger_family.csv analysis

# Install reticulate if not already installed
if (!require("reticulate")) {
  install.packages("reticulate")
}

library(reticulate)

# Check Python installation
if (!py_available()) {
  stop("Python is not available. Please install Python 3.8 or higher.")
}

# Create virtual environment
virtualenv_create("perception_env", python = NULL)
use_virtualenv("perception_env", required = TRUE)

# Install perceptionml
py_install("perceptionml", pip = TRUE)

# Check if required files exist
required_files <- c("anger_family.csv", "perceptionml_wrapper.py")
missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(paste("Missing required files:", paste(missing_files, collapse = ", "),
             "\nPlease ensure these files are in your working directory."))
}

# Source the wrapper
source_python("perceptionml_wrapper.py")

cat("Setup complete! You can now run:\n")
cat("result <- analyze_anger_family()\n")