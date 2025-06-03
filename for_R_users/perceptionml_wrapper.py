"""
Wrapper for perceptionml to use with R's reticulate
"""
import subprocess
import sys

def run_perceptionml_analysis(
    data_file,
    y_var,
    x_var,
    control_vars,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    num_gpus=4,
    batch_size=32,
    sample_size=10000,
    auto_cluster="descriptions",
    outcome_mode="zero_presence",
    stratify_by=None,
    output_file="analysis.html"
):
    """
    Run perceptionml analysis with specified parameters
    
    Returns: tuple (success: bool, output: str)
    """
    cmd = [
        "perceptionml",
        "--data", data_file,
        "--y-var", y_var,
        "--x-var", x_var,
        "--control-vars", control_vars,
        "--embedding-model", embedding_model,
        "--num-gpus", str(num_gpus),
        "--batch-size", str(batch_size),
        "--sample-size", str(sample_size),
        "--auto-cluster", auto_cluster,
        "--outcome-mode", outcome_mode,
        "--output", output_file
    ]
    
    if stratify_by:
        cmd.extend(["--stratify-by", stratify_by])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

# Function specifically for the anger_family analysis
def analyze_anger_family(output_file="minilm_reddit_analysis.html"):
    """
    Run the specific anger_family.csv analysis
    """
    return run_perceptionml_analysis(
        data_file="anger_family.csv",
        y_var="gpt_sum_score",
        x_var="human_sum_score",
        control_vars="num_raters",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        num_gpus=4,
        batch_size=32,
        sample_size=10000,
        auto_cluster="descriptions",
        outcome_mode="zero_presence",
        stratify_by="human_sum_score",
        output_file=output_file
    )