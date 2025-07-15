"""
Setup script for PerceptionML
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="perceptionML",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Text perception analysis using Double Machine Learning with Language Model Embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/perceptionML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "perceptionml=perceptionML.__main__:main",
            "perceptionml-basic=perceptionML.basic_mode.__main__:main",
            "perceptionml-advanced=perceptionML.advanced_mode.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "perceptionML": [
            "basic_mode/*.py",
            "advanced_mode/*.py",
            "advanced_mode/configs/*.yaml",
            "advanced_mode/templates/*.html",
            "examples/*.py",
            "examples/*.R",
            "examples/*.sh",
        ],
    },
)