# Enhancing NHL Salary Evaluation through Dimensionality Reduction

> Raphaël Fontaine  
ECSE 526: Artificial Intelligence  
McGill University  
Montreal, Canada  
raphael.fontaine@mail.mcgill.ca  


## Description
This project explores the use of dimensionality
reduction techniques to evaluate and analyze the salaries of NHL
players. By applying methods such as PCA, PLS, Random
Projection, and feature selection, the project aims to simplify the
data while preserving important information. These techniques
are assessed based on their ability to preserve model performance,
reduce training time, and identify the most significant features in
the dataset.

## Report

The report is available as a PDF called `Raphael_Fontaine_Report.pdf`.

## Prerequisites

The project contains mostly Jupyter Notebooks. They can be consulted without running them. However, to run them or regenerate the dataset, you need:

- A way to open [Jupyter Notebooks](https://jupyter.org/) (VS Code, Google Colab, etc...)
- Python 3.12 (not tested on older versions)
- Python packages installed:
```bash
# in the /src folder
pip install -r requirements.txt
```

## Dataset
The dataset has two sources: [spotrac](spotrac.com) and [Moneypuck](https://moneypuck.com/data.htm). The data from Moneypuck has been downloaded manually from [here](https://moneypuck.com/data.htm) and is available in `/src/data`. The data from [spotrac](spotrac.com) is downloaded by a script. The dataset is called `dataset.csv` and is already built and located in `/src/data`. It can be regenerated if needed using the following commands:
```bash
# in /src/data
python spotrac.py
python merge_salary_info.py
python create_dataset.py
```

## Code

The source code is located in `/src` and is divided into multiple Jupyter Notebooks. There is also a common python script (`common.py`) that has multiple functions shared by the Notebooks. Here is a list of the Notebooks:
- Baseline: `baseline.py`
- Principal Component Analysis: `pca.py`
- Random Projection: `random_projection.py`
- Partial Least Squares: `pls.py`
- Feature Selection: `feature_selection.py`
- Case Studies: `case_studies.py`


_NOTE: The Jupyter Notebooks are also available as rendered PDFs in the `/output` folder._