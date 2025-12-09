# Customer Churn Prediction Project

This repository includes all the code necessary to predict customer churn based on synthetic customer data for the purpose of demonstrating the ML skills I have learned in MAT311 this semester.

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   └── raw/
|       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── customer_churn.ipynb
|   ├── DataCleaningTest.ipynb
|   └── NotebookThingy.ipynb
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py
    │   └── split_data.py
    ├── models/
    │   ├── train_model.py
    │   ├── dumb_model.py
    │   ├── knn_model.py
    |   └── rand_forest_model.py
    ├── utils/
    │   └── helper_functions.py
    └── visualization/
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results.

## Running the script

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n customer_churn --file requirements.txt
conda activate customer_churn
python main.py
```

This will load the dataset, perform basic data cleaning, train a simple model and produce visualizations similar to those in the notebook.
The cleaned data will be written to `data/processed/` and all plots will be displayed interactively.
