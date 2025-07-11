# Hydrological Discharge Data Imputation Application

This repository presents a Streamlit-powered web application designed for the imputation and comprehensive evaluation of missing data in hydrological discharge time series. It utilizes a custom **Modified MissForest** imputation model that intelligently incorporates spatial (geodesic distance) and hydrological (connectivity) information, alongside temporal features, to provide accurate and robust estimates for missing discharge values.

## ✨ Key Features

* **Interactive User Interface:** A user-friendly Streamlit application enables seamless data upload, flexible parameter configuration, and clear visualization of results.

* **Versatile Data Integration:**
    * Accepts daily hydrological discharge data (CSV).
    * Integrates station location data (latitude/longitude CSV) for spatial calculations.
    * Supports optional hydrological connectivity data (CSV, using original V-code identifiers) to model upstream-downstream relationships.

* **Automated Data Preprocessing:** Handles data cleaning, standardizes station names, parses geographical coordinates, and generates cyclical temporal features (day of year sine/cosine) to capture seasonal patterns.

* **Dynamic Model Selection:** The application intelligently adapts its imputation strategy based on available data:
    * **Full Model (Inverse Weighting):** Utilizes geodesic distance, hydrological connectivity, and temporal features when connectivity data is provided.
    * **No Connectivity Info Model:** Falls back to using only geodesic distance and temporal features if connectivity data is absent.

* **Robust Imputation Engine:** Features a custom **Modified MissForest** algorithm, an iterative imputation method based on Random Forests. This model is enhanced with spatial and hydrological weighting mechanisms for improved accuracy in hydrological contexts.

* **Comprehensive Performance Evaluation:**
    * Simulates continuous missing data gaps (e.g., 5, 10, 20, 30 days) in the test dataset.
    * Calculates key performance metrics: Root Mean Squared Error (RMSE), Normalized Root Mean Squared Error (NRMSE), Nash-Sutcliffe Efficiency (NSE), and the coefficient of determination ($R^2$).
    * Generates diagnostic plots (True vs. Predicted, Error Distribution, Time Series Examples) for each gap length, aiding in model understanding.
    * Presents comparative plots summarizing metrics across different models and time periods.

* **Detailed Output Summaries:** Provides comprehensive textual summaries of all evaluation results, facilitating easy analysis and reporting.

## 🛠️ Technologies Used

* **Python 3.x**
* **Streamlit:** For building the interactive web application.
* **Pandas:** For efficient data manipulation and analysis.
* **NumPy:** For high-performance numerical operations.
* **Scikit-learn:** Provides the foundational `RandomForestRegressor` for the imputation model.
* **Matplotlib & Seaborn:** For generating informative data visualizations.
* **Geopy:** Enables accurate geodesic distance calculations.

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites

Ensure you have Python 3.x installed on your system.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    *(Replace `<repository_url>` with your actual GitHub repository URL and `<repository_name>` with the name of your cloned directory.)*

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    # Activate the virtual environment:
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file in the root of your repository with the following content:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    geopy
    openpyxl # Optional: If your data processing involves .xlsx files
    xlrd # Optional: If your data processing involves older .xls files
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Data Setup

1.  **Place your data files:**
    Ensure your input data files (`discharge_data_cleaned.csv`, `lat_long_discharge.csv`, and `mahanadi_contribs.csv` - the latter being optional) are placed in the same directory as `app.py`. Alternatively, you can use the file uploader within the Streamlit application.

    * **`discharge_data_cleaned.csv`**: Contains daily discharge values. Expected columns: `Date`, followed by station columns (e.g., `StationA`, `StationB`).
    * **`lat_long_discharge.csv`**: Contains station names and their corresponding latitude/longitude. Expected columns: `Name of site`, `Latitude (N)`, `Longitude (E)`.
    * **`mahanadi_contribs.csv` (Optional)**: Contains hydrological connectivity information. Expected: V-codes as index/columns, with a 'Name of site' column for mapping.

### 4. Running the Application

1.  **Start the Streamlit application:**
    With your virtual environment activated, run:
    ```bash
    streamlit run app.py
    ```
    This command will automatically open the application in your default web browser.

## 📂 Project Structure
```
├── app.py                      # Main Streamlit application entry point
├── data_processing.py          # Handles data loading, cleaning, and preprocessing
├── model_evaluation.py         # Functions for model evaluation, metrics, and plotting
├── missforest_imputer.py       # Implementation of the custom ModifiedMissForest algorithm
├── model_configurations.py     # Defines and trains different imputation model variants
├── (your_data_files)/          # Directory for input data files (e.g., CSVs)
│   ├── discharge_data_cleaned.csv
│   ├── lat_long_discharge.csv
│   └── mahanadi_contribs.csv   # (Optional)
└── requirements.txt            # Lists all Python dependencies
```

##  Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

##  Acknowledgements

This project builds upon established hydrological modeling and machine learning techniques, benefiting greatly from the open-source community's tools and libraries.

Special acknowledgement to **Prof. Maheswaran R.** : Assistant Professor, Civil Engineering, IIT-Hyderabad, for his guidance and invaluable support.

