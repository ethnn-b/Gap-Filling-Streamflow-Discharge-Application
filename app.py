import streamlit as st
import pandas as pd
import io # For handling file uploads
from datetime import datetime
import warnings

# Suppress specific warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import functions from new modular scripts
from data_processing import _load_and_preprocess_data_for_streamlit, add_temporal_features
from model_evaluation import run_full_analysis
# Removed: from eda_plots import show_eda_page_content

# --- Main Page Content ---
def show_main_analysis_page():
    st.title("💧 Hydrological Data Imputation and Model Evaluation")
    st.markdown("""
        This application allows you to upload hydrological discharge data, define training and testing periods, 
        and evaluate imputation models.
        
        The model selection is dynamic:
        - If **Connectivity Data** is provided, the **Full Model (Inverse Weighting)** will be used, incorporating spatial (geodesic distance) and hydrological (connectivity) weights, along with temporal features.
        - If **Connectivity Data** is **not** provided, the **No Connectivity Info Model** will be used, relying on spatial (geodesic distance) and temporal features only.
        
        Upload your data files below and configure the evaluation parameters.
    """)

    # How it Works / Flowchart section
    st.markdown("---")
    with st.expander("How the Model Works: Detailed Workflow"):
        st.markdown("""
            This flowchart illustrates the step-by-step process, from data upload to model evaluation and final data imputation, within the application.
            Click on the image to enlarge.
            """)
        # Updated Mermaid flowchart to reflect current functionality (no EDA, specific model selection)
        st.components.v1.html(
            """
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({startOnLoad:true});</script>
            <div class="mermaid">
                graph TD
                    A[Start Application] --> B(User Uploads Data Files);
                    B --> B1{Discharge Data<br>CSV: Date, Station1, Station2...};
                    B --> B2{Connectivity Data<br>CSV: Vcode, name of site, lat, lon, Vcode1... (Optional)};
                    B --> B3{Station Location Data<br>CSV: Name of site, Latitude (N), Longitude (E)};

                    B --> C{Data Loading & Initial Preprocessing};
                    C --> C1[Parse Dates, Clean Station Names];
                    C --> C2[Filter to Common Stations Across Files];
                    C --> C3[Add Cyclical Temporal Features (Day of Year Sin/Cos)];

                    C3 --> F{Build Spatial Matrices};
                    F --> F1[Build Geodesic Distance Matrix];
                    F --> F2{Check Connectivity Data Availability};
                    F2 -- Yes --> F3[Build Hydrological Connectivity Matrix];
                    F2 -- No --> F4[Use Dummy Connectivity Matrix (All Zeros)];

                    F3 --> G{Define Training & Testing Periods};
                    F4 --> G;
                    G --> G1[User Input: Train Start/End Years, Test Start/End Years];
                    G --> G2[Partition Data into Training and Testing Sets];

                    G2 --> H{Create Artificial Gaps in Training Data};
                    H --> H1[Simulate 10% Random Missingness for Model Learning];

                    H1 --> I{Model Training & Evaluation};
                    I --> I1{Conditional Model Selection};
                    I1 -- Connectivity Available --> I2[Train Full Model<br>(Distance, Connectivity, Temporal)];
                    I1 -- No Connectivity --> I3[Train No Connectivity Model<br>(Distance, Temporal only)];

                    I2 --> J[Evaluate Model Performance];
                    I3 --> J;
                    J --> J1[Introduce Continuous Gaps (5, 10, 20, 30 days) in Test Data];
                    J --> J2[Impute Gapped Test Data using Trained Model];
                    J --> J3[Calculate Metrics: RMSE, NRMSE, NSE, R2];
                    J --> J4[Generate Diagnostic Plots (True vs Predicted, Error Dist, Time Series)];

                    J4 --> K[Collect All Model Results & Plots];

                    K --> N[Display Summary of Results];
                    N --> O[End Application];
            </div>
            """,
            height=800, # Adjust height as needed to fit the flowchart
            scrolling=True
        )
    st.markdown("---")

    st.sidebar.header("Upload Data Files")

    # Info button for data format
    if st.sidebar.button("Data Format Info"):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Expected Data Formats:")
        
        st.sidebar.markdown("**1. Discharge Data**")
        st.sidebar.markdown(
            """
            - **Purpose:** Contains daily discharge values for various hydrological stations.
            - **Format:** CSV file.
            - **Required Columns:**
                - `Date`: Date column in `DD/MM/YYYY` format (e.g., `01/01/2008`).
                - Other columns: Numeric discharge values for each station. Column headers should be unique station names.
            - **Example:**
            ```csv
            Date,Station_A,Station_B,Station_C
            01/01/2008,10.5,20.1,5.0
            02/01/2008,11.2,21.5,5.2
            ...
            ```
            """
        )

        st.sidebar.markdown("**2. Connectivity Data (Optional)**")
        st.sidebar.markdown(
            """
            - **Purpose:** Defines hydrological connectivity or upstream-downstream relationships between stations. Used for the 'Full Model'.
            - **Format:** CSV file.
            - **Required Columns:**
                - First column (index): Unique identifier (e.g., V-code, Site ID) for each station. This will be used as the DataFrame index.
                - `name of site`: The corresponding station name (must exactly match names in your Discharge Data columns).
                - `latitude (n)`: Latitude of the station (required for internal mapping, but not directly used for distance calculation in this file).
                - `longitude (e)`: Longitude of the station (required for internal mapping, but not directly used for distance calculation in this file).
                - Other columns: Unique identifiers (V-codes/Site IDs) of other stations. Values indicate connectivity (e.g., 1 for connected, 0 for not).
            - **Example:**
            ```csv
            SiteID,name of site,latitude (n),longitude (e),SiteID1,SiteID2,SiteID3
            S1,Station_A,21.0 N,82.0 E,0,1,0
            S2,Station_B,20.5 N,81.5 E,1,0,1
            ...
            ```
            """
        )

        st.sidebar.markdown("**3. Station Location Data**")
        st.sidebar.markdown(
            """
            - **Purpose:** Provides geographical coordinates (latitude and longitude) for each station.
            - **Format:** CSV file.
            - **Required Columns:**
                - `Name of site`: The station name (must exactly match names in your Discharge Data columns).
                - `Latitude (N)`: Latitude of the station. Can be in decimal degrees (e.g., `21.45`) or Degrees-Minutes-Seconds (DMS) format (e.g., `21o 50’ 02’’ N`).
                - `Longitude (E)`: Longitude of the station. Can be in decimal degrees (e.g., `81.90`) or DMS format (e.g., `81o 54’ 00’’ E`).
            - **Example:**
            ```csv
            Name of site,Latitude (N),Longitude (E)
            Station_A,21.45,82.00
            Station_B,20o 50' 30'' N,81o 30' 15'' E
            ...
            ```
            """
        )
        st.sidebar.markdown("---")


    discharge_file = st.sidebar.file_uploader("Upload Discharge Data", type=["csv"], key="discharge_uploader")
    connectivity_file = st.sidebar.file_uploader("Upload Connectivity Data (Optional)", type=["csv"], key="connectivity_uploader")
    lat_long_file = st.sidebar.file_uploader("Upload Station Location Data", type=["csv"], key="lat_long_uploader")

    # Store uploaded files in session state
    if discharge_file:
        st.session_state['discharge_file_buffer'] = discharge_file
    if connectivity_file:
        st.session_state['connectivity_file_buffer'] = connectivity_file
    else: # Clear if no file is uploaded or if it's explicitly removed
        if 'connectivity_file_buffer' in st.session_state:
            del st.session_state['connectivity_file_buffer']
    if lat_long_file:
        st.session_state['lat_long_file_buffer'] = lat_long_file

    st.sidebar.header("Define Time Periods")
    st.sidebar.subheader("Training Period")
    train_start_year = st.sidebar.number_input("Train Start Year", min_value=1900, max_value=datetime.now().year, value=2008, step=1, key="train_start_year")
    train_end_year = st.sidebar.number_input("Train End Year", min_value=1900, max_value=datetime.now().year, value=2012, step=1, key="train_end_year")

    st.sidebar.subheader("Testing Period")
    test_start_year = st.sidebar.number_input("Test Start Year", min_value=1900, max_value=datetime.now().year, value=2013, step=1, key="test_start_year")
    test_end_year = st.sidebar.number_input("Test End Year", min_value=1900, max_value=datetime.now().year, value=2015, step=1, key="test_end_year")

    if st.sidebar.button("Run Analysis", key="run_analysis_button"):
        if 'discharge_file_buffer' not in st.session_state or 'lat_long_file_buffer' not in st.session_state:
            st.sidebar.error("Please upload 'Discharge Data' and 'Station Location Data' to proceed.")
        elif train_start_year >= train_end_year:
            st.sidebar.error("Train Start Year must be less than Train End Year.")
        elif test_start_year >= test_end_year:
            st.sidebar.error("Test Start Year must be less than Test End Year.")
        elif test_start_year <= train_end_year:
            st.sidebar.error("Test Period must start after the Training Period ends.")
        else:
            with st.spinner("Loading and preprocessing data..."):
                try:
                    # Explicitly initialize these variables before the function call
                    # This is a safeguard against potential NameErrors if unpacking fails unexpectedly
                    # These are now initialized in main_streamlit_app()
                    # vcode_to_station_name = {}
                    # station_name_to_vcode = {}

                    df_discharge, df_connectivity_filtered, df_coords, vcode_to_station_name_local, station_name_to_vcode_local = \
                        _load_and_preprocess_data_for_streamlit(
                            st.session_state['discharge_file_buffer'],
                            st.session_state.get('connectivity_file_buffer', io.BytesIO(b'')), # Pass empty BytesIO for optional file
                            st.session_state['lat_long_file_buffer']
                        )
                    st.session_state['df_discharge_raw'] = df_discharge # Store raw processed data
                    st.session_state['df_discharge_raw_with_features'] = add_temporal_features(df_discharge.copy()) # Store with features
                    st.session_state['df_connectivity_filtered'] = df_connectivity_filtered
                    st.session_state['df_coords'] = df_coords
                    st.session_state['vcode_to_station_name'] = vcode_to_station_name_local # Store in session state
                    st.session_state['station_name_to_vcode'] = station_name_to_vcode_local # Store in session state
                    st.session_state['data_loaded'] = True
                    st.success("Data loaded and preprocessed successfully!")
                except Exception as e:
                    st.error(f"Error during data loading/preprocessing: {e}")
                    st.session_state['data_loaded'] = False
                    return

            if st.session_state.get('data_loaded', False):
                with st.spinner("Running analysis... This might take a while, please wait."):
                    results, plots, summary_messages = run_full_analysis(
                        st.session_state['df_discharge_raw_with_features'], # Pass preprocessed data
                        st.session_state['df_connectivity_filtered'],
                        st.session_state['df_coords'],
                        st.session_state['vcode_to_station_name'], # Pass from session state
                        st.session_state['station_name_to_vcode'], # Pass from session state
                        train_start_year, train_end_year,
                        test_start_year, test_end_year
                    )
                    st.session_state['analysis_results'] = results
                    st.session_state['analysis_plots'] = plots
                    st.session_state['analysis_summary'] = summary_messages
                    st.session_state['analysis_run'] = True
            else:
                st.warning("Analysis cannot run without successfully loaded data.")

    # Removed: "All Generated Plots" section
    if st.session_state.get('analysis_run', False):
        st.success("Analysis completed successfully!")
        st.header("Summary of Results")
        st.text_area("Detailed Summary", "\n".join(st.session_state['analysis_summary']), height=300)
        
        # Removed the header and info about "All Generated Plots"
        # st.header("All Generated Plots")
        # st.info("Plots for individual model evaluations and the final comparative plot (if applicable) are displayed below.")
        # Plots are now displayed directly within the model_evaluation functions,
        # so this summary section is no longer needed.

# --- Main Streamlit App Logic ---
def main_streamlit_app():
    st.set_page_config(layout="wide", page_title="Hydrological Data Imputation")

    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'Model Analysis'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False

    # Explicitly initialize vcode_to_station_name and station_name_to_vcode in session state
    # This ensures they are always defined, preventing NameError
    if 'vcode_to_station_name' not in st.session_state:
        st.session_state['vcode_to_station_name'] = {}
    if 'station_name_to_vcode' not in st.session_state:
        st.session_state['station_name_to_vcode'] = {}

    st.sidebar.title("Navigation")
    # Removed "EDA" from the radio button options
    page_selection = st.sidebar.radio("Go to", ["Model Analysis"]) 

    if page_selection == "Model Analysis":
        st.session_state.page = 'Model Analysis'
    # Removed EDA page selection logic
    # elif page_selection == "EDA":
    #     st.session_state.page = 'EDA'

    if st.session_state.page == 'Model Analysis':
        show_main_analysis_page()
    # Removed EDA page content display
    # elif st.session_state.page == 'EDA':
    #     show_eda_page_content()

# Run the Streamlit app
if __name__ == "__main__":
    main_streamlit_app()
