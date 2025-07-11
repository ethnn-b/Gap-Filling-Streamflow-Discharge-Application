import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import core logic from local scripts
from missforest_imputer import ModifiedMissForest
from model_configurations import (
    train_full_model,
    train_no_contributor_model,
)
# Corrected import: build_connectivity_matrix now takes vcode_to_station_name
from data_processing import build_distance_matrix, build_connectivity_matrix 

def create_continuous_gaps(data, gap_lengths, random_seed=42):
    """
    Creates continuous missing gaps in the data for evaluation purposes.
    Args:
        data (pd.DataFrame): The original, complete data to introduce gaps into.
        gap_lengths (list): A list of integers, each representing a desired gap length in days.
        random_seed (int): Seed for reproducibility of gap generation.
    Returns:
        dict: A dictionary where keys are gap lengths and values are dicts containing:
              'mask': A boolean mask of where gaps were introduced.
              'gapped_data': The DataFrame with simulated continuous gaps.
              'true_values': The true values from `data` at the masked locations.
    """
    gap_results = {}
    np.random.seed(random_seed)

    discharge_cols = [col for col in data.columns if not col.startswith('day_of_year_')]
    num_station_cols = len(discharge_cols)
    
    if num_station_cols == 0:
        st.warning("No discharge columns found for creating gaps. Skipping gap creation.")
        return {}

    for length in gap_lengths:
        data_gapped = data.copy()
        mask_full_df = np.zeros(data.shape, dtype=bool)
        
        for col_idx, col_name in enumerate(discharge_cols):
            num_rows = data.shape[0]
            n_gaps = int(num_rows / (length * 10)) if length > 0 else 0
            
            if num_rows - length <= 0 or n_gaps == 0:
                continue
            
            valid_start_indices = np.arange(num_rows - length + 1)
            if len(valid_start_indices) < n_gaps:
                n_gaps = len(valid_start_indices)
            
            if n_gaps > 0:
                gap_starts = np.random.choice(
                    valid_start_indices,
                    size=n_gaps,
                    replace=False
                )
                
                original_col_idx = data.columns.get_loc(col_name)
                for start in gap_starts:
                    mask_full_df[start:start+length, original_col_idx] = True
        
        gapped_data_temp = data.copy()
        gapped_data_temp[mask_full_df] = np.nan

        if 'day_of_year_sin' in data.columns and 'day_of_year_cos' in data.columns:
            gapped_data_temp['day_of_year_sin'] = data['day_of_year_sin']
            gapped_data_temp['day_of_year_cos'] = data['day_of_year_cos']
        
        true_values_for_masked_discharge = data[discharge_cols].values[mask_full_df[:, :num_station_cols]]

        gap_results[length] = {
            'mask': mask_full_df[:, :num_station_cols],
            'gapped_data': gapped_data_temp,
            'true_values': true_values_for_masked_discharge
        }
    return gap_results

def evaluate_metrics(y_true, y_pred):
    """
    Calculates RMSE, NRMSE, NSE, and R2 for imputation performance.
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted (imputed) values.
    Returns:
        dict: Dictionary of calculated metrics.
    """
    from sklearn.metrics import r2_score, mean_squared_error
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'NRMSE': np.nan, 'NSE': np.nan, 'R2': np.nan}

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    range_true = y_true_clean.max() - y_true_clean.min()
    nrmse = rmse / (range_true + 1e-6) if range_true > 0 else np.nan
    
    numerator = np.sum((y_true_clean - y_pred_clean) ** 2)
    denominator = np.sum((y_true_clean - y_true_clean.mean()) ** 2)
    nse = 1 - (numerator / denominator) if denominator != 0 else float('-inf')
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    return {'RMSE': rmse, 'NRMSE': nrmse, 'NSE': nse, 'R2': r2}

def _plot_results_for_streamlit(y_true, y_pred, gap_length, title_suffix=""):
    """
    Generates diagnostic plots for imputation performance and returns matplotlib figures.
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_plot = y_true[valid_mask]
    y_pred_plot = y_pred[valid_mask]

    if len(y_true_plot) == 0:
        st.warning(f"No valid data points for plotting for {gap_length}-day gaps {title_suffix}.")
        return []

    figures = []

    # Plot 1: True vs Predicted & Error Distribution
    fig1, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.regplot(x=y_true_plot, y=y_pred_plot, line_kws={'color': 'red'}, scatter_kws={'alpha':0.6}, ax=axes[0])
    axes[0].plot([min(y_true_plot.min(), y_pred_plot.min()), max(y_true_plot.max(), y_pred_plot.max())],
                 [min(y_true_plot.min(), y_pred_plot.min()), max(y_true_plot.max(), y_pred_plot.max())], '--k', label='1:1 Line')
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'True vs Predicted ({gap_length}-day gaps) {title_suffix}')
    axes[0].legend()
    
    errors = y_pred_plot - y_true_plot
    sns.histplot(errors, kde=True, bins=30, ax=axes[1])
    axes[1].set_xlabel('Prediction Error (Predicted - True)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    
    plt.tight_layout()
    figures.append(fig1)
    
    # Plot 2: Time Series Example
    fig2 = plt.figure(figsize=(12, 4))
    plot_points_count = min(200, len(y_true_plot))
    
    plt.plot(range(plot_points_count), y_true_plot[:plot_points_count], 'o', label='True', markersize=4, alpha=0.7)
    plt.plot(range(plot_points_count), y_pred_plot[:plot_points_count], 'x', label='Predicted', markersize=4, alpha=0.7)
    plt.xlabel('Sample Index (within gapped data)')
    plt.ylabel('Discharge Value')
    plt.legend()
    plt.title(f'Example Predictions for {gap_length}-day Gaps (First {plot_points_count} points) {title_suffix}')
    plt.grid(True, linestyle=':', alpha=0.7)
    figures.append(fig2)
    
    return figures

def run_evaluation_for_streamlit(model, test_data, gap_lengths, model_type):
    """
    Evaluates a trained imputation model on test data with various gap lengths.
    Generates performance metrics and returns a list of matplotlib figures.
    """
    evaluation_metrics_by_gap_length = {}
    all_model_plots = [] # To collect all plots generated by this model's evaluation

    gap_data_sets = create_continuous_gaps(test_data, gap_lengths, random_seed=42)

    for length, gap_dict in gap_data_sets.items():
        st.write(f"  Evaluating for {length}-day gaps...")
        
        imputed_data = model.transform(gap_dict['gapped_data'])
        
        y_true = gap_dict['true_values']
        discharge_cols_imputed = [col for col in imputed_data.columns if not col.startswith('day_of_year_')]
        y_pred = imputed_data[discharge_cols_imputed].values[gap_dict['mask']]
        
        metrics = evaluate_metrics(y_true, y_pred)
        evaluation_metrics_by_gap_length[length] = metrics
        
        st.write(f"    Metrics for {length}-day gaps: {metrics}")
        
        if not np.all(np.isnan(y_true)):
            figs = _plot_results_for_streamlit(y_true, y_pred, length, title_suffix=f"({model_type})")
            for fig in figs:
                all_model_plots.append(fig) # Collect figures
                st.pyplot(fig) # Display figure in Streamlit
                plt.close(fig) # Close figure to free memory
        else:
            st.info(f"    Skipping plot for {length}-day gaps: no valid true values to plot.")
            
    results_df = pd.DataFrame(evaluation_metrics_by_gap_length).T
    return results_df, all_model_plots

# Removed: plot_full_data function (no longer needed for main page or final imputed data plot)

# --- Main Analysis Logic (Adapted from run_all_evaluations.py) ---
# This function now expects pre-loaded dataframes from session_state
def run_full_analysis(
    df_discharge_with_features_raw, df_connectivity_filtered, df_coords, vcode_to_station_name, station_name_to_vcode,
    train_start_year, train_end_year,
    test_start_year, test_end_year
):
    all_results = {}
    all_plots = [] # Plots will still be generated and collected, but not displayed by this function
    summary_messages = []

    st.subheader("Starting Hydrological Data Imputation Evaluation...")
    summary_messages.append("--- Comprehensive Model Evaluation Summary ---")
    summary_messages.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    summary_messages.append("="*80)

    temporal_features = ['day_of_year_sin', 'day_of_year_cos']

    # 2. Build Spatial Matrices
    st.markdown("### Step 2: Building Distance and Connectivity Matrices")
    actual_station_cols_from_df_raw = [col for col in df_discharge_with_features_raw.columns if not col.startswith('day_of_year_')]

    distance_matrix_original = build_distance_matrix(df_coords, actual_station_cols_from_df_raw)
    
    is_connectivity_info_available = not df_connectivity_filtered.empty and vcode_to_station_name # Check if mapping exists
    if is_connectivity_info_available:
        # Pass vcode_to_station_name to build_connectivity_matrix as it's now needed for mapping
        connectivity_matrix_original = build_connectivity_matrix(df_connectivity_filtered, actual_station_cols_from_df_raw, vcode_to_station_name)
    else:
        connectivity_matrix_original = pd.DataFrame(0, index=actual_station_cols_from_df_raw,
                                                    columns=actual_station_cols_from_df_raw, dtype=int)
        st.info("Using a dummy (all zeros) connectivity matrix due to missing connectivity information or V-code mapping.")
        summary_messages.append("Using a dummy (all zeros) connectivity matrix due to missing connectivity information or V-code mapping.")


    if distance_matrix_original.empty:
        st.error("FATAL ERROR: Distance matrix is empty. Cannot proceed with imputation.")
        return all_results, all_plots, summary_messages

    gap_lengths = [5, 10, 20, 30]

    period_label = f"Train_{train_start_year}-{train_end_year}_Test_{test_start_year}-{test_end_year}"
    st.markdown(f"### Processing Period: {period_label}")
    summary_messages.append(f"\n--- Processing Period: {period_label} ---")
    
    # 3. Data Partitioning for current period
    df_train_full_period = df_discharge_with_features_raw[(df_discharge_with_features_raw.index.year >= train_start_year) & \
                                                          (df_discharge_with_features_raw.index.year <= train_end_year)]
    df_test_full_period = df_discharge_with_features_raw[(df_discharge_with_features_raw.index.year >= test_start_year) & \
                                                         (df_discharge_with_features_raw.index.year <= test_end_year)]

    if df_train_full_period.empty or df_test_full_period.empty:
        st.error(f"Skipping period {period_label}: Training or Test data is empty after year slicing. Check your date ranges and data.")
        summary_messages.append(f"Skipping period {period_label}: Training or Test data is empty after year slicing. Check your date ranges and data.")
        return all_results, all_plots, summary_messages

    discharge_cols_only_train_period = [col for col in df_train_full_period.columns if col not in temporal_features]
    min_non_na_threshold_period = int(0.9 * len(df_train_full_period))
    
    cols_to_keep_discharge_period = df_train_full_period[discharge_cols_only_train_period].columns[
        df_train_full_period[discharge_cols_only_train_period].notna().sum() >= min_non_na_threshold_period
    ].tolist()
    
    cols_to_keep_period = list(set(cols_to_keep_discharge_period + temporal_features))

    df_train_period = df_train_full_period[cols_to_keep_period]
    df_test_period = df_test_full_period[cols_to_keep_period]
    
    final_actual_station_cols_for_imputer_period = [col for col in cols_to_keep_period if col not in temporal_features]
    
    if not final_actual_station_cols_for_imputer_period:
        st.error(f"Skipping period {period_label}: No discharge columns left after filtering for non-NA threshold in training data.")
        summary_messages.append(f"Skipping period {period_label}: No discharge columns left after filtering for non-NA threshold in training data.")
        return all_results, all_plots, summary_messages

    distance_matrix_filtered_period = distance_matrix_original.loc[final_actual_station_cols_for_imputer_period, final_actual_station_cols_for_imputer_period]
    connectivity_matrix_filtered_period = connectivity_matrix_original.loc[final_actual_station_cols_for_imputer_period, final_actual_station_cols_for_imputer_period]
    
    if distance_matrix_filtered_period.empty:
        st.error(f"Skipping period {period_label}: Filtered distance matrix is empty.")
        summary_messages.append(f"Skipping period {period_label}: Filtered distance matrix is empty.")
        return all_results, all_plots, summary_messages

    discharge_cols_train_to_mask_period = [col for col in df_train_period.columns if col not in temporal_features]
    df_train_discharge_only_period = df_train_period[discharge_cols_train_to_mask_period].fillna(df_train_period[discharge_cols_train_to_mask_period].mean())
    
    np.random.seed(42)
    train_mask_period = np.random.rand(*df_train_discharge_only_period.shape) < 0.1
    
    df_train_masked_period = df_train_period.copy()
    df_train_masked_period[discharge_cols_train_to_mask_period] = df_train_discharge_only_period.mask(train_mask_period)
    
    st.write(f"Simulated 10% random missingness in training data for model learning ({period_label}).")
    summary_messages.append(f"Simulated 10% random missingness in training data for model learning ({period_label}).")

    models_to_run = {}
    if is_connectivity_info_available:
        st.info("Connectivity data detected. Running only 'Full Model (Inverse Weighting)'.")
        summary_messages.append("Connectivity data detected. Running only 'Full Model (Inverse Weighting)'.")
        models_to_run["Full Model (Inverse Weighting)"] = {
            "train_fn": train_full_model,
            "train_args": (df_train_masked_period, distance_matrix_filtered_period, connectivity_matrix_filtered_period, temporal_features)
        }
    else:
        st.info("No connectivity data provided. Running only 'No Connectivity Info Model'.")
        summary_messages.append("No connectivity data provided. Running only 'No Connectivity Info Model'.")
        models_to_run["No Connectivity Info Model"] = { # Renamed for display
            "train_fn": train_no_contributor_model, # Keeping original function name as it's from an external file
            "train_args": (df_train_masked_period, distance_matrix_filtered_period, connectivity_matrix_filtered_period, temporal_features)
        }

    current_period_results_for_plotting = {}

    for model_name, config in models_to_run.items():
        st.markdown(f"#### Evaluating: {model_name}")
        summary_messages.append(f"\n--- Evaluating: {model_name} ---")
        
        trained_model = config["train_fn"](*config["train_args"])

        current_model_results_df, model_specific_plots = run_evaluation_for_streamlit(
            model=trained_model,
            test_data=df_test_period,
            gap_lengths=gap_lengths,
            model_type=f"{model_name} ({period_label})"
        )
        current_period_results_for_plotting[model_name] = current_model_results_df
        all_plots.extend(model_specific_plots) # Collect plots from each model evaluation

        st.write(f"Metrics for {model_name}:")
        st.dataframe(current_model_results_df.round(4))
        summary_messages.append(f"Metrics for {model_name}:\n{current_model_results_df.to_string(float_format='%.4f')}")

    # Generate a comparative plot for the models within this specific period
    # This will still work even if only one model is run, showing a single line.
    st.markdown(f"### Comparative Plot for Period: {period_label}")
    summary_messages.append(f"\n--- Generating Comparative Plot for Period: {period_label} ---")
    
    fig_comp, axes = plt.subplots(1, 3, figsize=(18, 7))
    metrics_to_plot = ['RMSE', 'NSE', 'R2']
    
    colors = plt.cm.get_cmap('tab10', len(models_to_run)).colors
    line_styles = ['-','--', '-.', ':']
    markers = ['o', 'x', 's', '^', 'D']

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        for j, (model_name, results_df) in enumerate(current_period_results_for_plotting.items()):
            if metric in results_df.columns:
                ax.plot(results_df.index, results_df[metric], 
                         marker=markers[j % len(markers)], 
                         linestyle=line_styles[j % len(line_styles)], 
                         label=model_name, 
                         color=colors[j % len(colors)])
            
        ax.set_xlabel('Gap Length (days)')
        ax.set_ylabel(metric)
        ax.set_xticks(gap_lengths)
        ax.set_title(f'{metric} Comparison for {period_label}')
        ax.grid(True, linestyle=':', alpha=0.7)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    all_plots.append(fig_comp) # Add comparative plot to list
    st.pyplot(fig_comp) # Display the plot
    plt.close(fig_comp) # Close the figure to free memory

    st.success("Analysis complete!")
    summary_messages.append("\n--- All Evaluations Complete ---")

    return all_results, all_plots, summary_messages
