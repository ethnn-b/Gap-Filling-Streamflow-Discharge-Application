# missforest_imputer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
import pickle # Used for saving/loading the imputer object

warnings.filterwarnings('ignore')

class ModifiedMissForest:
    """
    A custom imputation class inspired by MissForest, incorporating
    spatial (geodesic distance) and hydrological (connectivity) weights
    when training RandomForest regressors for imputation.
    """
    def __init__(self, distance_matrix, connectivity, max_iter=10, n_estimators=100, random_state=42,
                 distance_weighting_type='inverse', decay_rate=0.1, temporal_feature_columns=None):
        """
        Initializes the ModifiedMissForest imputer.

        Args:
            distance_matrix (pd.DataFrame): DataFrame of geodesic distances between stations.
            connectivity (pd.DataFrame): DataFrame indicating hydrological connectivity.
            max_iter (int): Maximum number of imputation iterations.
            n_estimators (int): Number of trees in each RandomForestRegressor.
            random_state (int): Seed for random number generation for reproducibility.
            distance_weighting_type (str): Type of distance weighting ('inverse' or 'exponential').
            decay_rate (float): Decay rate for exponential weighting. Only used if distance_weighting_type='exponential'.
            temporal_feature_columns (list): List of column names that represent temporal or exogenous features
                                             that should be used directly (unweighted) as predictors.
        """
        self.distance_matrix = distance_matrix
        self.connectivity = connectivity
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}  # Stores trained RandomForest models for each station
        self.col_means = {} # Stores column means (from training data) for initial imputation/fallback
        self.col_names = None # Will be set during fit; stores the order of all columns (discharge + temporal features)
        self.discharge_columns = None # Will be set during fit; stores only discharge station names
        self.temporal_feature_columns = temporal_feature_columns if temporal_feature_columns is not None else []
        
        self.site_to_idx = None # Will be set during fit; maps column name to its index in internal data representation
        self.distance_weighting_type = distance_weighting_type
        self.decay_rate = decay_rate

        self.plot_output_dir = None # Set by train.py/eval.py for plot saving

    def _calculate_weights(self, target_station, station_predictors):
        """
        Calculates blended weights for *station-based* predictor columns for a given target station.
        Weights combine distance-based weights (inverse or exponential) and hydrological connectivity.
        Temporal features are NOT processed here; they are handled separately as unweighted predictors.

        Args:
            target_station (str): The name of the discharge station being imputed.
            station_predictors (list): A list of column names representing other discharge stations
                                       that will be used as predictors.
        Returns:
            pd.Series: Weights for `station_predictors` indexed by their names.
        """
        if target_station not in self.distance_matrix.index:
            return pd.Series(0.0, index=station_predictors)

        actual_predictors_dist = [p for p in station_predictors if p in self.distance_matrix.columns]
        if not actual_predictors_dist:
            return pd.Series(0.0, index=station_predictors)

        distances = self.distance_matrix.loc[target_station, actual_predictors_dist]

        if self.distance_weighting_type == 'inverse':
            dist_weights = 1 / (distances + 1e-9)
            dist_weights = dist_weights.fillna(0).replace([np.inf, -np.inf], 0)
        elif self.distance_weighting_type == 'exponential':
            dist_weights = np.exp(-self.decay_rate * distances)
            dist_weights = dist_weights.fillna(0)
        else:
            raise ValueError(f"Unknown distance_weighting_type: {self.distance_weighting_type}")

        actual_predictors_conn = [p for p in station_predictors if p in self.connectivity.columns]
        if target_station in self.connectivity.index and actual_predictors_conn:
            connectivity_weights = self.connectivity.loc[target_station, actual_predictors_conn]
        else:
            connectivity_weights = pd.Series(0.0, index=actual_predictors_conn)

        # Reindex both weight series to match the input `station_predictors` list.
        aligned_dist_weights = dist_weights.reindex(station_predictors, fill_value=0.0)
        aligned_connectivity = connectivity_weights.reindex(station_predictors, fill_value=0.0)

        if aligned_dist_weights.sum() > 0:
            aligned_dist_weights = aligned_dist_weights / aligned_dist_weights.sum()
        else:
            aligned_dist_weights = pd.Series(0.0, index=station_predictors)

        if aligned_connectivity.sum() > 0:
            aligned_connectivity = aligned_connectivity / aligned_connectivity.sum()
        else:
            aligned_connectivity = pd.Series(0.0, index=station_predictors)

        alpha = 0.5
        blended_weights = alpha * aligned_dist_weights + (1 - alpha) * aligned_connectivity

        sum_blended_weights = blended_weights.sum()
        if sum_blended_weights > 0:
            return blended_weights / sum_blended_weights
        else:
            return pd.Series(0.0, index=station_predictors)

    def fit(self, X_incomplete):
        """
        Trains RandomForest models iteratively to impute missing values in the training data.
        This method refines the imputation models until convergence or max_iter.

        Args:
            X_incomplete (pd.DataFrame): DataFrame with missing values (NaNs) in discharge columns.
                                         Must also contain temporal feature columns if specified.
        Returns:
            self: The trained imputer object.
        """
        X = X_incomplete.copy()
        self.col_names = X.columns.tolist() # All columns (discharge + temporal features)
        self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
        self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}

        # 1. Initial Imputation: Fill all NaNs in discharge columns with column means.
        # Temporal features are assumed to be complete and are not imputed here.
        for col in self.discharge_columns:
            self.col_means[col] = X_incomplete[col].mean()
            X[col] = X[col].fillna(self.col_means[col])

        # Mask to track original missing positions (for convergence check).
        # Only consider missingness in discharge columns.
        original_missing_mask = X_incomplete[self.discharge_columns].isna()
        total_original_nans = original_missing_mask.sum().sum()
        print(f"  Total original NaNs in training data (for convergence check): {total_original_nans}")

        if total_original_nans == 0:
            print("  No original NaNs in training data, fit method effectively skips iterative imputation.")
            # Even if no NaNs, train models for all discharge columns using available data.
            for col_name in self.discharge_columns: # Iterate only over discharge columns
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns

                # Calculate weights for station predictors
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                # Combine weighted station predictors with unweighted temporal features
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    if X_predictors_combined.empty:
                        X_predictors_combined = X[temporal_predictors]
                    else:
                        X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1)
                
                if X_predictors_combined.empty or (X_predictors_combined == 0).all().all():
                     self.models[col_name] = None
                     continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_predictors_combined, X[col_name])
                self.models[col_name] = model
            return self

        print(f"  Starting iterative training (Max: {self.max_iter} iterations)...")
        for iteration in range(self.max_iter):
            print(f"  Training Iteration {iteration + 1}/{self.max_iter}")
            X_prev = X.copy() # Store previous iteration's imputed state for convergence check

            # Randomize order of discharge columns for this iteration. Temporal features are not reordered.
            discharge_cols_order = np.random.RandomState(self.random_state + iteration).permutation(self.discharge_columns)

            for col_name in discharge_cols_order: # Iterate only over discharge columns for imputation
                y_known = X_incomplete.loc[~original_missing_mask[col_name], col_name]
                if y_known.empty:
                    self.models[col_name] = None
                    continue

                # Separate predictors into station-based and temporal features
                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns

                # Calculate weights for station predictors
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                # Combine weighted station predictors with unweighted temporal features
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    if X_predictors_combined.empty: # If no station predictors, just use temporal
                        X_predictors_combined = X[temporal_predictors]
                    else: # Concatenate if both exist
                        X_predictors_combined = pd.concat([X_predictors_combined, X[temporal_predictors]], axis=1)
                
                # Filter training data: use rows where the target column was *originally known*.
                X_train_for_model = X_predictors_combined.loc[y_known.index]
                y_train_for_model = y_known.loc[X_train_for_model.index]

                if X_train_for_model.empty or y_train_for_model.empty or (X_train_for_model == 0).all().all():
                    self.models[col_name] = None
                    if X_train_for_model.empty:
                        print(f"    DEBUG: Column {col_name} - X_train_for_model is EMPTY (no training data or index mismatch).")
                    elif (X_train_for_model == 0).all().all():
                        print(f"    DEBUG: Column {col_name} - X_train_for_model is ALL ZEROS (no meaningful predictors for training).")
                    continue

                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, max_features='sqrt')
                model.fit(X_train_for_model, y_train_for_model)
                self.models[col_name] = model

                missing_in_col = original_missing_mask[col_name]
                if missing_in_col.any():
                    X_predict_for_model = X_predictors_combined.loc[missing_in_col]
                    if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                        X.loc[X_predict_for_model.index, col_name] = model.predict(X_predict_for_model)
                    else:
                        X.loc[missing_in_col, col_name] = self.col_means[col_name]
                        if X_predict_for_model.empty:
                            print(f"    DEBUG: Column {col_name} - X_predict_for_model is EMPTY (no prediction data), falling back to mean.")
                        elif (X_predict_for_model == 0).all().all():
                            print(f"    DEBUG: Column {col_name} - X_predict_for_model is ALL ZEROS (no meaningful predictors for prediction), falling back to mean.")

            # Check for convergence: only consider changes in originally missing discharge values.
            current_imputed_vals = X[self.discharge_columns][original_missing_mask].stack().values
            prev_imputed_vals = X_prev[self.discharge_columns][original_missing_mask].stack().values

            if current_imputed_vals.size == 0:
                change_norm = 0.0
            else:
                change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals)

            print(f"  Change: {change_norm:.6f}")
            if change_norm < 1e-6:
                print(f"  Converged after {iteration + 1} iterations.")
                break

        return self

    def transform(self, X_incomplete):
        """
        Imputes missing values in new data using the trained models.
        Performs iterative refinement during transformation as well.

        Args:
            X_incomplete (pd.DataFrame): DataFrame with missing values (NaNs) to be imputed.
                                         Should have the same column structure as training data,
                                         including temporal features.
        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.
        """
        X_imp = X_incomplete.copy()

        # If transform is called before fit, initialize basic attributes and fall back to mean imputation.
        if self.col_names is None: # This indicates imputer was not fitted
            self.col_names = X_incomplete.columns.tolist()
            self.discharge_columns = [col for col in self.col_names if col not in self.temporal_feature_columns]
            self.site_to_idx = {col: i for i, col in enumerate(self.col_names)}
            for col in self.discharge_columns: # Only calculate means for discharge columns
                self.col_means[col] = X_incomplete[col].mean()
            print("Warning: Transform called on an untrained imputer. Initializing with current data's means and no iterative refinement.")
            return X_imp[self.discharge_columns].fillna(self.col_means) # Only return discharge columns filled

        # Ensure X_imp has all columns the model expects (discharge + temporal features)
        # Fill NaNs only in discharge columns with stored means.
        for col in self.discharge_columns:
            X_imp[col] = X_imp[col].fillna(self.col_means[col])

        # Mask to track missing values in *this* new data for convergence check during transform.
        # Only consider missingness in discharge columns.
        missing_mask_new_data = X_incomplete[self.discharge_columns].isna()
        total_nans_in_transform = missing_mask_new_data.sum().sum()
        print(f"  Total NaNs in input data for transform: {total_nans_in_transform}")
        if total_nans_in_transform == 0:
            print("  No NaNs in input data for transform, returning original.")
            return X_imp

        print(f"  Starting iterative transformation (Max: {self.max_iter} iterations)...")
        for iteration in range(self.max_iter):
            print(f"  Transformation Iteration {iteration + 1}/{self.max_iter}")
            X_prev_imp = X_imp.copy()

            for col_name in self.discharge_columns: # Iterate only over discharge columns for imputation
                if col_name not in self.models or self.models[col_name] is None:
                    continue

                missing_in_col = missing_mask_new_data[col_name]
                if not missing_in_col.any(): continue

                station_predictors = [c for c in self.discharge_columns if c != col_name]
                temporal_predictors = self.temporal_feature_columns
                
                weights_for_stations = self._calculate_weights(col_name, station_predictors)
                
                X_predictors_combined = pd.DataFrame()
                if not weights_for_stations.empty:
                    X_predictors_combined = X_imp[station_predictors].multiply(weights_for_stations, axis=1)
                if temporal_predictors:
                    if X_predictors_combined.empty:
                        X_predictors_combined = X_imp[temporal_predictors]
                    else:
                        X_predictors_combined = pd.concat([X_predictors_combined, X_imp[temporal_predictors]], axis=1)

                X_predict_for_model = X_predictors_combined.loc[missing_in_col]

                if not X_predict_for_model.empty and not (X_predict_for_model == 0).all().all():
                    X_imp.loc[X_predict_for_model.index, col_name] = self.models[col_name].predict(X_predict_for_model)
                else:
                    X_imp.loc[missing_in_col, col_name] = self.col_means[col_name]
                    if X_predict_for_model.empty:
                        print(f"    DEBUG: Column {col_name} (Transform) - X_predict_for_model is EMPTY, falling back to mean.")
                    elif (X_predict_for_model == 0).all().all():
                        print(f"    DEBUG: Column {col_name} (Transform) - X_predict_for_model is ALL ZEROS, falling back to mean.")

            # Check for convergence during transformation.
            current_imputed_vals = X_imp[self.discharge_columns][missing_mask_new_data].stack().values
            prev_imputed_vals = X_prev_imp[self.discharge_columns][missing_mask_new_data].stack().values

            if current_imputed_vals.size == 0:
                change_norm = 0.0
            else:
                change_norm = np.linalg.norm(current_imputed_vals - prev_imputed_vals)

            print(f"  Change: {change_norm:.6f}")
            if change_norm < 1e-6:
                print(f"  Converged after {iteration + 1} iterations.")
                break

        return X_imp
