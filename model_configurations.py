# model_configurations.py
import pandas as pd
import numpy as np
from missforest_imputer import ModifiedMissForest

def train_full_model(df_train_masked, distance_matrix, connectivity_matrix, temporal_features):
    """
    Trains the full ModifiedMissForest model with inverse distance weighting,
    hydrological connectivity, and temporal features.
    """
    print("\n--- Training Full Modified MissForest Model (Inverse Distance Weighting, Connectivity, Temporal) ---")
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features
    )
    mf_imputer.fit(df_train_masked)
    print("Full Modified MissForest model trained successfully.")
    return mf_imputer

def train_no_temporal_model(df_train_masked, distance_matrix, connectivity_matrix, temporal_features):
    """
    Trains ModifiedMissForest without temporal features.
    Retains inverse distance weighting and hydrological connectivity.
    """
    # Create a training dataframe without temporal features if they exist
    df_train_masked_no_temporal = df_train_masked.drop(columns=temporal_features, errors='ignore')

    print("\n--- Training Modified MissForest Model (No Temporal Features) ---")
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_matrix,
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=[] # Explicitly no temporal features
    )
    # Fit with the version of data that does not contain temporal features
    mf_imputer.fit(df_train_masked_no_temporal)
    print("Modified MissForest (No Temporal Features) model trained successfully.")
    return mf_imputer


def train_no_contributor_model(df_train_masked, distance_matrix, connectivity_matrix, temporal_features):
    """
    Trains ModifiedMissForest without hydrological contributor info (connectivity).
    Retains inverse distance weighting and temporal features.
    """
    # Create a connectivity matrix of zeros to effectively remove its influence
    connectivity_zero = pd.DataFrame(0, index=connectivity_matrix.index,
                                     columns=connectivity_matrix.columns, dtype=int)

    print("\n--- Training Modified MissForest Model (No Contributor Info) ---")
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_matrix,
        connectivity=connectivity_zero, # No connectivity
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse',
        temporal_feature_columns=temporal_features # Still uses temporal features
    )
    mf_imputer.fit(df_train_masked)
    print("Modified MissForest (No Contributor Info) model trained successfully.")
    return mf_imputer

def train_no_spatial_temporal_model(df_train_masked, distance_matrix, connectivity_matrix, temporal_features):
    """
    Trains a baseline MissForest model with no spatial (distance or connectivity)
    or temporal features. This is essentially a standard MissForest.
    """
    # Create a uniform distance matrix (effectively no spatial weighting influence after normalization)
    distance_uniform = pd.DataFrame(1.0, index=distance_matrix.index,
                                    columns=distance_matrix.columns, dtype=float)
    np.fill_diagonal(distance_uniform.values, 0.0) # Distance to self is 0

    # Create a connectivity matrix of zeros
    connectivity_zero = pd.DataFrame(0, index=connectivity_matrix.index,
                                     columns=connectivity_matrix.columns, dtype=int)

    # Create a training dataframe without temporal features
    df_train_masked_no_temporal = df_train_masked.drop(columns=temporal_features, errors='ignore')

    print("\n--- Training Baseline MissForest Model (No Spatial/Temporal Info) ---")
    mf_imputer = ModifiedMissForest(
        distance_matrix=distance_uniform, # Uniform spatial weighting
        connectivity=connectivity_zero, # No connectivity
        max_iter=10,
        n_estimators=100,
        random_state=42,
        distance_weighting_type='inverse', # Type doesn't matter much with uniform dist
        temporal_feature_columns=[] # No temporal features
    )
    mf_imputer.fit(df_train_masked_no_temporal)
    print("Baseline MissForest (No Spatial/Temporal Info) model trained successfully.")
    return mf_imputer

