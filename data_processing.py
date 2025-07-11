import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, pi
import re
from geopy.distance import geodesic
import io # For handling file buffers

def clean_station_name(name):
    """
    Standardizes station names: converts to lowercase, strips whitespace,
    and removes/standardizes various apostrophe/quote characters.
    This ensures consistent naming across all datasets.
    """
    name = str(name).strip().lower()
    name = name.replace("’", "").replace("'", "").replace("\"", "").replace("`", "")
    return name

def parse_lat_lon(coord_str, is_latitude=True):
    """
    Parses a latitude or longitude string into a decimal degree float.
    Handles various formats including decimal, DMS, and ambiguous separators.
    Automatically adjusts sign based on direction (S/W for negative) or original degree sign.
    Includes explicit range validation.

    Args:
        coord_str (str or float): The coordinate string (e.g., "21o 50’ 02’’", "21.45 N").
        is_latitude (bool): True if parsing latitude (checks [-90, 90]), False for longitude (checks [-180, 180]).

    Returns:
        float: The coordinate in decimal degrees.

    Raises:
        ValueError: If parsing fails or the value is out of bounds.
    """
    if isinstance(coord_str, (int, float)):
        val = float(coord_str)
        if is_latitude and not -90 <= val <= 90:
            raise ValueError(f"Latitude {val} is out of bounds (-90 to 90): '{coord_str}'")
        if not is_latitude and not -180 <= val <= 180:
            raise ValueError(f"Longitude {val} out of bounds (-180 to 180): '{coord_str}'")
        return val

    s = str(coord_str).strip()
    s_spaced = re.sub(r"[o°\'\"’‘’]", " ", s)
    numbers_str = re.findall(r'(-?\d+\.?\d*)', s_spaced)
    numbers = [float(n) for n in numbers_str]

    if not numbers:
        raise ValueError(f"No numerical parts found in coordinate string: '{coord_str}'")

    deg = numbers[0]
    min_ = 0.0
    sec = 0.0

    if len(numbers) >= 2:
        min_ = numbers[1]
    if len(numbers) >= 3:
        sec = numbers[2]

    decimal_deg_abs = abs(deg) + min_/60 + sec/3600
    final_decimal_deg = decimal_deg_abs
    direction_match = re.search(r'([NSEWnsew])', s)
    direction_char = direction_match.group(1).upper() if direction_match else None
    
    if direction_char:
        if (is_latitude and direction_char == 'S') or (not is_latitude and direction_char == 'W'):
            final_decimal_deg = -decimal_deg_abs
    elif deg < 0:
        final_decimal_deg = -decimal_deg_abs

    if is_latitude and not -90 <= final_decimal_deg <= 90:
        raise ValueError(f"Parsed Latitude {final_decimal_deg} out of bounds (-90 to 90): '{coord_str}'")
    if not is_latitude and not -180 <= final_decimal_deg <= 180:
        raise ValueError(f"Parsed Longitude {final_decimal_deg} out of bounds (-180 to 180): '{coord_str}'")

    return final_decimal_deg

@st.cache_data
def _load_and_preprocess_data_for_streamlit(discharge_file_buffer, connectivity_file_buffer, coords_file_buffer):
    """
    Loads discharge, connectivity, and coordinate data from file buffers provided by Streamlit.
    Standardizes station names, filters to common stations, and prepares dataframes.
    Returns: df_discharge, df_connectivity_filtered, df_coords, vcode_to_station_name, station_name_to_vcode
    """
    st.info("Processing discharge data...")
    df_discharge_raw_io = pd.read_csv(io.BytesIO(discharge_file_buffer.getvalue()), parse_dates=["Date"], dayfirst=True)
    df_discharge = df_discharge_raw_io.set_index("Date").select_dtypes(include=[np.number])
    df_discharge.columns = [clean_station_name(col) for col in df_discharge.columns] # Clean discharge columns

    vcode_to_station_name = {}
    station_name_to_vcode = {}
    df_connectivity_raw = pd.DataFrame() # Initialize as empty

    if connectivity_file_buffer and connectivity_file_buffer.getvalue(): # Check if buffer is not empty
        try:
            st.info("Processing connectivity data (original V-code format, as in utils.py)...")
            # Load connectivity data: V-code as index, 'Name of site' as a column
            # This matches the logic from utils.py's load_and_preprocess_data
            df_connectivity_raw = pd.read_csv(io.BytesIO(connectivity_file_buffer.getvalue()), header=0, index_col=0)
            
            # Clean index (V-codes) and column names (V-codes for connectivity)
            df_connectivity_raw.index = [clean_station_name(idx) for idx in df_connectivity_raw.index]
            df_connectivity_raw.columns = [clean_station_name(col) for col in df_connectivity_raw.columns]

            # Populate vcode_to_station_name and station_name_to_vcode mapping
            # Ensure 'Name of site' column exists and clean its values
            if 'name of site' in df_connectivity_raw.columns: # Use 'name of site' as per utils.py
                vcode_to_station_name = {vcode: clean_station_name(name)
                                         for vcode, name in df_connectivity_raw['name of site'].items()}
                station_name_to_vcode = {name: vcode for vcode, name in vcode_to_station_name.items()}
            else:
                st.warning("Connectivity file missing 'name of site' column. V-code to station name mapping will be incomplete.")

            # Drop lat/lon and 'name of site' if they are columns in the connectivity file, as they are handled by df_coords
            # Use 'latitude (n)' and 'longitude (e)' as per utils.py
            cols_to_drop = [col for col in ['latitude (n)', 'longitude (e)', 'name of site'] if col in df_connectivity_raw.columns]
            df_connectivity_raw = df_connectivity_raw.drop(columns=cols_to_drop, errors='ignore')
            st.success("Connectivity data processed.")
        except Exception as e:
            st.warning(f"Could not load connectivity data in original V-code format: {e}. Proceeding without connectivity information.")
            df_connectivity_raw = pd.DataFrame()
    else:
        st.info("No connectivity data provided or it was empty. Proceeding without connectivity information.")


    st.info("Processing latitude/longitude data...")
    df_coords_raw_io = pd.read_csv(io.BytesIO(coords_file_buffer.getvalue()), header=0)
    df_coords = df_coords_raw_io.copy()
    df_coords["Name_cleaned"] = [clean_station_name(name) for name in df_coords["Name of site"]]
    df_coords = df_coords.set_index("Name_cleaned")
    st.success("Latitude/Longitude data processed.")

    common_names = set(df_discharge.columns) & set(df_coords.index)
    
    # If connectivity data is present, filter common_names by stations that have a V-code mapping
    if not df_connectivity_raw.empty and station_name_to_vcode:
        # Get the set of station names that have corresponding V-codes in the connectivity file
        connectivity_station_names_from_vcodes = set(station_name_to_vcode.keys())
        common_names = common_names & connectivity_station_names_from_vcodes
    
    st.info(f"Found {len(common_names)} common stations across all available files.")

    if not common_names:
        raise ValueError("No common stations found across all input files! Please check data files and names.")

    df_discharge = df_discharge[list(common_names)]
    df_coords = df_coords.loc[list(common_names)]

    df_connectivity_filtered = pd.DataFrame()
    if not df_connectivity_raw.empty and station_name_to_vcode:
        # Filter df_connectivity_raw to only include V-codes that map to common station names
        common_vcodes = [station_name_to_vcode[name] for name in common_names if name in station_name_to_vcode]
        # Ensure V-codes are in the index and columns of df_connectivity_raw
        valid_vcodes_in_raw = [vcode for vcode in common_vcodes if vcode in df_connectivity_raw.index and vcode in df_connectivity_raw.columns]
        
        if valid_vcodes_in_raw:
            df_connectivity_filtered = df_connectivity_raw.loc[valid_vcodes_in_raw, valid_vcodes_in_raw]
        else:
            st.warning("No valid V-codes found in connectivity data after filtering for common stations. Connectivity matrix will be empty.")
            df_connectivity_filtered = pd.DataFrame()
    else:
        st.info("df_connectivity_raw is empty or station_name_to_vcode is empty, df_connectivity_filtered will also be empty.")


    assert set(df_discharge.columns) == set(df_coords.index), \
        "Internal Error: Station name mismatch between discharge and coords after filtering."
    
    # Assertions for connectivity now check against V-codes if df_connectivity_filtered is not empty
    if not df_connectivity_filtered.empty:
        # Convert common_names to V-codes for comparison
        common_vcodes_from_names = set(station_name_to_vcode[name] for name in common_names if name in station_name_to_vcode)
        assert set(df_connectivity_filtered.index) == common_vcodes_from_names, \
            "Internal Error: V-code mismatch between filtered connectivity index and common V-codes."
        assert set(df_connectivity_filtered.columns) == common_vcodes_from_names, \
            "Internal Error: V-code mismatch between filtered connectivity columns and common V-codes."

    return df_discharge, df_connectivity_filtered, df_coords, vcode_to_station_name, station_name_to_vcode

def add_temporal_features(df):
    """
    Adds cyclical temporal features (day of year sine/cosine) to the DataFrame.
    These features help the model capture seasonality.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a DateTimeIndex.
    
    Returns:
        pd.DataFrame: The DataFrame with new temporal features.
    """
    from math import pi
    df_copy = df.copy()
    
    day_of_year = df_copy.index.dayofyear
    
    df_copy['day_of_year_sin'] = np.sin(2 * pi * day_of_year / 365.25)
    df_copy['day_of_year_cos'] = np.cos(2 * pi * day_of_year / 365.25)
    
    st.info("Added 'day_of_year_sin' and 'day_of_year_cos' temporal features.")
    return df_copy

def build_distance_matrix(df_coords, sites_lower):
    """
    Constructs a DataFrame of geodesic distances (in km) between all pairs of stations.
    Uses parsed latitude/longitude from df_coords.
    """
    from geopy.distance import geodesic
    coords_parsed = {}
    for name_lower, row in df_coords.iterrows():
        try:
            lat = parse_lat_lon(row['Latitude (N)'], is_latitude=True)
            lon = parse_lat_lon(row['Longitude (E)'], is_latitude=False)
            coords_parsed[name_lower] = (lat, lon)
        except ValueError as e:
            st.warning(f"Warning: Failed to parse coordinates for '{name_lower}': {e}. Skipping this station in distance matrix construction.")
            continue

    distance_matrix = pd.DataFrame(index=sites_lower, columns=sites_lower, dtype=float)

    available_sites = [s for s in sites_lower if s in coords_parsed]
    distance_matrix = distance_matrix.loc[available_sites, available_sites]

    for s1_name in available_sites:
        for s2_name in available_sites:
            if s1_name == s2_name:
                distance_matrix.loc[s1_name, s2_name] = 0.0
            else:
                distance_matrix.loc[s1_name, s2_name] = geodesic(coords_parsed[s1_name], coords_parsed[s2_name]).km
    return distance_matrix

def build_connectivity_matrix(df_connectivity_filtered, sites_lower, vcode_to_station_name):
    """
    Constructs a DataFrame of connectivity, ensuring it aligns with station names.
    Converts V-code based df_connectivity_filtered to station name based for consistency.
    """
    if df_connectivity_filtered.empty or not vcode_to_station_name: 
        st.info("No valid connectivity data or V-code mapping available to build connectivity matrix. Returning an empty matrix.")
        return pd.DataFrame(0, index=sites_lower, columns=sites_lower, dtype=int)

    # Create a new, empty DataFrame indexed and columned by station names
    connectivity_matrix_by_name = pd.DataFrame(0, index=sites_lower, columns=sites_lower, dtype=int)
    
    # Populate the new matrix by mapping V-codes to station names
    # Iterate through the V-code based filtered connectivity DataFrame
    for vcode_row in df_connectivity_filtered.index:
        station_name_row = vcode_to_station_name.get(vcode_row)
        if station_name_row and station_name_row in sites_lower: # Ensure name exists and is in common sites
            for vcode_col in df_connectivity_filtered.columns:
                station_name_col = vcode_to_station_name.get(vcode_col)
                if station_name_col and station_name_col in sites_lower: # Ensure name exists and is in common sites
                    connectivity_matrix_by_name.loc[station_name_row, station_name_col] = df_connectivity_filtered.loc[vcode_row, vcode_col]
            
    return connectivity_matrix_by_name
