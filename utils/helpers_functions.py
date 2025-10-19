import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def z_score_normalize(data, global_mean, global_Std):
    """
    Normalize the input data using z-score normalization.

    Parameters:
    data (array-like): Input data to be normalized.
    global_mean (float): Mean of the data distribution.
    global_Std (float): Standard deviation of the data distribution.

    Returns:
    array-like: Z-score normalized data.
    """
    return (data - global_mean) / global_Std

def reverse_z_score(normalized_data, global_mean, global_Std):
    """
    Reverse the z-score normalization to retrieve the original data.

    Parameters:
    normalized_data (array-like): Z-score normalized data.
    global_mean (float): Mean of the original data distribution.
    global_Std (float): Standard deviation of the original data distribution.

    Returns:
    array-like: Original data before normalization.
    """
    return (normalized_data * global_Std) + global_mean

def denormalize_mae(mae_z_normalized, global_Std):
    """
    Convert z-score normalized MAE values back to degrees.

    Parameters:
    mae_z_normalized (array-like): The MAE values in z-score normalized form.
    global_Std (float): The standard deviation used for z-score normalization.

    Returns:
    array-like: The MAE values in degrees.
    """
    return mae_z_normalized * global_Std  # Reverse normalization (mean not needed for MAE)

def interpolation(input_image, target_sizeor):
    """
    Resize an image using nearest-neighbor interpolation.

    Parameters:
    input_image (ndarray): Input image to be resized.
    target_size (tuple): Desired output size as (height, width).

    Returns:
    ndarray: Resized image using nearest-neighbor interpolation.
    """
    
    upsampled_image = cv2.resize(input_image, (target_sizeor[1], target_sizeor[0]), interpolation=cv2.INTER_CUBIC)
    
    return upsampled_image

def reshape(data):
    """
    Expand the dimensions of the input data by adding a new axis at the last position.

    Parameters:
    data (array-like): The input data to be reshaped.

    Returns:
    array-like: The reshaped data with an additional dimension.
    """
    data = np.expand_dims(data, axis=-1)
    return data

def split_and_reshape(lr_data, hr_data):
    """
    Split the input data into training and validation sets, then reshape them.

    Parameters:
    lr_data (array-like): Low-resolution input data.
    hr_data (array-like): High-resolution target data.

    Returns:
    tuple: Reshaped training and validation data (x_train, x_val, y_train, y_val).
    """
    x_train, x_val, y_train, y_val = train_test_split(lr_data, hr_data, test_size=0.2, random_state=42)

    return reshape(x_train), reshape(x_val), reshape(y_train), reshape(y_val)

def input_preprocessing(file_path, global_mean, global_Std, target_size=(0, 0)):
    """
    Load and preprocess input data by normalizing it.

    Parameters:
    file_path (str): Path to the file containing input data.
    global_mean (float): Mean of the data distribution.
    global_Std (float): Standard deviation of the data distribution.
    target_size (tuple): The target size for resizing the images (height, width).

    Returns:
    array-like: The preprocessed input data.
    """
    lr_data = np.load(file_path)
    lr_data = z_score_normalize(lr_data, global_mean, global_Std)

    return lr_data

def is_leap_year(year):
    """
    Determines whether a given year is a leap year.

    Parameters:
    - year (int): The year to check.

    Returns:
    - bool: True if it's a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_season_data(data, year, season):
    """
    Extracts seasonal data from a given dataset based on the year and season.

    Parameters:
    data (numpy.ndarray): The dataset containing time-series data for a full year.
    year (int): The year for which the seasonal data is required.
    season (str): The season to extract ('spring', 'summer', 'autumn', 'winter' or 'all').

    Returns:
    numpy.ndarray: A subset of the data corresponding to the specified season.
                   For 'winter', the function includes data from the beginning of the year 
                   to handle the transition between years.

    Notes:
    - The function accounts for leap years, adjusting the indices accordingly.
    - Uses predefined indices for seasonal segmentation.
    - Assumes that `is_leap_year(year)` is a valid function that determines leap years.
    """
    
    season_indices = {
        'spring': (236, 604),
        'summer': (604, 972),
        'autumn': (972, 1336),
        'winter': (1336, 1460)
    }

    leap_year_season_indices = {
        'spring': (240, 608),
        'summer': (608, 976),
        'autumn': (976, 1340),
        'winter': (1340, 1464)
    }
    if season == 'all':
        return data
    else:
        if is_leap_year(year):
            start_idx, end_idx = leap_year_season_indices[season]
            if season == 'winter':
                return np.concatenate([data[0:240], data[start_idx:end_idx]])
            else:
                return data[start_idx:end_idx]
        else:
            start_idx, end_idx = season_indices[season]
            if season == 'winter':
                return np.concatenate([data[0:236], data[start_idx:end_idx]])
            else:
                return data[start_idx:end_idx]


def split_data(dates, low_res_data, high_res_data):
    """
    Splits the dataset into training, validation, and test sets while maintaining reproducibility.

    Parameters:
    dates (numpy.ndarray): Array of date values corresponding to the dataset.
    low_res_data (numpy.ndarray): Lower resolution dataset (features).
    high_res_data (numpy.ndarray): Higher resolution dataset (labels/targets).

    Returns:
    tuple: A tuple containing the following:
        - X_train (numpy.ndarray): Training set (low resolution).
        - X_val (numpy.ndarray): Validation set (low resolution).
        - X_test (numpy.ndarray): Test set (low resolution).
        - y_train (numpy.ndarray): Training set (high resolution).
        - y_val (numpy.ndarray): Validation set (high resolution).
        - y_test (numpy.ndarray): Test set (high resolution).
        - dates_train (numpy.ndarray): Training set dates.
        - dates_val (numpy.ndarray): Validation set dates.
        - dates_test (numpy.ndarray): Test set dates.

    Notes:
    - The function shuffles the data using a fixed random seed (42) for reproducibility.
    - The data is split into 70% training, 15% validation, and 15% testing.
    - Shuffling ensures a random distribution of data across the splits.
    """
    
    # Set a random seed for reproducibility
    random_seed = 42

    # Shuffle the data using the seed (this ensures the same shuffle every time)
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(dates))

    # Extract the shuffled arrays
    shuffled_dates = dates[shuffled_indices]
    shuffled_low_res = low_res_data[shuffled_indices]
    shuffled_high_res = high_res_data[shuffled_indices]

    # Now split the shuffled data into train, validation, and test sets:
    # 70% for training, 15% for validation, 15% for testing
    train_size = int(0.7 * len(dates))
    val_size = int(0.15 * len(dates))

    # Define the splits
    X_train = shuffled_low_res[:train_size]
    y_train = shuffled_high_res[:train_size]
    dates_train = shuffled_dates[:train_size]

    X_val = shuffled_low_res[train_size:train_size + val_size]
    y_val = shuffled_high_res[train_size:train_size + val_size]
    dates_val = shuffled_dates[train_size:train_size + val_size]

    X_test = shuffled_low_res[train_size + val_size:]
    y_test = shuffled_high_res[train_size + val_size:]
    dates_test = shuffled_dates[train_size + val_size:]


    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test

def read_dataset(dates_path, lr_data_path, hr_data_path, season, year_start, year_end, global_mean, global_Std, target_size=(0,0)):
    """
    Reads and processes seasonal climate data over a specified range of years.
    
    Parameters:
    dates_path(str): Path to the folder that contains the files with the dates.
    lr_data_path(str): Path to the folder with the low resolution data files.
    hr_data_path(str): Path to the folder with the high resolution data files.
    season (str): The season to extract ('spring', 'summer', 'autumn', or 'winter').
    year_start (int): The starting year of the dataset.
    year_end (int): The ending year of the dataset.
    global_mean (float): The global mean value used for normalization.
    global_Std (float): The global standard deviation used for normalization.
    target_size (tuple, optional): The target size for resizing low-resolution data.
    
    Returns:
    tuple: A tuple containing the following datasets split into training, validation, and test sets:
        - x_train (numpy.ndarray): Training set (low-resolution data).
        - x_val (numpy.ndarray): Validation set (low-resolution data).
        - x_test (numpy.ndarray): Test set (low-resolution data).
        - y_train (numpy.ndarray): Training set (high-resolution data).
        - y_val (numpy.ndarray): Validation set (high-resolution data).
        - y_test (numpy.ndarray): Test set (high-resolution data).
        - dates_train (numpy.ndarray): Training set dates.
        - dates_val (numpy.ndarray): Validation set dates.
        - dates_test (numpy.ndarray): Test set dates.
    
    Notes:
    - Loads climate data for the specified years and extracts the requested season.
    - Applies preprocessing and normalization to both low-resolution and high-resolution datasets.
    - Uses `split_data` to divide the data into training (70%), validation (15%), and test (15%) sets.
    - Assumes the existence of helper functions such as `get_season_data`, `input_preprocessing`, and `z_score_normalize`.
    - Data is loaded from file paths constructed using `project_path`, `dates_path`, `lr_data_path`, and `hr_data_path`.
    """
    dates, low_res_data, high_res_data = [], [], []
    
    years = list(range(year_start, year_end+1))
    for year in years:
        dates_data = np.load(os.path.join(dates_path, str(year)+'_t2m_sfc_idx.npy'), allow_pickle=True)
        season_dates_data = get_season_data(dates_data, year, season)
        dates.append(season_dates_data)
        
        lr_data = input_preprocessing(os.path.join(lr_data_path, str(year)+'_t2m_sfc.npy'), global_mean, global_Std,target_size)
        season_lr_data = get_season_data(lr_data, year, season)
        low_res_data.append(season_lr_data)
        
        hr_data = np.load(os.path.join(hr_data_path, str(year)+'_t2m_sfc.npy'))
        hr_data = z_score_normalize(hr_data, global_mean, global_Std)
        season_hr_data = get_season_data(hr_data, year, season)
        high_res_data.append(season_hr_data)
    
    dates = np.concatenate(dates, axis=0)
    low_res_data = np.concatenate(low_res_data, axis=0)
    high_res_data = np.concatenate(high_res_data, axis=0)
    
    x_train, x_val, x_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = split_data(dates, low_res_data, high_res_data)
    
    return reshape(x_train), reshape(x_val), reshape(x_test), reshape(y_train), reshape(y_val), reshape(y_test), dates_train, dates_val, dates_test

def mean_with_land_mask(data, deg):
    """
    Computes the mean of data values based on a land mask.

    This function loads a land-sea mask from a NumPy file, applies a threshold 
    (values > 0.5 are considered land), and calculates the mean of the `data` 
    elements corresponding to land areas.

    Parameters:
    data (numpy.ndarray): 2D array representing data values (e.g., temperature, MAE).
    deg (str): String indicating the resolution or identifier for selecting the mask file.

    Returns:
    float: The mean of `data` values over land areas.
    """

    # Load the land-sea mask
    mask = np.load(os.path.join('/content/drive/My Drive/MSc_Thesis/input_data/mask/', 
                                'mask_' + deg + '.npy'), allow_pickle=True)

    # Create a boolean mask (True for land, False for sea)
    bool_mask = mask > 0.5

    # Compute mean only for land areas
    masked_mean = np.mean(data[bool_mask])

    return masked_mean

def mean_with_coastline_mask(data, deg):
    """
    Computes the mean of data values based on a land mask.

    This function loads a land-sea mask from a NumPy file, applies a threshold 
    (values > 0.5 are considered land), and calculates the mean of the `data` 
    elements corresponding to land areas.

    Parameters:
    data (numpy.ndarray): 2D array representing data values (e.g., temperature, MAE).
    deg (str): String indicating the resolution or identifier for selecting the mask file.

    Returns:
    float: The mean of `data` values over land areas.
    """

    # Load the land-sea mask
    mask = np.load(os.path.join('/content/drive/My Drive/MSc_Thesis/input_data/mask/', 
                                'mask_' + deg + '.npy'), allow_pickle=True)

    bool_mask = np.logical_and(mask > 0.3, mask < 0.7)

    # Compute mean only for land areas
    masked_mean = np.mean(data[bool_mask])

    return masked_mean

def compute_slope(dem, resolution_deg, lat_start=80, lat_end=0, lon_start=-60, lon_end=85):

    # Earth's radius approximation (in meters)
    R = 6371000

    nrows, ncols = dem.shape
    lats = np.linspace(lat_start, lat_end, nrows)

    # Approximate dy = meridional distance between rows
    dy = resolution_deg * (np.pi / 180) * R  

    # Precompute dx for each latitude row
    dx = resolution_deg * (np.pi / 180) * R * np.cos(np.deg2rad(lats))

    # Compute gradient in rows (y direction) and cols (x direction)
    dzdy = np.gradient(dem, axis=0) / dy
    dzdx = np.zeros_like(dem)

    for i in range(nrows):
        dzdx[i, :] = np.gradient(dem[i, :], edge_order=2) / dx[i]

    # Compute slope in radians
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))

    return slope

def compute_aspect(dem, cell_size=1.0):
    dzdx = np.gradient(dem, axis=1) / cell_size
    dzdy = np.gradient(dem, axis=0) / cell_size

    # Aspect in radians (0 = East, pi/2 = North)
    aspect = np.arctan2(-dzdy, dzdx)
    aspect = np.where(aspect < 0, 2 * np.pi + aspect, aspect)

    return aspect