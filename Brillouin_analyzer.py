import os
import re
import signal
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import (
    find_peaks,
    peak_widths,
    peak_prominences,
    savgol_filter
)
from scipy.optimize import least_squares
from scipy.ndimage import median_filter, gaussian_filter
from scipy.interpolate import griddata
from itertools import combinations, product
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize

from tqdm.notebook import tqdm
from tqdm.contrib import itertools

warnings.filterwarnings("ignore")

def plot_raw_spectra(array, x, y, z):
    plt.figure(figsize=(12, 7))
    plt.plot(array[x, y, z])
    plt.title(f"Raw Brillouin Spectrum at (x={x}, y={y}, z={z})")
    plt.xlabel("Index")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.show()
    
def parse_brillouin_directory(directory, label):
    # Initialize a dictionary to store the data temporarily
    data_dict = {}

    # List all files in the directory
    files = os.listdir(directory)

    # Loop through each file
    for file in files:
        # Check if the file starts with the given label and ends with '.asc'
        if file.startswith(label+"B") and file.endswith('.asc'):
            # Example filename format: "label_B_Z1Y2X3.asc"
            parts = file.split('_')
            
            # Extract the coordinates from the filename (e.g., Z1Y2X3)
            coords = parts[-1].replace('.asc', '')  # Remove '.asc'
            z = int(coords.split('Z')[1].split('Y')[0])  # Extract Z coordinate
            y = int(coords.split('Y')[1].split('X')[0])  # Extract Y coordinate
            x = int(coords.split('X')[1])  # Extract X coordinate

            # Load the data from the file (assuming two columns)
            file_path = os.path.join(directory, file)
            data = np.loadtxt(file_path)

            # Assume the second column of the .asc file contains the spectral data
            # Store this data in the dictionary with (x, y, z) as the key
            data_dict[(x, y, z)] = data[:, 1]  # Spectral data is the second column

    # Determine the size of the 3D grid (max x, y, z)
    max_x = max([key[0] for key in data_dict.keys()]) + 1
    max_y = max([key[1] for key in data_dict.keys()]) + 1
    max_z = max([key[2] for key in data_dict.keys()]) + 1

    # Initialize an empty 3D numpy array with the shape (max_x, max_y, max_z)
    brillouin_spectra = np.zeros((max_x, max_y, max_z), dtype=object)  # Using object to store arrays of variable lengths

    # Populate the 3D numpy array with the data from the dictionary
    for (x, y, z), spectrum in data_dict.items():
        brillouin_spectra[x][y][z] = spectrum

    return brillouin_spectra

def analyze_brillouin(
    brillouin_spectra, x, y, z, FSR=29.95, shift_range=(7.2, 7.4),
    shift_tolerance = 0.3, # How different the shifts can be from the median shift
    prominence_range=(0, 100), prominence_step=20,
    height_range=(600, 1100), height_step=10,
    height_tolerance=4,  # How different the brilluen peak heights can be from the laser peak height
    symmetry_tolerance = 0.2,  # Tolerance for distance symmetry for Brillouin peaks before fitting
    FSR_tolerance = 0.2,  # Tolerance for the Free Spectral Range (FSR) in GHz
    max_exclusions=3,  # Maximum number of peaks to exclude  # 20% tolerance
    min_laser_peaks = 3,
    min_peak_distance = 25,  # Minimum distance between peaks
    make_plot=True
):
    """
    This function extracts the Brillouin spectrum at the given (x, y, z) coordinates, rescales the X-axis
    using a quadratic fit (y = ax^2 + bx + c) with constraints to ensure monotonicity,
    identifies laser and Brillouin peaks, calculates the Brillouin shift and FWHM,
    optimizes the prominence and height parameters, ensures peak height consistency and distance symmetry,
    and plots the spectrum with annotations showing distances between neighboring laser peaks.

    Parameters:
    brillouin_spectra (np.ndarray): 3D array containing the Brillouin spectra.
    x (int): The x-coordinate.
    y (int): The y-coordinate.
    z (int): The z-coordinate.
    prominence_range (tuple): Range of prominence values (min, max) for peak detection.
    prominence_step (float): Step size for prominence.
    height_range (tuple): Range of height values (min, max) for peak detection.
    height_step (float): Step size for height.
    FSR (float): Free Spectral Range (Hz). Default is 29.95 Hz.
    max_exclusions (int): Maximum number of peaks to exclude in the iterative exclusion process.
    distance_symmetry_tolerance (float): Tolerance for distance symmetry (e.g., 0.2 for 20%).
    make_plot (bool): Whether or not to plot the spectrum and highlight peaks.

    Returns:
    median_shift (float): Best median Brillouin shift found.
    median_fwhm (float): Best median FWHM of Brillouin peaks found.
    """

    # Extract the spectrum from the array at the given coordinates (x, y, z)
    try:
        spectrum = brillouin_spectra[x][y][z]
    except IndexError:
        print(f"Coordinates ({x}, {y}, {z}) are out of bounds.")
        return None, None

    if spectrum is None or len(spectrum) == 0:
        print(f"No spectrum found at coordinates ({x}, {y}, {z})")
        return None, None
    
    
    # Substitute negative values with zeros
    spectrum[spectrum < 0] = 0
    
    laser_height_tolerance = 4
    
    # Cut off beginning
    # spectrum = spectrum[400:]
    spectrum=spectrum[800:]
    # Remove baseline
    spectrum = spectrum - np.min(spectrum)

    # apply savgol 
    # spectrum = savgol_filter(spectrum, 25, 3)
    # # Set first 400 values to zero
    # spectrum[:400] = 0
    # Create the initial X-axis (assuming arbitrary units)
    x_axis = np.arange(len(spectrum))

    # Define ranges for prominence and height
    prominences_range = np.arange(prominence_range[0], prominence_range[1] + prominence_step, prominence_step)
    heights = np.arange(height_range[0], height_range[1] + height_step, height_step)
    
    # Reverse the prominence from high to low
    # prominences_range = np.flip(prominences_range)
    
    # Reverse the height from high to low
    # heights = np.flip(heights)

    # Initialize variables to store the best results
    best_median_shift = None
    best_median_fwhm = None
    best_a = None
    best_b = None  # Quadratic and linear coefficients
    best_c = None
    best_laser_peaks_indices = None
    best_brillouin_peaks_indices = None
    best_shifts = []
    best_fwhms = []
    best_rescaled_x_axis = None
    best_excluded_peaks = ()
    best_fwhms_right_list = []
    best_fwhms_left_list = []
    # Define expected Brillouin shift range
    expected_shift_min = shift_range[0]
    expected_shift_max = shift_range[1]

    
    all_broad_peaks, all_properties = find_peaks(spectrum, prominence=prominence_range[0], height=height_range[0], distance=min_peak_distance)
    all_prominences = peak_prominences(spectrum, all_broad_peaks)[0]

    previous_guess=[1e-16, 1e-16, 0]
    
    # Iterate over exclusion levels first
    for exclusion_level in range(0, max_exclusions + 1):
        if exclusion_level == 0:
            excluded_combinations = [()]
        else:
            # Find all possible combinations of peaks to exclude at this exclusion level
            # First, we need to find all peaks with any prominence and height (broadest search)

            
            # Calculate the prominences for the broad peaks

            
            # # Find all combinations of peaks to exclude at this exclusion level
            # excluded_combinations = list(combinations(all_broad_peaks, exclusion_level))

            # # Sort the combinations based on the sum of the prominences of excluded peaks
            # excluded_combinations = sorted(
            #     excluded_combinations,
            #     key=lambda combo: sum([all_prominences[np.where(all_broad_peaks == peak)[0][0]] for peak in combo]))
            
            
            # # Sort peaks by their prominence (least prominent first)
            sorted_peaks_by_prominence = [x for _, x in sorted(zip(all_prominences, all_broad_peaks))]

            # Find all combinations of peaks to exclude at this exclusion level
            excluded_combinations = list(combinations(sorted_peaks_by_prominence, exclusion_level))

            # Sort the combinations based on the prominence of the first excluded peak in each combination
            excluded_combinations = sorted(
                excluded_combinations,
                key=lambda combo: all_prominences[np.where(all_broad_peaks == combo[0])[0][0]] if combo else 0
            )

        


        # Iterate over all combinations of excluded peaks
        for excluded_peaks in excluded_combinations:
            # Determine which peaks to include by excluding the specified peaks
            if exclusion_level > 0:
                exclusion_set = set(excluded_peaks)
            else:
                exclusion_set = set()

            # Now iterate over all prominence and height combinations
            for prominence in prominences_range:
                for height in heights:
                    # Find peaks with current prominence and height
                    # peaks, properties = find_peaks(spectrum, prominence=prominence, height=height, distance = 25)
                    mask = (all_properties['prominences'] >= prominence) & (all_properties['peak_heights'] >= height)
                    peaks = all_broad_peaks[mask]
                    # properties = all_properties
                    if len(peaks) < 6:
                        continue 
                    if len(peaks) > 40:
                        continue
                    
                    if excluded_peaks:
                        if not all(ep in peaks for ep in excluded_peaks):
                            continue
                        
                    # Exclude the specified peaks
                    if exclusion_level > 0:
                        peaks = np.array([p for p in peaks if p not in exclusion_set])
                    if len(peaks) == 0:
                        continue  # All peaks excluded, skip
                    
                    # Initialize lists
                    assigned_peaks = set()
                    laser_peaks_indices = []
                    brillouin_peaks_indices = []

                    # Identify laser peaks and Brillouin peaks based on symmetry
                    for idx in peaks:
                        if idx in assigned_peaks:
                            continue
                        # Find peaks to the left and right
                        left_peaks = peaks[peaks < idx]
                        right_peaks = peaks[peaks > idx]
                        found = False
                        for left_idx in reversed(left_peaks):
                            if left_idx in assigned_peaks:
                                continue
                            d_left = idx - left_idx
                            for right_idx in right_peaks:
                                if right_idx in assigned_peaks:
                                    continue
                                d_right = right_idx - idx

                                # # *** Begin: Height Criteria Check ***
                                laser_peak_height = spectrum[idx]
                                brillouin_left_height = spectrum[left_idx]
                                brillouin_right_height = spectrum[right_idx]
                                
                                # Find laser peak heights as scipy property
                                # laser_peak_height = properties["peak_heights"][np.where(peaks == idx)][0]
                                # brillouin_left_height = properties["peak_heights"][np.where(peaks == left_idx)][0]
                                # brillouin_right_height = properties["peak_heights"][np.where(peaks == right_idx)][0]
                                

                                # Check if Brillouin peaks are within ±300% of the laser peak height
                                # This means Brillouin peak height should be between 25% and 400% of the laser peak height
                                if not (1/height_tolerance * laser_peak_height <= brillouin_left_height <= height_tolerance * laser_peak_height):
                                    continue  # Skip this pair if left Brillouin peak does not meet the criteria
                                if not (1/height_tolerance * laser_peak_height <= brillouin_right_height <= height_tolerance * laser_peak_height):
                                    continue  # Skip this pair if right Brillouin peak does not meet the criteria
                                # *** End: Height Criteria Check ***

                                
                                
                                
                                
                                # Check distance symmetry
                                # Ensure that the distances are approximately equal within the specified tolerance
                                if not symmetry_tolerance <= (d_left / d_right) <= 1/symmetry_tolerance:
                                    continue
                                
                                # Check if the brilluen peaks fwhm are within 0.2 and 0.8
                                # widths_left = peak_widths(spectrum, [left_idx], rel_height=0.5)
                                # fwhm_left = widths_left[0][0] * np.abs(x_axis[1] - x_axis[0])
                                # widths_right = peak_widths(spectrum, [right_idx], rel_height=0.5)
                                # fwhm_right = widths_right[0][0] * np.abs(x_axis[1] - x_axis[0])
                                # avg_fwhm = np.median([fwhm_left, fwhm_right])
                                # if not 0.2 <= avg_fwhm <= 1:
                                #     continue
                                
                                # If scipy find peak bases sizes of the peaks are more than 300% different, skip
                                # bases_left_left = properties["left_bases"][np.where(peaks == left_idx)][0]
                                # bases_left_right = properties["right_bases"][np.where(peaks == left_idx)][0]
                                # bases_left_avg = np.mean([bases_left_left, bases_left_right])
                                # bases_right_left = properties["left_bases"][np.where(peaks == right_idx)][0]
                                # bases_right_right = properties["right_bases"][np.where(peaks == right_idx)][0]
                                # bases_right_avg = np.mean([bases_right_left, bases_right_right])
                                # if not (0.2 <= bases_left_avg / bases_right_avg <= 5):
                                #     print('bases')
                                #     continue
                            
                                # If both criteria are met, assign the peaks
                                laser_peaks_indices.append(idx)
                                brillouin_peaks_indices.extend([left_idx, right_idx])
                                assigned_peaks.update([left_idx, idx, right_idx])
                                found = True
                                break
                            if found:
                                break

                    # Convert to numpy arrays
                    laser_peaks_indices = np.array(laser_peaks_indices)
                    brillouin_peaks_indices = np.array(brillouin_peaks_indices)
                    

            
                    if len(laser_peaks_indices) == 0:
                        continue  # Skip if no laser peaks found
                    # Should have at least 3 laser peaks
                    if len(laser_peaks_indices) < min_laser_peaks:
                        continue
                    
                    # Height (signal) between all laser peaks should be not more than 5 times different
                    laser_heights = [spectrum[idx] for idx in laser_peaks_indices]
                    if not max(laser_heights) / min(laser_heights) <= laser_height_tolerance:
                        continue
                    
                    # Same for brillouin peaks
                    brillouin_heights = [spectrum[idx] for idx in brillouin_peaks_indices]
                    if not max(brillouin_heights) / min(brillouin_heights) <= laser_height_tolerance:
                        continue
                    
                    # Rescale X-axis using the fitted quadratic polynomial y = a*x^2 + b*x + c
                    laser_peaks_positions = x_axis[laser_peaks_indices]
                    N_laser_peaks = len(laser_peaks_positions)
                    laser_peaks_expected_positions = np.arange(N_laser_peaks) * FSR

                    # # Define the residual function for least squares
                    def residuals(params, x, y):
                        a, b, c = params
                        return a * x**2 + b * x + c - y

                    # Initial guess for a, b, c
                    initial_guess = previous_guess  # Small a to start with

                    # Bounds: a > 0, b > 0, c is unbounded
                    lower_bounds = [1e-16, 1e-16, -np.inf]
                    upper_bounds = [np.inf, np.inf, np.inf]

                    # Perform least squares fitting with constraints
                    result = least_squares(
                        residuals,
                        x0=initial_guess,
                        args=(laser_peaks_indices, laser_peaks_expected_positions),
                        bounds=(lower_bounds, upper_bounds)
                    )

                    
                    
                    if not result.success:
                        continue  # Fit did not converge, skip

                    a, b, c= result.x
                    previous_guess = result.x

                    # Compute the rescaled X-axis
                    rescaled_x_axis = a * x_axis**2 + b * x_axis + c
                    
                    # Check if distance between laser peaks is consistent with FSR
                    distances = np.diff(rescaled_x_axis[laser_peaks_indices])
                    if not all(np.abs(d - FSR) <= FSR_tolerance for d in distances):
                        continue
                    # Compute shifts and FWHM
                    temp_shifts = []
                    temp_right_shifts = []
                    temp_left_shifts = []
                    temp_fwhms = []
                    temp_fwhms_right= []
                    temp_fwhms_left = []
                    for lp_idx in laser_peaks_indices:
                        # Left Brillouin peak (Stokes)
                        left_bp_indices = brillouin_peaks_indices[brillouin_peaks_indices < lp_idx]
                        # Right Brillouin peak (Anti-Stokes)
                        right_bp_indices = brillouin_peaks_indices[brillouin_peaks_indices > lp_idx]
                        if len(left_bp_indices) > 0 and len(right_bp_indices) > 0:
                            left_bp_idx = left_bp_indices[-1]
                            right_bp_idx = right_bp_indices[0]

                            # Compute shifts
                            shift = (rescaled_x_axis[right_bp_idx] - rescaled_x_axis[left_bp_idx]) / 2
                            temp_shifts.append(shift)
                            left_shift = rescaled_x_axis[lp_idx] - rescaled_x_axis[left_bp_idx]
                            right_shift = rescaled_x_axis[right_bp_idx] - rescaled_x_axis[lp_idx]
                            temp_right_shifts.append(right_shift)
                            temp_left_shifts.append(left_shift)
                            
                            # Compute FWHM
                            widths_left = peak_widths(spectrum, [left_bp_idx], rel_height=0.5)
                            fwhm_left = widths_left[0][0] * np.abs(rescaled_x_axis[1] - rescaled_x_axis[0])
                            temp_fwhms_left.append(fwhm_left)
                            widths_right = peak_widths(spectrum, [right_bp_idx], rel_height=0.5)
                            fwhm_right = widths_right[0][0] * np.abs(rescaled_x_axis[1] - rescaled_x_axis[0])
                            temp_fwhms_right.append(fwhm_right)
                            avg_fwhm = np.mean([fwhm_left, fwhm_right])
                            temp_fwhms.append(avg_fwhm)

                    # Compute median shift and FWHM
                    if temp_shifts and temp_fwhms:
                        median_shift = np.median(temp_shifts)
                        median_fwhm = np.median(temp_fwhms)

                        # Check if the median_shift is within the expected range
                        if expected_shift_min <= median_shift <= expected_shift_max:
                            if all(
                                np.abs(shift - median_shift) <= shift_tolerance * median_shift
                                for shift in temp_shifts
                            ):
                                # Check if FWHM is within the expected range
                                
                                # if 0.2 <= median_fwhm <= 0.8:
                                #     if all(
                                #         np.abs(fwhm - median_fwhm) <= 0.8 * median_fwhm
                                #         for fwhm in temp_fwhms
                                #     ):
                                best_median_shift = median_shift
                                best_median_fwhm = median_fwhm
                                best_a = a
                                best_b = b  # Quadratic and linear coefficients
                                best_c = c
                                best_laser_peaks_indices = laser_peaks_indices
                                best_brillouin_peaks_indices = brillouin_peaks_indices
                                best_shifts = temp_shifts
                                best_left_shifts = temp_left_shifts
                                best_right_shifts = temp_right_shifts
                                best_fwhms = temp_fwhms
                                best_fwhms_right_list = temp_fwhms_right
                                best_fwhms_left_list = temp_fwhms_left
                                best_rescaled_x_axis = rescaled_x_axis
                                best_excluded_peaks = excluded_peaks
                            break  # Exit the height loop

                if best_median_shift is not None:
                    break  # Exit the prominence loop

            if best_median_shift is not None:
                break  # Exit the exclusion_level loop
        if best_median_shift is not None:
            break
        
    
    # If no valid shift/FWHM is found
    if best_median_shift is None or best_median_fwhm is None:
        # print(f"Failed to find optimal values within the given ranges and exclusion levels.")
        return None, None
    
    # Shift x axis to zero
    best_rescaled_x_axis = best_rescaled_x_axis - best_rescaled_x_axis[0]
    
    # Create a dataframe with the results
    laser_df = pd.DataFrame({
        "Peak Index": best_laser_peaks_indices,
        "Peak Type": "Laser",
        "FWHM": best_fwhms,
        "Shift": best_shifts

    })
    
    brillouin_left_df = pd.DataFrame({
        "Peak Index": best_brillouin_peaks_indices[::2],
        "Peak Type": "Brillouin Left",
        "FWHM": best_fwhms_left_list,
        'Shift': best_left_shifts
    })
    brillouin_right_df = pd.DataFrame({
        "Peak Index": best_brillouin_peaks_indices[1::2],
        "Peak Type": "Brillouin Right",
        "FWHM": best_fwhms_right_list,
        'Shift': best_right_shifts
    })
    # Concatenate the dataframes
    df = pd.concat([laser_df, brillouin_left_df, brillouin_right_df], ignore_index=True)
    # display(df)
    
    # if isinstance(best_excluded_peaks, tuple):
    #     return None
    
    if make_plot:
        plt.figure(figsize=(12, 7))
        
        plt.plot(best_rescaled_x_axis, spectrum, label=f"Spectrum at (x={x}, y={y}, z={z})", color='blue')

        # Highlight laser peaks
        plt.plot(
            best_rescaled_x_axis[best_laser_peaks_indices],
            spectrum[best_laser_peaks_indices],
            "x",
            label="Laser Peaks",
            color='green',
            markersize=10,
            markeredgewidth=4
        )

        # Highlight Brillouin peaks
        plt.plot(
            best_rescaled_x_axis[best_brillouin_peaks_indices],
            spectrum[best_brillouin_peaks_indices],
            "o",
            label="Brillouin Peaks",
            color='red',
            markersize=8
        )

        # Annotate Brillouin shift and FWHM
        for i in range(len(best_shifts)):
            shift = best_shifts[i]
            fwhm = best_fwhms[i]
            lp_idx = best_laser_peaks_indices[i]

            # Position for annotation
            lp_pos = best_rescaled_x_axis[lp_idx]

            # Annotate shift
            plt.annotate(
                f"Shift: {shift:.2f} GHz",
                (lp_pos, spectrum[lp_idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='black'
            )
        
        # Annotate FWHM above brillouin peaks
        for i in range(len(best_fwhms_right_list)):
            plt.annotate(
                f"FWHM: {best_fwhms_right_list[i]:.2f} GHz",
                (best_rescaled_x_axis[best_brillouin_peaks_indices[i*2+1]], spectrum[best_brillouin_peaks_indices[i*2+1]]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='black'
            )
            plt.annotate(
                f"FWHM: {best_fwhms_left_list[i]:.2f} GHz",
                (best_rescaled_x_axis[best_brillouin_peaks_indices[i*2]], spectrum[best_brillouin_peaks_indices[i*2]]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='black'
            )
                
            
            
        # *** Begin: Annotate Distances Between Neighboring Laser Peaks ***
        if len(best_laser_peaks_indices) > 1:
            for i in range(len(best_laser_peaks_indices) - 1):
                # Get the indices of the two neighboring laser peaks
                idx1 = best_laser_peaks_indices[i]
                idx2 = best_laser_peaks_indices[i + 1]

                # Get the rescaled x-axis positions
                x1 = best_rescaled_x_axis[idx1]
                x2 = best_rescaled_x_axis[idx2]

                # Calculate the distance
                distance = x2 - x1

                # Draw a horizontal line with double-headed arrow
                plt.annotate(
                    '',
                    xy=(x1, 0),
                    xytext=(x2, 0),
                    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5)
                )

                # Annotate the distance value at the midpoint
                mid_x = (x1 + x2) / 2
                mid_y = 0
                plt.text(
                    mid_x,
                    mid_y,
                    f"{distance:.2f} GHz",
                    color='green',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        # *** End: Annotate Distances Between Neighboring Laser Peaks ***

        # *** Begin: Annotate Distances Between Brillouin and corresponding Laser Peaks ***
        # Vertical position of arrows so it's below the brillouin arrows
        arrow_y = -max(spectrum) * 0.05
        if len(best_laser_peaks_indices) > 1:
            for i in range(len(best_laser_peaks_indices)):
                lp_idx = best_laser_peaks_indices[i]
                bp_idx = best_brillouin_peaks_indices[i * 2]
                lp_pos = best_rescaled_x_axis[lp_idx]
                bp_pos = best_rescaled_x_axis[bp_idx]
                distance = np.abs(bp_pos - lp_pos)
                plt.annotate(
                    '',
                    xy=(lp_pos, arrow_y),
                    xytext=(bp_pos, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
                )
                mid_x = (lp_pos + bp_pos) / 2
                plt.text(
                    mid_x,
                    arrow_y,
                    f"{distance:.2f} GHz",
                    color='red',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            for i in range(len(best_laser_peaks_indices)):
                lp_idx = best_laser_peaks_indices[i]
                bp_idx = best_brillouin_peaks_indices[i * 2 + 1]
                lp_pos = best_rescaled_x_axis[lp_idx]
                bp_pos = best_rescaled_x_axis[bp_idx]
                distance = np.abs(bp_pos - lp_pos)
                plt.annotate(
                    '',
                    xy=(lp_pos, arrow_y),
                    xytext=(bp_pos, arrow_y),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5)
                )
                mid_x = (lp_pos + bp_pos) / 2
                plt.text(
                    mid_x,
                    arrow_y,
                    f"{distance:.2f} GHz",
                    color='red',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
                    
        # *** End: Annotate Distances Between Brillouin and Laser Peaks ***
        
        
        
        
        
        
        # Plot green vertical lines at the laser peak positions
        for lp_idx in best_laser_peaks_indices:
            lp_pos = best_rescaled_x_axis[lp_idx]
            plt.axvline(x=lp_pos, color='green', linestyle='--', linewidth=1.5)
            
        # Plot red vertical lines at the brillouin peak positions
        for bp_idx in best_brillouin_peaks_indices:
            bp_pos = best_rescaled_x_axis[bp_idx]
            plt.axvline(x=bp_pos, color='red', linestyle='--', linewidth=1.5)
            
        # Highlight excluded peaks with red 'x'
        if exclusion_level > 0 and len(best_excluded_peaks) > 0:
            for ep_idx in best_excluded_peaks:
                plt.plot(
                    best_rescaled_x_axis[ep_idx],
                    spectrum[ep_idx],
                    "x",
                    color='purple',
                    markersize=15,
                    markeredgewidth=4,
                    label='Excluded Peaks' if ep_idx == best_excluded_peaks[0] else ""
                )

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Intensity (Arbitrary Units)")
        plt.title(
            f"Brillouin Spectrum at (x={x}, y={y}, z={z})\n"
            f"Median Shift: {best_median_shift:.2f} GHz, Median FWHM: {best_median_fwhm:.2f} GHz\n"
            f"Fitted Quadratic Coefficients: a={best_a:.4e}, b={best_b:.4f}, c={best_c:.2f}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        

    return df


# Define a handler for timeouts
def timeout_handler(signum, frame):
    raise TimeoutError

# Set the signal to call timeout_handler after a specified timeout period
signal.signal(signal.SIGALRM, timeout_handler)

def get_brillouin_peaks_2d_list(brillouin_spectra, z, shift_range, shift_tolerance, prominence_range, prominence_step, height_range, height_step, height_tolerance=15, symmetry_tolerance=0.5, FSR=29.95, FSR_tolerance = 0.2, max_exclusions=4, timeout_seconds=20):
    """
    This function calculates the Brillouin shift and FWHM for each (x, y) point at a given Z position in the Brillouin spectra.
    
    Parameters:
    brillouin_spectra (np.ndarray): 3D array containing the Brillouin spectra (with shape [x, y, z]).
    z (int): The Z-coordinate for which to calculate the shift and FWHM maps.
    shift_range (tuple): The expected range of Brillouin shifts.
    prominence_range (tuple): Range of prominence values (min, max) for peak detection.
    prominence_step (float): Step size for prominence.
    height_range (tuple): Range of height values (min, max) for peak detection.
    height_step (float): Step size for height.
    FSR (float): Free Spectral Range (Hz). Default is 29.95 Hz.
    timeout_seconds (int): Time limit in seconds for the calculation of each (x, y) point.
    
    Returns:
    shift_map (np.ndarray): 2D array containing the Brillouin shift for each (x, y) point.
    fwhm_map (np.ndarray): 2D array containing the FWHM for each (x, y) point.
    """
    
    # Get the dimensions of the 3D Brillouin spectra
    x_dim, y_dim, z_dim = brillouin_spectra.shape
    
    if z >= z_dim:
        print(f"Z-coordinate {z} is out of bounds.")
        return None, None
    
    # Initialize 2D arrays to store the shifts and FWHM
    # shift_map = np.full((x_dim, y_dim), np.nan)
    # fwhm_map = np.full((x_dim, y_dim), np.nan)
    
    peaks_map = {}
    
    # pbar = tqdm(itertools.product(range(x_dim), range(y_dim)), desc="Processing Brillouin Shift and FWHM")
    pbar = tqdm(itertools.product(range(x_dim), range(y_dim)), desc="Processing Brillouin Shift and FWHM}", total=x_dim*y_dim)
    # Loop through each (x, y) coordinate at the specified Z position
    for x, y in pbar:
        # Add tqdm message about what processing (x, y)
        pbar.set_description(f"Processing Brillouin Shift and FWHM at (x={x}, y={y}, z={z})", refresh=True)
        
        try:
            # Set an alarm to trigger a timeout after `timeout_seconds`
            signal.alarm(timeout_seconds)
            
            # Get the DataFrame containing the Brillouin peaks data for the current (x, y, z) position
            df_peaks = analyze_brillouin(
                brillouin_spectra=brillouin_spectra,
                x=x,
                y=y,
                z=z,
                shift_range=shift_range,
                shift_tolerance=shift_tolerance,
                prominence_range=prominence_range,
                prominence_step=prominence_step,
                height_range=height_range,
                height_step=height_step,
                height_tolerance=height_tolerance,
                symmetry_tolerance=symmetry_tolerance,
                FSR=FSR,
                FSR_tolerance=FSR_tolerance,
                max_exclusions=max_exclusions,
                make_plot=False
            )
            
            # Reset the alarm after successful calculation
            signal.alarm(0)
            
            # Check if the DataFrame is valid and has data
            if df_peaks is not None:
                peaks_map[x, y] = df_peaks
            else:
                # If no valid data is found, assign NaN
                peaks_map[x, y] = np.nan


        except TimeoutError:
            # If the calculation takes longer than `timeout_seconds`, assign NaN
            # print(f"Timeout at (x={x}, y={y}, z={z})")
            # Reset the alarm to avoid accidental triggers
            signal.alarm(0)

    return peaks_map

def extract_median_values(df_map, data_type="shift"):
    """
    Extracts the median "Shift" or "FWHM" values from a dictionary of DataFrames or NaNs.
    
    Parameters:
    -----------
    df_map : dict
        A dictionary where keys are tuples (x, y) and values are DataFrames or NaNs.
    data_type : str, optional
        Specifies which data to extract: "shift" or "fwhm". Default is "shift".
        
    Returns:
    --------
    np.ndarray
        A 2D array containing the extracted median values.
    """
    # Get the dimensions of the data
    x_dim = max([key[0] for key in df_map.keys()]) + 1
    y_dim = max([key[1] for key in df_map.keys()]) + 1
    
    # Initialize an array with NaN values
    data_map = np.full((x_dim, y_dim), np.nan)
    
    # Loop over the dictionary and extract values
    for (i, j), df in df_map.items():
        if df is not None and not isinstance(df, float) and not isinstance(df, tuple):
            # Extract "Shift" or "FWHM" based on the data_type
            brillouin_peaks = df[df["Peak Type"].str.contains("Brillouin")]
            if not brillouin_peaks.empty:
                if data_type == "shift":
                    data_map[i, j] = brillouin_peaks["Shift"].median()
                elif data_type == "fwhm":
                    data_map[i, j] = brillouin_peaks["FWHM"].median()
    return data_map

def extract_first(df_map, data_type="shift"):
    """
    Extracts the median "Shift" or "FWHM" values from the first two Brillouin peaks (Stokes and Anti-Stokes)
    from a dictionary of DataFrames or NaNs.
    
    Parameters:
    -----------
    df_map : dict
        A dictionary where keys are tuples (x, y) and values are DataFrames or NaNs.
    data_type : str, optional
        Specifies which data to extract: "shift" or "fwhm". Default is "shift".
        
    Returns:
    --------
    np.ndarray
        A 2D array containing the extracted median values from the first two Brillouin peaks.
    """
    # Get the dimensions of the data
    x_dim = max([key[0] for key in df_map.keys()]) + 1
    y_dim = max([key[1] for key in df_map.keys()]) + 1
    
    # Initialize an array with NaN values
    data_map = np.full((x_dim, y_dim), np.nan)
    
    # Loop over the dictionary and extract values from the first two Brillouin peaks
    for (i, j), df in df_map.items():
        if df is not None and not isinstance(df, float) and not isinstance(df, tuple):
            # Extract "Shift" or "FWHM" for the first two Brillouin peaks (Stokes and Anti-Stokes)
            brillouin_peaks = df[df["Peak Type"].str.contains("Brillouin")]
            if len(brillouin_peaks) >= 2:
                first_two_peaks = brillouin_peaks.iloc[:2]
                if data_type == "shift":
                    # Take the mean of the shifts of the first two peaks
                    data_map[i, j] = first_two_peaks["Shift"].mean()
                elif data_type == "fwhm":
                    # Take the mean of the FWHM of the first two peaks
                    data_map[i, j] = first_two_peaks["FWHM"].mean()
    return data_map

def plot_brillouin_heatmap(df_list, title, data_type="shift", peaks='all',pixel_size = 1, apply_median_filter=False, median_filter_size=3,
                           apply_gaussian_filter=False, gaussian_sigma=1,
                           interpolate_nan=True, colorbar_range=None,
                           cmap='jet', interpolation=None, annotate=True):
    """
    Plots a 2D heatmap based on median values extracted from a 2D list of DataFrames or NaNs.
    
    Parameters:
    -----------
    df_list : list of list of pd.DataFrame or NaN
        2D list containing DataFrames or NaNs. Each DataFrame contains Brillouin data.
    title : str
        Title of the heatmap.
    data_type : str, optional
        Specifies which data to extract: "shift" or "fwhm". Default is "shift".
    apply_median_filter : bool, optional
        Whether to apply a median filter to the data. Default is False.
    median_filter_size : int, optional
        Size of the median filter window. Default is 3.
    apply_gaussian_filter : bool, optional
        Whether to apply a Gaussian filter to the data. Default is False.
    gaussian_sigma : float, optional
        Sigma value for the Gaussian filter. Default is 1.
    interpolate_nan : bool, optional
        Whether to interpolate over NaN values in the data. Default is True.
    colorbar_range : tuple of (float, float) or None, optional
        Tuple specifying the (vmin, vmax) for the colorbar. If None, uses data's min and max.
    cmap : str, optional
        Colormap for the heatmap. Default is 'viridis'.
    annotate : bool, optional
        Whether to add annotations for statistical information. Default is True.
        
    Returns:
    --------
    None
    """
    # Extract median "Shift" or "FWHM" values from the 2D list of DataFrames
    if peaks == 'all':
        data_map = extract_median_values(df_list, data_type=data_type)
    if peaks == 'first':
        data_map = extract_first(df_list, data_type=data_type)
    
    # Make a copy to avoid modifying the original data
    processed_data = np.copy(data_map)
    
    # Interpolate NaN values if required
    if interpolate_nan:
        mask = ~np.isnan(processed_data)
        if np.sum(mask) == 0:
            raise ValueError("All data points are NaN. Cannot perform interpolation.")
        
        # Coordinates of valid and invalid points
        x, y = np.indices(processed_data.shape)
        x_valid = x[mask]
        y_valid = y[mask]
        data_valid = processed_data[mask]
        
        x_nan = x[~mask]
        y_nan = y[~mask]
        
        if len(x_nan) > 0:
            # Perform linear interpolation
            interpolated_values = griddata(
                points=(x_valid, y_valid),
                values=data_valid,
                xi=(x_nan, y_nan),
                method='linear'
            )
            # Handle any remaining NaNs with nearest interpolation
            nan_after_interp = np.isnan(interpolated_values)
            if np.any(nan_after_interp):
                interpolated_values[nan_after_interp] = griddata(
                    points=(x_valid, y_valid),
                    values=data_valid,
                    xi=(x_nan[nan_after_interp], y_nan[nan_after_interp]),
                    method='nearest'
                )
            # Fill the NaNs with interpolated values
            processed_data[~mask] = interpolated_values

    # Apply median filter if required
    if apply_median_filter:
        processed_data = median_filter(processed_data, size=median_filter_size)
    
    # Apply Gaussian filter if required
    if apply_gaussian_filter:
        processed_data = gaussian_filter(processed_data, sigma=gaussian_sigma)
    
    # Plot using imshow
    plt.figure(figsize=(10, 8))
    im = plt.imshow(processed_data, cmap=cmap, origin='lower', aspect='auto',
                    vmin=colorbar_range[0] if colorbar_range else None,
                    vmax=colorbar_range[1] if colorbar_range else None, interpolation=interpolation)
    plt.title(title, fontsize=14)
    plt.xlabel(r"Y Coordinate $\mu m$ ", fontsize=12)
    plt.ylabel(r"X Coordinate $\mu m$", fontsize=12)
    # Rescale the x-axis to micrometers
    x_ticks = np.arange(0, processed_data.shape[1], 5)
    x_labels = [f"{x * pixel_size:.2f}" for x in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45)
    # Rescale the y-axis to micrometers
    y_ticks = np.arange(0, processed_data.shape[0], 5)
    y_labels = [f"{y * pixel_size:.2f}" for y in y_ticks]
    plt.yticks(y_ticks, y_labels)
    
    
    
    # Add colorbar with label
    cbar = plt.colorbar(im)
    cbar_label = "Shift (GHz)" if data_type == "shift" else "FWHM (GHz)"
    cbar.set_label(cbar_label, fontsize=12)
    
    # Optionally annotate statistical information
    if annotate:
        mean_val = np.nanmean(processed_data)
        median_val = np.nanmedian(processed_data)
        std_val = np.nanstd(processed_data)
        plt.text(0.95, 0.95, f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes,
                 color='white', fontsize=10,
                 bbox=dict(facecolor='black', alpha=0.5, pad=5))
    
    plt.tight_layout()
    plt.show()
    
    
def plot_3d_surface_heatmaps(df_maps, z_values, title, data_type="shift", peaks='all', pixel_size=1, 
                             apply_median_filter=False, median_filter_size=3,
                             apply_gaussian_filter=False, gaussian_sigma=1,
                             interpolate_nan=True, cmap='jet', colorbar_range=None):
    """
    Plots a 3D surface map of multiple heatmaps, each corresponding to a different Z level.
    
    Parameters:
    -----------
    df_maps : list of list of pd.DataFrame or NaN
        A list of 2D lists containing DataFrames or NaNs, one for each Z level.
    z_values : list
        A list of Z values corresponding to each heatmap (level).
    title : str
        Title of the 3D plot.
    data_type : str, optional
        Specifies which data to extract: "shift" or "fwhm". Default is "shift".
    peaks : str, optional
        Specifies whether to extract "all" Brillouin peaks or just the "first" two (Stokes and Anti-Stokes). Default is 'all'.
    pixel_size : float, optional
        Size of each pixel in micrometers. Default is 1.
    apply_median_filter : bool, optional
        Whether to apply a median filter to the data. Default is False.
    median_filter_size : int, optional
        Size of the median filter window. Default is 3.
    apply_gaussian_filter : bool, optional
        Whether to apply a Gaussian filter to the data. Default is False.
    gaussian_sigma : float, optional
        Sigma value for the Gaussian filter. Default is 1.
    interpolate_nan : bool, optional
        Whether to interpolate over NaN values in the data. Default is True.
    cmap : str, optional
        Colormap for the surface maps. Default is 'jet'.
    colorbar_range : tuple of (float, float) or None, optional
        Tuple specifying the (vmin, vmax) for the colorbar. If None, uses data's min and max.
    
    Returns:
    --------
    None
    """
    fig = plt.figure(figsize=(12, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get x, y coordinates based on the pixel grid
    x_dim, y_dim = max([key[0] for key in df_maps[0].keys()]) + 1, max([key[1] for key in df_maps[0].keys()]) + 1
    x = np.arange(0, x_dim) * pixel_size
    y = np.arange(0, y_dim) * pixel_size
    X, Y = np.meshgrid(x, y)
    
    # Determine global min and max for normalization if colorbar_range is not provided
    if colorbar_range is None:
        all_data = []
        for df_map in df_maps:
            if peaks == 'all':
                all_data.append(extract_median_values(df_map, data_type=data_type))
            elif peaks == 'first':
                all_data.append(extract_first(df_map, data_type=data_type))
        all_data = np.array(all_data)
        vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)
    else:
        vmin, vmax = colorbar_range

    # Loop through each Z level and plot the corresponding heatmap as a surface
    for df_map, z in zip(df_maps, z_values):
        if peaks == 'all':
            data_map = extract_median_values(df_map, data_type=data_type)
        elif peaks == 'first':
            data_map = extract_first(df_map, data_type=data_type)
        
        # Apply filters and interpolation if necessary
        processed_data = np.copy(data_map)
        
        if interpolate_nan:
            mask = ~np.isnan(processed_data)
            if np.sum(mask) > 0:
                x_valid, y_valid = np.indices(processed_data.shape)
                x_valid = x_valid[mask]
                y_valid = y_valid[mask]
                data_valid = processed_data[mask]
                
                # Generate interpolation points for NaN locations
                x_nan, y_nan = np.indices(processed_data.shape)
                x_nan = x_nan[~mask]
                y_nan = y_nan[~mask]
                
                # Perform interpolation over NaNs
                interpolated_values = griddata((x_valid, y_valid), data_valid, (x_nan, y_nan), method='linear')
                
                # Assign interpolated values back to the NaN locations in processed_data
                processed_data[~mask] = interpolated_values
        
        if apply_median_filter:
            processed_data = median_filter(processed_data, size=median_filter_size)
        
        if apply_gaussian_filter:
            processed_data = gaussian_filter(processed_data, sigma=gaussian_sigma)
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Plot the surface at this Z level (Z is fixed for each layer)
        facecolors = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(processed_data)
        ax.plot_surface(X, Y, np.full_like(X, z), facecolors=facecolors, 
                        rstride=1, cstride=1, shade=False, zorder=z, antialiased=False)
    
    # Set plot titles and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'X Coordinate ($\mu m$)', fontsize=12)
    ax.set_ylabel(r'Y Coordinate ($\mu m$)', fontsize=12)
    ax.set_zlabel(r'Layer Coordinate ($\mu m$)', fontsize=12)
    ax.set_zlim(min(z_values), max(z_values))
    

    
    # Z axis scale should be in micrometers
    z_ticks = np.arange(0, max(z_values) + 1, 1)
    z_labels = [f"{z * pixel_size:.2f}" for z in z_ticks]
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels)
    
    # Change aspect of Z
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.5, 1]))
    
    
    # Display colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.array([]))  # Colorbar needs an array to map to colors
    fig.colorbar(mappable, shrink=0.5, aspect=5, label=f'{data_type.capitalize()} (GHz)')

    plt.show()