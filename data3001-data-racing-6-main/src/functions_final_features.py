################################################################################
# Functions for Building the Final Dataset
#
# Usage example (in your scripts):
#   from functions import load_track_section
#   track = load_track_section()
#
################################################################################

import pandas as pd
from functions import create_invalid_lap_flag, get_apex_coords
from functions_interpolation import interpolate_section_times
from functions_feature_computation import compute_distace_to_apex, \
    compute_distance_to_track_edges, compute_angle_to_apex, \
    compute_braking_throttle_areas
import settings
import re
import time

ID_COLS = ['SESSION_IDENTIFIER', 'LAP_NUM']
FIXED_DISTANCE_MOMENTS = [360, 387, 414, 441, 468, 495, 522]

def build_dataset(race_data: pd.DataFrame, local_data_dir: str):
    """
    Given a dataframe in the 2022/2023 format, returns a dataset with one row
    per lap containing all of our derived features ready for modelling.
    """
    overall_start = time.time()
    settings.set_local_data_dir(local_data_dir)
    all_feature_tables = []

    print('Building dataset...\n---------------------')

    start = time.time()
    first_brake_moment = get_first_brake(race_data)
    first_brake_features = compute_features_for_moment(first_brake_moment, 'FB')
    all_feature_tables.append(first_brake_features)
    end = time.time()
    print(f'    First brake features ({first_brake_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    max_brake_moment = generic_get_max(race_data, 'BRAKE')
    max_brake_features = compute_features_for_moment(max_brake_moment, 'MB')
    all_feature_tables.append(max_brake_features)
    end = time.time()
    print(f'    Max brake features ({max_brake_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    first_steer_moment = get_first_steering(race_data)
    first_steer_features = compute_features_for_moment(first_steer_moment, 'FS')
    all_feature_tables.append(first_steer_features)
    end = time.time()
    print(f'    First steering features ({first_steer_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    max_steer_moment = generic_get_max(race_data, 'STEERING')
    max_steer_features = compute_features_for_moment(max_steer_moment, 'MS')
    all_feature_tables.append(max_steer_features)
    end = time.time()
    print(f'    Max steering features ({max_steer_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    first_throttle_moment = get_first_throttle_after_initial_brake(race_data)
    first_throttle_features = compute_features_for_moment(first_throttle_moment, 'FT')
    all_feature_tables.append(first_throttle_features)
    end = time.time()
    print(f'    First throttle features ({first_throttle_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    max_throttle_moment = get_max_throttle_after_initial_brake(race_data)
    max_throttle_features = compute_features_for_moment(max_throttle_moment, 'MT')
    all_feature_tables.append(max_throttle_features)
    end = time.time()
    print(f'    Max throttle features ({max_throttle_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    # Compute and add features for fixed distance moments
    for lap_dist in FIXED_DISTANCE_MOMENTS:
        start = time.time()
        fixed_dist_moment = get_fixed_distance_moment(race_data, lap_dist)
        fixed_dist_features = compute_features_for_moment(fixed_dist_moment, str(lap_dist))
        all_feature_tables.append(fixed_dist_features)
        end = time.time()
        print(f'    Fixed distance {lap_dist}m features ({fixed_dist_features.shape[0]} rows) took {round(end-start, 2)} seconds')

    # Compute and add standalone features
    start = time.time()
    invalid_lap_flag = create_invalid_lap_flag(race_data, local_data_dir) # takes 2+ mins
    invalid_lap_flag = invalid_lap_flag[ID_COLS + ['INVALID_LAP']].drop_duplicates()
    all_feature_tables.append(invalid_lap_flag)
    end = time.time()
    print(f'    Invalid lap flag ({invalid_lap_flag.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    section_times = interpolate_section_times(race_data)
    all_feature_tables.append(section_times)
    end = time.time()
    print(f'    Section time ({section_times.shape[0]} rows) took {round(end-start, 2)} seconds')

    start = time.time()
    area_braking_throttle = compute_braking_throttle_areas(race_data, FIXED_DISTANCE_MOMENTS)
    all_feature_tables.append(area_braking_throttle)
    end = time.time()
    print(f'    Braking/throttle areas ({area_braking_throttle.shape[0]} rows) took {round(end-start, 2)} seconds')

    # Join all feature tables
    start = time.time()
    joined = pd.merge(all_feature_tables[0], all_feature_tables[1], on=ID_COLS)
    for feature_table in all_feature_tables[2:]:
        joined = pd.merge(joined, feature_table, on=ID_COLS)
    end = time.time()
    print(f'    Joining all feature tables took {round(end-start, 2)} seconds')

    overall_end = time.time()
    print('---------------------')
    print(f'{joined.shape[0]} rows, {joined.shape[1]} columns')
    print(f"Total time taken: {round(overall_end-overall_start, 2)} seconds")
    return joined.drop_duplicates()


def get_first_steering(race_data: pd.DataFrame):
    """
    A function to get the row where the STEERING first is greater than some
    threshold.
    """
    STEERING_THRESHOLD = 0.075
    return race_data[race_data['STEERING'] > STEERING_THRESHOLD] \
        .sort_values(['CURRENT_LAP_TIME_MS'], ascending=True) \
        .groupby(ID_COLS) \
        .first() \
        .reset_index()


def get_first_brake(race_data: pd.DataFrame):
    """
    A function to get the row for the first brake moment (defined as the point
    where BRAKE > THROTTLE).
    """
    return race_data[race_data['BRAKE'] > race_data['THROTTLE']] \
        .sort_values(['CURRENT_LAP_TIME_MS'], ascending=True) \
        .groupby(ID_COLS) \
        .first() \
        .reset_index()


def get_first_throttle_after_initial_brake(race_data: pd.DataFrame):
    """
    A function to get the row for the moment a driver hits the throttle after
    initially braking for turn 1.
    """
    first_throttle_idx_list = []

    grouped = race_data.groupby(ID_COLS)
    for _, group in grouped:

        # Iterate through rows to find first point where the throttle > brake
        # -> this is our definition of first throttle point
        # If such a point doesn't exist, then we take the first throttle point to be
        # some lap distance (one that's obviously too late) into the lap
        has_braked = False
        first_throttle_found = False
        first_throttle_idx = None
        for idx, row in group.sort_values(['CURRENT_LAP_TIME_MS'], ascending=True).iterrows():
            if has_braked and row['THROTTLE'] > row['BRAKE']:
                first_throttle_idx = idx
                first_throttle_found = True
                break
            else:
                has_braked = has_braked or row['BRAKE'] >= row['THROTTLE']

        if not first_throttle_found:
            # If no throttle point found after initial brake (i.e. there wasn't an
            # initial brake), then take closest point to lap distance 700m to be
            # first throttle
            FIRST_THROTTLE_DEFAULT = 700
            group['dist_to_700'] = abs(group['LAP_DISTANCE'] - FIRST_THROTTLE_DEFAULT)
            first_throttle_idx = group['dist_to_700'].idxmin()

        first_throttle_idx_list.append(first_throttle_idx)

    return race_data.loc[first_throttle_idx_list]


def get_max_throttle_after_initial_brake(race_data: pd.DataFrame):
    """
    A function to get the row for the moment a driver first hits maximum
    throttle after initially braking for turn 1.
    """
    max_throttle_idx_list = []

    # Logic below is based largely on the same logic as
    # 'get_first_throttle_after_initial_brake' above

    grouped = race_data.groupby(ID_COLS)
    for _, group in grouped:
        has_braked = False
        max_throttle_found = False
        max_throttle_idx = None
        for idx, row in group.sort_values(['CURRENT_LAP_TIME_MS'], ascending=True).iterrows():
            if not has_braked and (row['BRAKE'] >= row['THROTTLE']):
                has_braked = True
                brake_idx = idx
                max_throttle_after_brake = max(group.loc[brake_idx:]['THROTTLE'])
            elif has_braked and row['THROTTLE'] == max_throttle_after_brake:
                max_throttle_idx = idx
                max_throttle_found = True
                break

        if not max_throttle_found:
            MAX_THROTTLE_DEFAULT = 700
            group['dist_to_700'] = abs(group['LAP_DISTANCE'] - MAX_THROTTLE_DEFAULT)
            max_throttle_idx = group['dist_to_700'].idxmin()

        max_throttle_idx_list.append(max_throttle_idx)

    return race_data.loc[max_throttle_idx_list]


def generic_get_max(race_data: pd.DataFrame, feature: str):
    """
    A generic function to get the row containing the maximum value of the
    specified feature for each lap.
    """
    if feature not in ['THROTTLE', 'BRAKE', 'STEERING']:
        raise ValueError(f'Feature {feature} not supported for first/max analysis.')

    max_row_indices = race_data.groupby(ID_COLS).idxmax()[feature]
    return race_data.loc[max_row_indices]


def get_fixed_distance_moment(race_data: pd.DataFrame, distance: int):
    """
    A function to get the row for the moments at (or nearest to) fixed INTEGER
    lap distances (e.g. 350m, 400m, 450m, 500m, 550m).
    """
    temp_name = f'dist_to_{distance}'
    race_data[temp_name] = abs(race_data['LAP_DISTANCE'] - distance)
    idxs = race_data.groupby(ID_COLS)[temp_name].idxmin()
    race_data.drop(temp_name, axis=1, inplace=True)
    return race_data.loc[idxs]


def compute_features_for_moment(moment_data: pd.DataFrame, pref: str):
    """
    A function to compute the set list of features given the data for a single
    moment (one row per lap). All transformations are column-wise; that is, the
    number of output rows is the same as the number of input rows.

    Args:
        moment_data: A dataframe containing the race data for a single moment
            (one row per lap). This must contain all columns in the 2022/2023
            format.
        pref: A string to prefix to the new feature names (e.g. FB for
            first brake).
    Returns:
        A dataframe containing the features for this moment.
    """
    # Add any new required columns
    apex1, apex2 = get_apex_coords(settings.LOCAL_DATA_DIR)
    moment_data = compute_angle_to_apex(moment_data, apex1, apex2)
    moment_data = compute_distace_to_apex(moment_data)
    moment_data = compute_distance_to_track_edges(moment_data)

    moment_data[pref + '_DIST_FROM_LEFT'] = moment_data['DISTFROMLEFT']
    moment_data[pref + '_DIST_FROM_RIGHT'] = moment_data['DISTFROMRIGHT']
    moment_data[pref + '_XPOS'] = moment_data['WORLDPOSX']
    moment_data[pref + '_YPOS'] = moment_data['WORLDPOSY']
    moment_data[pref + '_SPEED'] = moment_data['SPEED_KPH']
    moment_data[pref + '_LAP_DIST'] = moment_data['LAP_DISTANCE']
    moment_data[pref + '_THROTTLE'] = moment_data['THROTTLE']
    moment_data[pref + '_BRAKE'] = moment_data['BRAKE']
    moment_data[pref + '_STEERING'] = moment_data['STEERING']
    moment_data[pref + '_CURR_LAPTIME'] = moment_data['CURRENT_LAP_TIME_MS']
    moment_data[pref + '_GEAR'] = moment_data['GEAR']
    moment_data[pref + '_ENGINE_RPM'] = moment_data['ENGINE_RPM']
    moment_data[pref + '_DIST_APEX_1'] = moment_data['dist_apex_1']
    moment_data[pref + '_DIST_APEX_2'] = moment_data['dist_apex_2']
    moment_data[pref + '_ANGLE_APEX_1'] = moment_data['angle_to_apex1']
    moment_data[pref + '_ANGLE_APEX_2'] = moment_data['angle_to_apex2']

    # Remove all features (except SESSION_ID and LAP_NUM) that don't contain
    # the moment prefix
    remove_cols = [col for col in moment_data.columns.to_list() if \
                   re.match('[^_]+', col).group(0) != pref]
    remove_cols.remove(ID_COLS[0])
    remove_cols.remove(ID_COLS[1])

    moment_data.drop(remove_cols, axis=1, inplace=True)
    return moment_data
