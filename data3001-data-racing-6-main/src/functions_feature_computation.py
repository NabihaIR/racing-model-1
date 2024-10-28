################################################################################
# Functions for Computing the Features For the Final Dataset
#
# These will sadly break if you call them individually at the moment (oops).
# If you want to do that, please first import settings, then call
# `settings.set_local_data_directory` with your local data directory as the
# argument.
#
################################################################################

import pandas as pd
import numpy as np
from functions import load_turn_data, update_data_w_distances, calculate_area
import settings

def compute_distace_to_apex(race_data: pd.DataFrame):
    """
    Given a dataframe in the 2022/2023 format, returns the same dataframe with
    two new columns, 'dist_apex_1' and 'dist_apex_2', which are the euclidean
    distances from the car's position to the apexes of turn 1 and turn 2.
    """
    turns = load_turn_data(settings.LOCAL_DATA_DIR)
    turn_1 = turns[turns['TURN'] == 1]
    apex_turn_1 = (turn_1['APEX_X1'].values[0], turn_1['APEX_Y1'].values[0])
    turn_2 = turns[turns['TURN'] == 2]
    apex_turn_2 = (turn_2['APEX_X1'].values[0], turn_2['APEX_Y1'].values[0])

    race_data['dist_apex_1'] = euclidean_distance(
        race_data['WORLDPOSX'], # x1
        race_data['WORLDPOSY'], # y1
        apex_turn_1[0],         # x2
        apex_turn_1[1]          # y2
    )
    race_data['dist_apex_2'] = euclidean_distance(
        race_data['WORLDPOSX'], # x1
        race_data['WORLDPOSY'], # y1
        apex_turn_2[0],         # x2
        apex_turn_2[1]          # y2
    )
    return race_data


def compute_distance_to_track_edges(race_data: pd.DataFrame):
    """
    Forwards to Tom's function in the 'functions.py' file. Only included here
    to put all feature computation functions in one place.
    """
    return update_data_w_distances(race_data, settings.LOCAL_DATA_DIR)


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def compute_angle_to_apex(df: pd.DataFrame, apex1: list, apex2: list):
    # each apex is a list; 0 index is x coord, 1 is the y coord

    """
    Inputs a df, returns a copy of the df with angle to apex1 and apex2 columns
    """

    df_new = df.copy()

    required_columns = ['WORLDFORWARDDIRX', 'WORLDFORWARDDIRY', 'WORLDPOSX', 'WORLDPOSY']

    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    df_new['forward_vector'] = list(zip(df_new['WORLDFORWARDDIRX'], df_new['WORLDFORWARDDIRY']))
    df_new['world_pos'] = list(zip(df_new['WORLDPOSX'], df_new['WORLDPOSY']))

    df_new['apex_vector1'] = [(apex1[0] - pos[0], apex1[1] - pos[1]) for pos in df_new['world_pos']]
    df_new['apex_vector2'] = [(apex2[0] - pos[0], apex2[1] - pos[1]) for pos in df_new['world_pos']]

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the signed angle in degrees between vectors 'v1' and 'v2' clockwise. """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)

        # Compute the dot product and angle
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot_product)

        # Compute the cross product (determinant) in 2D to find the sign of the angle
        cross = v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0]

        # If the cross product is negative, it means the angle is clockwise, so negate it
        if cross < 0:
            angle = -angle

        # Convert the angle from radians to degrees
        return np.degrees(angle)

    # Apply the angle function to calculate angles to both apexes
    df_new['angle_to_apex1'] = [angle_between(forward, apex) for forward, apex \
                    in zip(df_new['forward_vector'], df_new['apex_vector1'])]
    df_new['angle_to_apex2'] = [angle_between(forward, apex) for forward, apex \
                    in zip(df_new['forward_vector'], df_new['apex_vector2'])]

    df_new = df_new.drop(['apex_vector1', 'apex_vector2', 'world_pos',
            'forward_vector'], axis=1)
    return df_new


def compute_braking_throttle_areas(race_data: pd.DataFrame, moment_lap_dists: list):
    """
    Given a dataframe in the 2022/2023 format, along with the list of lap
    distances being used for the fixed moments, returns a dataframe with one row
    per lap containing the areas under the throttle and braking curves between
    each fixed distance moment.
    """
    result_df = race_data[['SESSION_IDENTIFIER', 'LAP_NUM']] \
        .drop_duplicates() \
        .sort_values(['SESSION_IDENTIFIER', 'LAP_NUM'])

    i = 0
    while i < len(moment_lap_dists) - 1:
        interval_start = moment_lap_dists[i]
        interval_end = moment_lap_dists[i+1]
        interval_braking = []
        interval_throttle = []
        for _, group in race_data.groupby(['SESSION_IDENTIFIER', 'LAP_NUM']):
            # Get total braking and throttle in the interval between the two moments
            interval_braking.append(
                calculate_area(group, interval_start, interval_end, 'BRAKE'))
            interval_throttle.append(
                calculate_area(group, interval_start, interval_end, 'THROTTLE'))

        result_df[f'TOTAL_BRAKING_{interval_start}_{interval_end}'] = interval_braking
        result_df[f'TOTAL_THROTTLE_{interval_start}_{interval_end}'] = interval_throttle
        i += 1

    return result_df
