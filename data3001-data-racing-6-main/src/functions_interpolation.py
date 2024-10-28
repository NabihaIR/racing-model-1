################################################################################
# Interpolation Functions
#
# Useful functions for linear interpolation (Algorithm 3 in PDF).
#
# Usage example (in your scripts):
#   from functions import load_track_section
#   track = load_track_section()
#
################################################################################

import pandas as pd

# Lap distance at our finish line is roughly 768m. Since this is where the track
# and race data is cut off, we can't get points that are AFTER this... so, we
# consider the end of the track 750m.
END_LAP_DIST = 750

def get_closest_points_to_cutoff(race_data: pd.DataFrame, feature: str):
    """
    Given race data in the 2022/2023 format, returns a dataframe containing
    the closest points before and after the cutoff, as well as all other info
    required for linear interpolation.
    """

    race_data['lap_dist_to_finish'] = END_LAP_DIST - race_data['LAP_DISTANCE']
    race_data['lap_dist_from_finish'] = race_data['LAP_DISTANCE'] - END_LAP_DIST

    cols = ['SESSION_IDENTIFIER', 'LAP_NUM', 'WORLDPOSX', 'WORLDPOSY',
            'LAP_DISTANCE', feature]

    # If this is the 2024 data, need to handle separately
    if '212F4920E1E312BFE0631218000A84D4' in race_data['SESSION_IDENTIFIER'].unique():
        # ^ this is a session ID that only belongs in the 2024 data
        race_data = race_data.loc[race_data[['SESSION_IDENTIFIER', 'LAP_NUM', 'WORLDPOSX', 'WORLDPOSY']].drop_duplicates().index]

    points_before = race_data[race_data['lap_dist_to_finish'] > 0]
    points_after = race_data[race_data['lap_dist_from_finish'] > 0]

    # Below dataframes have 1 row per lap, with the closest point (before/after) to the finish
    closest_point_before = points_before.loc[
            points_before.groupby(['SESSION_IDENTIFIER', 'LAP_NUM'])['lap_dist_to_finish'].idxmin()
        ][cols]
    closest_point_after = points_after.loc[
            points_after.groupby(['SESSION_IDENTIFIER', 'LAP_NUM'])['lap_dist_from_finish'].idxmin()
        ][cols]

    closest_point_before.rename(
        columns={
            'WORLDPOSX': 'x_before',
            'WORLDPOSY': 'y_before',
            'LAP_DISTANCE': 'lap_dist_before',
            feature: f'{feature}_before'
        },
        inplace=True
    )
    closest_point_after.rename(
        columns={
            'WORLDPOSX': 'x_after',
            'WORLDPOSY': 'y_after',
            'LAP_DISTANCE': 'lap_dist_after',
            feature: f'{feature}_after'
        },
        inplace=True
    )

    return pd.merge(closest_point_before, closest_point_after,
                      on=['SESSION_IDENTIFIER', 'LAP_NUM'])


def compute_interpolation_ratios(race_data: pd.DataFrame, feature: str):
    """
    Computes the interpolation ratios ('c' in Algorithm 3 - see PDF) for each
    lap, using linear interpolation with the two closest points to the lap
    distance cutoff (with one on each side). The interpolation ratio allows any
    of the features to be linearly interpolated to find its value at the track
    section cutoff.

    Args:
        race_data: a dataframe of race data in 2022/2023 format.
    Returns:
        The input dataframe with a new column for the interpolation ratios ('c')
        for each lap.
    """
    closest_points = get_closest_points_to_cutoff(race_data, feature)

    # Sorry for this horribly long line - the value 'c' is the lap distance from
    # the after point to END_LAP_DIST divided by the Euclidean distance between
    # the before and after points. See Algorithm 3 in the PDF for details.
    closest_points['c'] = (closest_points['lap_dist_after'] - END_LAP_DIST) / ((closest_points['x_after'] - closest_points['x_before'])**2 + (closest_points['y_after'] - closest_points['y_before'])**2 )**0.5
    return closest_points


def interpolate_section_times(race_data: pd.DataFrame):
    """
    Computes the time taken to complete the section of the track for each unique
    lap (identified by SESSION_IDENTIFIER + LAP_NUM), using linear interpolation
    with the two closest points to the lap distance cutoff (with one on each
    side).

    Args:
        race_data: a dataframe of race data in the 2022/2023 format.
    Returns:
        A dataframe containing the time taken for the track section, with one
        row for each lap.
    """
    feature = 'CURRENT_LAP_TIME_MS'
    df = compute_interpolation_ratios(race_data, feature)
    df['SECTION_TIME_MS'] = round((1-df['c']) * df[f'{feature}_before'] + df['c'] * df[f'{feature}_after'])
    output_cols = ['SESSION_IDENTIFIER', 'LAP_NUM', 'SECTION_TIME_MS']
    section_times = df.drop(
        columns=[col for col in df.columns.to_list() if col not in output_cols]
    )
    return section_times


def add_section_time_column(race_data: pd.DataFrame):
    """
    Adds a new column to the given dataframe containing the time taken to
    complete our defined section of track, estimated by linear interpolation.

    Args:
        race_data: a dataframe of race data in the 2022/2023 format.
    Returns:
        The original dataframe with an additional column for the time taken to
        complete the track section.

    This does the same thing as calling `compute_feature_at_section_cutoff` with
    'CURRENT_LAP_TIME_MS' as the feature parameter, except that the latter does
    not round the computed value to the nearest millisecond. This function is
    provided simply as a convenience.
    """
    section_times = interpolate_section_times(race_data)
    section_times_for_join = section_times[['SESSION_IDENTIFIER', 'LAP_NUM', 'SECTION_TIME_MS']]
    race_data_with_time = pd.merge(race_data, section_times_for_join, on=['SESSION_IDENTIFIER', 'LAP_NUM'])
    return race_data_with_time


def compute_feature_at_section_cutoff(race_data: pd.DataFrame, feature: str):
    """
    Uses linear interpolation to estimate the value of a specified feature at
    the end of the track section (after turns 1 and 2, at lap distance END_LAP_DIST).

    Args:
        race_data: a dataframe of race data in the 2022/2023 format.
        feature: the exact column name of the feature to interpolate.
    Returns:
        The original dataframe with an additional column for the interpolated
        feature value at the track section cutoff.
    """
    df = compute_interpolation_ratios(race_data, feature)
    df[f'{feature}_at_cutoff'] = (1-df['c']) * df[f'{feature}_before'] + df['c'] * df[f'{feature}_after']
    feature_df = df.drop(columns=['x_before', 'y_before', 'x_after', 'y_after', 'c'])
    return feature_df


def add_interpolated_column(race_data: pd.DataFrame, feature: str):
    """
    Adds a new column to the given dataframe containing the value of the specified
    feature at the end of the track section, estimated by linear interpolation.

    Args:
        race_data: a dataframe of race data in the 2022/2023 format.
        feature: the exact column name of the feature to interpolate.
    Returns:
        The original dataframe with an additional column for the interpolated
        feature value at the track section cutoff.
    """
    feature_df = compute_feature_at_section_cutoff(race_data, feature)
    feature_df_for_join = feature_df[['SESSION_IDENTIFIER', 'LAP_NUM', f'{feature}_at_cutoff']]
    race_data_with_feature = pd.merge(race_data, feature_df_for_join, on=['SESSION_IDENTIFIER', 'LAP_NUM'])
    return race_data_with_feature
