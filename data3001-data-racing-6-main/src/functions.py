################################################################################
# Functions
#
# Add any useful functions for this project here and import them into your own
# notebooks/scripts!
#
# Usage example (in your scripts):
#   from functions import load_track_section
#   track = load_track_section()
#
################################################################################

STARTPOS_X, STARTPOS_Y, STARTPOS_Z = 109.9433441, 467.9519043, 3.136914492
ENDPOS_X, ENDPOS_Y, ENDPOS_Z = 544.483276, -113.066963, 3.143606

LEFT_TRACK_X_MIN, LEFT_TRACK_Y_MAX = 112.62434, 470.734971
LEFT_TRACK_X_MAX, LEFT_TRACK_Y_MIN = 550.477262, -107.62354
RIGHT_TRACK_X_MIN, RIGHT_TRACK_Y_MAX = 103.154164, 460.917389
RIGHT_TRACK_X_MAX, RIGHT_TRACK_Y_MIN = 543.402143, -114.033838

RACE_DATA_FILENAME_2022 = 'F1Sim Data 2022.csv'
RACE_DATA_FILENAME_2023 = 'f1sim-data-2023.csv'
RACE_DATA_FILENAME_2024 = 'F124 Data Export UNSW csv.csv'
TURNS_FILENAME = 'f1sim-ref-turns.csv'

CAR_WIDTH_BUFFER = 0.54

INCLUDE_COLS = [
    'SESSION_IDENTIFIER',
    'LAP_NUM',
    'CURRENT_LAP_TIME_MS',
    'LAP_TIME_MS',
    'LAP_DISTANCE',
    'WORLDPOSX',
    'WORLDPOSY',
    'SPEED_KPH',
    'THROTTLE',
    'STEERING',
    'BRAKE',
    'GEAR',
    'ENGINE_RPM'
]

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_start():
    """
    Returns the XYZ position for the start of the track section (turns 1 and 2)
    as a pandas DataFrame with one row. (See Tom's notebook for details on how
    the start point was found)

    Returns:
        A pandas DataFrame containing the XYZ position for the start of the
        track section.
    """
    return pd.DataFrame(
        columns=['WORLDPOSX','WORLDPOSY','WORLDPOSZ'],
        data=np.array([[STARTPOS_X, STARTPOS_Y, STARTPOS_Z]])
    )

def get_end():
    """
    Returns the XYZ position for the end of the track section (turns 1 and 2)
    as a pandas DataFrame with one row. (See Kento's notebook for details on how
    the end point was found)

    Returns:
        A pandas DataFrame containing the XYZ position for the end of the
        track section.
    """
    return pd.DataFrame(
        columns=['WORLDPOSX','WORLDPOSY','WORLDPOSZ'],
        data=np.array([[ENDPOS_X, ENDPOS_Y, ENDPOS_Z]])
    )

def load_entire_track(directory):
    """
    Loads the left and right points of the whole track as a single DataFrame.
    Includes filtering for removing start point and second lap of data.

    Args:
        directory (str): The directory containing the track data.
    Returns:
        A pandas DataFrame containing the track left/right data.
    """
    left = pd.read_csv(os.path.join(directory, 'f1sim-ref-left.csv'))
    left = left[(left.FRAME < 6100) & (left.FRAME > 56)]
    right = pd.read_csv(os.path.join(directory, 'f1sim-ref-right.csv'))
    right = right[(right.FRAME < 6395) & (right.FRAME > 140)]
    track = pd.concat([left, right], axis=0)
    return track

def load_track_section(directory):
    """
    Loads the left and right points of the track section (turns 1 and 2) as a
    single pandas DataFrame.

    Args:
        directory (str): The directory containing the track section data.
    Returns:
        A pandas DataFrame containing the track left/right data, limited to
        turns 1 and 2.
    """
    return filter_for_track_section(load_entire_track(directory))

def filter_track_side(
        trackSide: pd.DataFrame, side: str,
        X_colname='WORLDPOSX', Y_colname='WORLDPOSY'
    ):
    """
    Given a DataFrame containing positional data for either the left or 
    right side of the track, filter it to remove anything before the 
    beginning of the track.

    Args:
        trackSide (pd.Dataframe): The DataFrame containing the positional data. This
            must contain columns 'WORLDPOSX' and 'WORLDPOSY'.
        side (str): The side of the track given, either 'LEFT' or 'RIGHT'
    Returns: 
        A DataFrame containing only the points after the start line.
    """
    if side == 'LEFT':
        X_min, X_max = LEFT_TRACK_X_MIN, LEFT_TRACK_X_MAX
        Y_min, Y_max = LEFT_TRACK_Y_MIN, LEFT_TRACK_Y_MAX
    elif side == 'RIGHT':
        X_min, X_max = RIGHT_TRACK_X_MIN, RIGHT_TRACK_X_MAX
        Y_min, Y_max = RIGHT_TRACK_Y_MIN, RIGHT_TRACK_Y_MAX
    else:
        return None
    trackSide = trackSide[trackSide['REFTYPE'] == side]
    trackSide = trackSide[
        (trackSide[X_colname] >= X_min) &
        (trackSide[X_colname] <= X_max) &
        (trackSide[Y_colname] >= Y_min) &
        (trackSide[Y_colname] <= Y_max)
    ]
    return trackSide


def get_apex_coords(directory):
    turns = load_turn_data(directory)
    apex1 = (turns['APEX_X1'][0], turns['APEX_Y1'][0])
    apex2 = (turns['APEX_X1'][1], turns['APEX_Y1'][1])
    return apex1, apex2


def filter_for_track_section(
        data, X_colname='WORLDPOSX', Y_colname='WORLDPOSY'
    ):
    """
    Given a DataFrame containing positional data, filters it to only include
    points that are within our track section (turns 1 and 2).

    Args:
        data (pd.DataFrame): The DataFrame containing the positional data. This
            must contain columns 'WORLDPOSX', 'WORLDPOSY', and 'LAP_DISTANCE.
    Returns:
        A DataFrame containing only the points within our track section.
    """
    start = get_start()
    end = get_end()

    # REFTYPE column only exists for track data
    if 'REFTYPE' in data.columns:
        left = filter_track_side(data, 'LEFT')
        right = filter_track_side(data, 'RIGHT')
        data = pd.concat([left, right], axis=0)
        return data

    return data[
        (data[X_colname] >= start['WORLDPOSX'].loc[0]) &
        (data[X_colname] <= end['WORLDPOSX'].loc[0]) &
        (data[Y_colname] <= start['WORLDPOSY'].loc[0]) &
        (data[Y_colname] >= end['WORLDPOSY'].loc[0]) &
        (data['LAP_DISTANCE'] < 1000) # bandaid fix for some end of lap data getting through
    ]

def load_turn_data(directory):
    """
    Loads the turn data as a pandas DataFrame.

    Args:
        directory (str): The directory containing the turn data (CSV).
    Returns:
        A pandas DataFrame containing the apex points of each turn in the track
        section.
    """
    return pd.read_csv(os.path.join(directory, TURNS_FILENAME))


def plot_track_2d(track, turns=None):
    """
    Creates a 2D scatter plot of the track from the given pandas Dataframe. Note
    that the resulting plot specifies xlim and ylim, so new data plotted on top
    may not be visible initially.

    Args:
        track (pd.DataFrame): The track data to plot. The DataFrame must contain
            columns 'WORLDPOSX' and 'WORLDPOSY'.
        turns (pd.DataFrame): optional DataFrame containing the apex points of
            each turn. Must contain columns 'TURN', 'APEX_X1' and 'APEX_Y1'.
    """
    start = get_start()
    fig = plt.figure()
    plt.scatter(data=track, x='WORLDPOSX', y='WORLDPOSY', s=0.1)
    plt.scatter(data=start, x='WORLDPOSX', y='WORLDPOSY', s=20)
    plt.annotate(
        'Start',
        (start['WORLDPOSX'].loc[0],start['WORLDPOSY'].loc[0]),
        textcoords="offset points",
        xytext=(5,10),
        ha='left',
        va='top'
    )
    # Adding turn numbers to the plot (if 'turns' is provided)
    if turns is not None:
        plt.scatter(data=turns, x='APEX_X1', y='APEX_Y1', c='red', s=20)
        for i in range(turns.shape[0]):
            plt.annotate(
                turns['TURN'][i],
                (turns['APEX_X1'][i], turns['APEX_Y1'][i]),
                textcoords="offset points",
                xytext=(5,12),
                ha='left',
                va='top'
            )
    plt.xlim([track['WORLDPOSX'].min() - 50, track['WORLDPOSX'].max() + 50])
    plt.ylim([track['WORLDPOSY'].min() - 50, track['WORLDPOSY'].max() + 50])


def plot_track_3d(track):
    """
    Creates a 3D scatter plot of the track from the given pandas Dataframe.

    Args:
        track (pd.DataFrame): The track data to plot. The DataFrame must contain
            columns 'WORLDPOSX', 'WORLDPOSY', and 'WORLDPOSZ'.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(zlim=[0,10])
    ax.scatter(track['WORLDPOSX'], track['WORLDPOSY'], track['WORLDPOSZ'], s=0.1)


def drop_unused_cols(df):
    """
    Removes all columns from the DataFrame that are not included in the
    INCLUDE_COLS list (see top of file).
    """
    return df[INCLUDE_COLS]


def load_race_data_2022(directory, filter_for_section=True, remove_unused_cols=False):
    """
    Loads the 2022 simulator data for the track section (turns 1 and 2) as a
    pandas DataFrame.

    Args:
        directory (str): The directory containing the simulator data.
    Returns:
        A pandas DataFrame containing the simulator data for the track section.
    """
    df = pd.read_csv(os.path.join(directory, RACE_DATA_FILENAME_2022))
    df_cleaned = filter_laps_with_insufficient_data(df)
    df_cleaned = filter_laps_with_missing_fwd_dir(df_cleaned)
    if remove_unused_cols:
        df_cleaned = drop_unused_cols(df_cleaned)
    if filter_for_section:
        return filter_for_track_section(df_cleaned)
    else:
        return df_cleaned.sort_values(
            by=['SESSION_IDENTIFIER', 'LAP_NUM', 'CURRENT_LAP_TIME_MS']
        )


def load_race_data_2023(directory, filter_for_section=True, remove_unused_cols=False):
    """
    Loads the 2023 simulator data for the track section (turns 1 and 2) as a
    pandas DataFrame.

    Args:
        directory (str): The directory containing the simulator data.
    Returns:
        A pandas DataFrame containing the simulator data for the track section.
    """
    df = pd.read_csv(os.path.join(directory, RACE_DATA_FILENAME_2023))
    df_cleaned = filter_laps_with_insufficient_data(df)
    df_cleaned = filter_laps_with_missing_fwd_dir(df_cleaned)
    if remove_unused_cols:
        df_cleaned = drop_unused_cols(df_cleaned)
    if filter_for_section:
        return filter_for_track_section(df_cleaned)
    else:
        return df_cleaned.sort_values(
            by=['SESSION_IDENTIFIER', 'LAP_NUM', 'CURRENT_LAP_TIME_MS']
        )


def load_race_data_2024(directory, filter_for_section=True, remove_unused_cols=False, include_invalid=False):
    """
    Loads the 2024 simulator data for the track section (turns 1 and 2) as a
    pandas DataFrame. Renames the columns to match the 2022/2023 format.

    Args:
        directory (str): The directory containing the simulator data.
    Returns:
        A pandas DataFrame containing the simulator data for the track section.
    """
    include_cols = [
        'SESSION_GUID',
        'M_CURRENTLAPNUM',
        'M_CURRENTLAPTIMEINMS_1',
        'M_LAPTIMEINMS',
        'M_LAPDISTANCE_1',
        'M_WORLDPOSITIONX_1',
        'M_WORLDPOSITIONY_1',
        'M_SPEED_1',
        'M_THROTTLE_1',
        'M_STEER_1',
        'M_BRAKE_1',
        'M_GEAR_1',
        'M_ENGINERPM_1',
        'M_WORLDFORWARDDIRX_1',
        'M_WORLDFORWARDDIRY_1'
    ]
    if include_invalid:
        include_cols.append('M_LAPINVALID') # exclude because not in 2022/2023

    df = pd.read_csv(os.path.join(directory, RACE_DATA_FILENAME_2024), dtype={77: str})
    df = df[include_cols]
    df = df.rename(columns={
        'SESSION_GUID': 'SESSION_IDENTIFIER',
        'M_CURRENTLAPNUM': 'LAP_NUM',
        'M_CURRENTLAPTIMEINMS_1': 'CURRENT_LAP_TIME_MS',
        'M_LAPTIMEINMS': 'LAP_TIME_MS',
        'M_LAPDISTANCE_1': 'LAP_DISTANCE',
        'M_WORLDPOSITIONX_1': 'WORLDPOSX',
        'M_WORLDPOSITIONY_1': 'WORLDPOSY',
        'M_SPEED_1': 'SPEED_KPH',
        'M_THROTTLE_1': 'THROTTLE',
        'M_STEER_1': 'STEERING',
        'M_BRAKE_1': 'BRAKE',
        'M_GEAR_1' : 'GEAR',
        'M_ENGINERPM_1' : 'ENGINE_RPM',
        'M_WORLDFORWARDDIRX_1' : 'WORLDFORWARDDIRX',
        'M_WORLDFORWARDDIRY_1' : 'WORLDFORWARDDIRY',
    })
    # Handle the one lap with null SESSION_IDENTIFIER (note: this is Max!)
    df['SESSION_IDENTIFIER'] = df['SESSION_IDENTIFIER'].fillna('MAX_VERSTAPPEN')

    if include_invalid:
        df = df.rename(columns={
            'M_LAPINVALID': 'INVALID_LAP'
        })
    if remove_unused_cols:
        df = drop_unused_cols(df)
    if filter_for_section:
        return filter_for_track_section(df)
    else:
        return df.sort_values(
            by=['SESSION_IDENTIFIER', 'LAP_NUM', 'CURRENT_LAP_TIME_MS']
        )


def euclid_distance(pointA: list, pointB: list):
    """
    Given two points returns the euclidean distance between them.

    Args:
        pointA, pointB (list): A list with index 0 as the x-pos and 1 as the y-pos.
    Returns:
        A float of the euclidean distance
    """
    return np.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)


def two_closest_points(pointA: list, listOfPoints:list):
    """
    Algorithm 1 in the helper sheet.
    Given a point, A, and a list of points, returns two closest points to A.

    Args:
        pointA (list): A list with index 0 as the x-pos and 1 as the y-pos.
        listOfPoints (list): A list of points, in the form described above.
    Returns:
        A tuple of two points: the closest point, and the second closest point
    """
    distance = [euclid_distance(pointA, point) for point in listOfPoints]
    minIndex = distance.index(min(distance))
    distance.pop(minIndex)
    secondMinIndex = distance.index(min(distance))
    if secondMinIndex >= minIndex:
        secondMinIndex+=1

    return listOfPoints[minIndex], listOfPoints[secondMinIndex]


def projection_values(pointA: list, pointB1: list, pointB2: list):
    """
    Algorithm 2 in the helper sheet.
    Given a point, A, and two points, B1 and B2, return a dictionary of projection values.
    The dictionary has three keys:
        'd': minimum distance d from A to the line defined by B1 and B2
        'projA': the location of the point on the line, Ap, that is nearest to A
        'c': the ratio of the distance from B1 to B2 that will get you from B1 to Ap

    Args:
        pointA (list): A list with index 0 as the x-pos and 1 as the y-pos.
        pointB1 (list): A list like above that represents one of the points on the line you want to project A onto
        pointB2 (list): A list like above that represents the other point on the line you want to project A onto
    Returns:
        A dictionary of projection values
    """
    c = ((pointA[0] - pointB1[0])*(pointB2[0] - pointB1[0]) + (pointA[1] - pointB1[1])*(pointB2[1] - pointB1[1])) / euclid_distance(pointB1, pointB2)**2
    projA = [pointB1[0] + c*(pointB2[0] - pointB1[0]), pointB1[1] + c*(pointB2[1] - pointB1[1])]
    d = euclid_distance(pointA, projA)
    return {'d': d, 'projA': projA, 'c': c}


def df_to_list(df: pd.DataFrame, rowIndex=0):
    """
    Helper function to convert from a row from a DataFrame into a list, so that 
    it can be used as an input into one of the point functions.

    Args:
        df: A pandas DataFrame
        rowIndex: An integer specfiying which row to convert into a list
    Returns:
        A list of an x and y coordinate
    """
    return df[['WORLDPOSX', 'WORLDPOSY']].iloc[rowIndex].to_list()


def get_proj_on_side(side: pd.DataFrame, point: list):
    """
    Helper function to project point onto side of track.
    Used in finding start and finish line for track

    Args:
        side (pd.DataFrame): A pandas DataFrame corresponding to a track side
        point (list): A point to project onto the side of the track
    Returns:
        A pandas dataFrame of one row corresponding to the point
    """

    trackList = []
    for i in range(len(side)):
        trackList.append(df_to_list(side, i))

    pointB1, pointB2 = two_closest_points(point, trackList)
    projA = projection_values(point, pointB1, pointB2)['projA']
    return pd.DataFrame(np.array([projA]), columns=['WORLDPOSX', 'WORLDPOSY'])


def get_track_sides(track: pd.DataFrame):
    """
    Given the whole track, split into two dataframes corresponding to each side of the track
    """
    return track[track.REFTYPE == 'LEFT'], track[track.REFTYPE == 'RIGHT']


def check_data_validity(df: pd.DataFrame) -> None:
    """
    #TODO may need to change/add to this, haven't fully thought out the logic of it
    Check that a given dataset (in 2022/2023 format) is valid, for columns of interest.
    Note: should be used once data is cleaned as a check as this does not fix/remove errors, just identifies them

    Args:
        df (pd.DataFrame): dataset in 2022/2023 format
    """
    for i, r in df.iterrows():
        assert r['FRAME'] > 0
        assert r['LAP_NUM'] > 0
        assert r['SECTOR'] in [0, 1, 2]
        assert r['LAP_DISTANCE'] >= 0
        assert r['CURRENT_LAP_TIME_MS'] >= 0
        assert r['LAP_TIME_MS'] > 0
        assert r['SPEED_KPH'] >= 0
        assert 0 <= r['THROTTLE'] <= 1
        assert 0 <= r['BRAKE'] <= 1
        assert -1 <= r['STEERING'] <= 1
        assert not np.isnan(r['WORLDPOSX'])
        assert not np.isnan(r['WORLDPOSY'])
        assert not np.isnan(r['WORLDPOSZ'])
        assert not np.isnan(r['WORLDFORWARDDIRX'])
        assert not np.isnan(r['WORLDFORWARDDIRY'])
        assert not np.isnan(r['WORLDFORWARDDIRZ'])
        assert not np.isnan(r['WORLDRIGHTDIRX'])
        assert not np.isnan(r['WORLDRIGHTDIRY'])
        assert not np.isnan(r['WORLDRIGHTDIRZ'])


def number_of_moments_in_lap_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a df in 2022/2023 format, calculates how many 'moments'/rows there are for each race's unique laps. 
    It was discovered that a significant amount (especially in 2022) have very few over the section of interest,
    therefore provide very little useful information and should not be used
    The information returned by this function can help make judgements on what laps to cut

    Args:
        df (pd.DataFrame): dataset in 2022/2023 format

    Returns:
        A dataframe with a row for each unique session/lap, and the number of rows from the original df for each over the track section
    """
    data_in_range = df[df['LAP_DISTANCE'] < 750]
    grouped_data = data_in_range.groupby(['SESSION_IDENTIFIER', 'LAP_NUM']).size().reset_index(name = 'count')
    return grouped_data


def update_data_w_distances(data: pd.DataFrame, directory: str):
    """
    Given a data set and track data, updates the data set to include the columns
    DISTFROMLEFT, DISTFROMRIGHT and USERTRACKWIDTH (defined by DISTFROMLEFT + DISTFROMRIGHT)

    Args:
        data (pd.DataFrame): dataset in 2022/2023 format
        directory (str): string with master path to data directory
    Returns:
        data: dataset in same format as before with DISTFROMLEFT, DISTFROMRIGHT and USERTRACKWIDTH added
    """
    track = load_track_section(directory)
    left, right = get_track_sides(track)
    distFromLeft = []
    distFromRight = []
    leftTrackList = []
    rightTrackList = []
    for i in range(len(left)):
        leftTrackList.append(df_to_list(left, i))
    for i in range(len(right)):
        rightTrackList.append(df_to_list(right, i))

    for index, row in data[['WORLDPOSX', 'WORLDPOSY']].iterrows():
        pointA = [row.WORLDPOSX, row.WORLDPOSY]
        leftPointB1, leftPointB2 = two_closest_points(pointA, leftTrackList)
        rightPointB1, rightPointB2 = two_closest_points(pointA, rightTrackList)
        distFromLeft.append(projection_values(pointA, leftPointB1, leftPointB2)['d'])
        distFromRight.append(projection_values(pointA, rightPointB1, rightPointB2)['d'])

    newData = data.copy()
    newData['DISTFROMLEFT'] = distFromLeft
    newData['DISTFROMRIGHT'] = distFromRight
    newData['USER_TRACKWIDTH'] = newData['DISTFROMLEFT'] + newData['DISTFROMRIGHT']

    return newData


def filter_one_lap(data, session_index):
    '''
    Given all race data, filter for just one lap, where the session_identifier is given as an index
    This may not work to identify precise laps, but more for when we just want one lap to test things.
    '''
    return data[(data.SESSION_IDENTIFIER == data.SESSION_IDENTIFIER.unique()[session_index]) & (data.LAP_NUM == 1)].reset_index(drop=True)


def get_track_width_reference_data(directory):
    '''
    Helper function for update_data_w_width
    Given a directory, loads the 2022 data and the track data and then finds the track width for 3 specific 
    laps in the 2022 data. This data can then be used as a line to interpolate the track wdith on for 
    any specified lap_distance.
    The three session IDs that were used were inspected manually and are three laps where the driver never
    goes off track, and have a large amount of data points.
    '''
    data22 = load_race_data_2022(directory)
    trackWidth = data22[(data22['SESSION_IDENTIFIER'].isin([1.020389177761665e+19, 1.0370910074670805e+19, 1.159612117810798e+19])) & (data22['LAP_NUM'] == 1)]
    trackWidth = update_data_w_distances(trackWidth, directory)
    trackWidth = trackWidth[['LAP_DISTANCE', 'USER_TRACKWIDTH']].sort_values(by='LAP_DISTANCE').reset_index(drop=True)
    trackWidth = trackWidth.rename(columns={'USER_TRACKWIDTH':'TRACKWIDTH'})
    return trackWidth


def update_data_w_width(data: pd.DataFrame, directory: str):
    '''
    Add the TRACKWIDTH column to the dataset. This is based on lap_distance and the trackWidth reference data.
    See the function get_track_width_reference_data for more information.
    '''
    trackWidth = get_track_width_reference_data(directory)
    data['TRACKWIDTH'] =  np.interp(data['LAP_DISTANCE'], trackWidth['LAP_DISTANCE'], trackWidth['TRACKWIDTH'])
    return data


def filter_laps_with_insufficient_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): a dataframe containing race data

    Returns:
        the input dataframe with laps with insufficient data filtered out

    Filters a given dataframe, removing laps with insufficient data or too large gaps between data points
    #TODO choose thresholds for both num points and gaps
    """
    grouped = df.groupby(['SESSION_IDENTIFIER', 'LAP_NUM'])
    filtered = grouped.filter(lambda x: len(x) >= 100)

    def has_large_gap(group):
        sorted_distances = group['LAP_DISTANCE'].sort_values()
        gaps = sorted_distances.diff().dropna()  # Calculate differences between consecutive rows
        return (gaps > 25).any()

    cleaned = filtered.groupby(['SESSION_IDENTIFIER', 'LAP_NUM']).filter(lambda x: not has_large_gap(x))
    return cleaned


def calculate_area(df: pd.DataFrame, start: float, end: float, feature: str) -> float:
    """
    Args:
        df (pd.DataFrame): a dataframe containing a single lap of data
        start (float): the starting lap distance
        end (float): the ending lap distance
        feature (str): the column being calculated on (probably only either throttle or brake)

    Returns:
        the total throttle/brake between these points for this lap

    Finds the 'total' braking/throttle for a given lap between two lap distances

    """
    df = df.sort_values('LAP_DISTANCE')

    if start not in df['LAP_DISTANCE'].values:
        start_throttle = np.interp(start, df['LAP_DISTANCE'], df[feature])
        df_start = pd.DataFrame({'LAP_DISTANCE': [start], feature: [start_throttle]})
        df = pd.concat([df, df_start]).sort_values('LAP_DISTANCE')

    if end not in df['LAP_DISTANCE'].values:
        end_throttle = np.interp(end, df['LAP_DISTANCE'], df[feature])
        df_end = pd.DataFrame({'LAP_DISTANCE': [end], feature: [end_throttle]})
        df = pd.concat([df, df_end]).sort_values('LAP_DISTANCE')

    df_filtered = df[(df['LAP_DISTANCE'] >= start) & (df['LAP_DISTANCE'] <= end)]

    area = np.trapz(df_filtered[feature], df_filtered['LAP_DISTANCE'])

    return area


def create_off_track_flag(df: pd.DataFrame, directory: str):
    """
    Given a df that contains USER_TRACKWIDTH, TRACKWIDTH create a flag that determines if the 
    car is off the track at the specified moment (row).

    Args:
        df (pd.DataFrame): a dataframe containing lap data
        directory (str): a string containing path to datasets

    Returns:
        df (pd.DataFrame): a dataframe updated with OFF_TRACK column
    """
    if 'USER_TRACKWIDTH' not in df.columns:
        df = update_data_w_distances(df, directory)
    if 'TRACKWIDTH' not in df.columns:
        df = update_data_w_width(df, directory)
    df['OFF_TRACK'] = np.where(df['TRACKWIDTH'] + CAR_WIDTH_BUFFER < df['USER_TRACKWIDTH'], 1, 0)
    return df


def create_invalid_lap_flag(df: pd.DataFrame, directory):
    """
    Given a df that contains OFF_TRACK, create a flag that determines if the car
    went off track at any point during the lap.

    Args:
        df (pd.DataFrame): a dataframe containing lap data
        track (pd.DataFrame): a dataframe containing track data
        directory (str): a string containing path to datasets

    Returns:
        df (pd.DataFrame): a dataframe updated with OFF_TRACK column
    """
    if 'OFF_TRACK' not in df.columns:
        df = create_off_track_flag(df, directory)
    df['INVALID_LAP'] = df.groupby(['SESSION_IDENTIFIER', 'LAP_NUM'])['OFF_TRACK'].transform("max")
    return df

def filter_laps_with_missing_fwd_dir(df: pd.DataFrame):
    # Doesn't matter if DIRX or DIRY is used - same laps
    return df[df.WORLDFORWARDDIRX.notna()]