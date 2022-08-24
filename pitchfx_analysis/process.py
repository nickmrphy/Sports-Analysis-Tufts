import pybaseball
import numpy as np
import pandas as pdi


def pre_process(data):
  pitch_arr = ['Cutter', '4-Seam Fastball', 'Sinker', 'Slider', 'Changeup', \
            'Knuckle Curve', 'Curveball', 'Split-Finger', 'Pitch Out', \
            'Eephus', 'Knuckleball', 'Forkball']
  type_arr = ['ball', 'blocked_ball', 'called_strike', 'foul', 'foul_bunt',
       'foul_tip', 'hit_by_pitch', 'hit_into_play', 'missed_bunt',
       'pitchout', 'swinging_strike', 'swinging_strike_blocked', 'bunt_foul_tip']
  stand_arr = ['L', 'R']
  of_ass_arr = ['Standard', 'Strategic']
  if_ass_arr = ['Standard', 'Strategic', 'Infield shift']
  d = data[data['pitch_name'].isin(pitch_arr)]
  d = mini_analysis(d)
  df = translate_pitch_type(d, pitch_arr)
  d = translate_outcome(df, type_arr)
  df = translate_p_throws(d, stand_arr)
  d = translate_stance(df, stand_arr)
  df = translate_runners(d)
  return df


def translate_pitch_type(df, pitch_arr):
  offspeed_col = []
  fastball_arr = ['4-Seam Fastball', 'Split-Finger']
  offspeed_arr = ['Cutter', 'Sinker', 'Slider', 'Changeup', \
            'Knuckle Curve', 'Curveball', 'Pitch Out', 
            'Eephus', 'Knuckleball', 'Forkball']
  for index, row in enumerate(df['pitch_name']):
    if row in fastball_arr:
      offspeed_col.append(0)
    else:
      offspeed_col.append(1)
  help_arr = np.arange(0,len(pitch_arr))
  df['pitch_name'].replace(pitch_arr,
                        help_arr, inplace=True)
  df.insert(loc=22, column='off_speed', value=offspeed_col)
  return df



def translate_outcome(df, type_arr):
  help_arr = np.arange(0, len(type_arr))
  df['description'].replace(type_arr,
                    help_arr, inplace=True)
  return df

def translate_stance(df, type_arr):
  help_arr = np.arange(0, len(type_arr))
  df['stand'].replace(type_arr,
                    help_arr, inplace=True)
  return df

def translate_p_throws(df, type_arr):
  help_arr = np.arange(0, len(type_arr))
  df['p_throws'].replace(type_arr,
                    help_arr, inplace=True)
  return df

def translate_runners(df):
  df['on_1b'] = df['on_1b'].fillna(0)
  df['on_2b'] = df['on_2b'].fillna(0)
  df['on_3b'] = df['on_3b'].fillna(0)
  on_1b_col = []
  on_2b_col = []
  on_3b_col = []
  for index, row in df.iterrows():
    if row['on_1b'] != 0:
      on_1b_col.append(1)
    else:
      on_1b_col.append(0)
    if row['on_2b'] != 0:
      on_2b_col.append(1)
    else:
      on_2b_col.append(0)
    if row['on_3b'] != 0:
      on_3b_col.append(1)
    else:
      on_3b_col.append(0)
  df.insert(loc=23, column='on_1b_bin', value=on_1b_col)
  df.insert(loc=24, column='on_2b_bin', value=on_2b_col)
  df.insert(loc=25, column='on_3b_bin', value=on_3b_col)
  df.drop(labels=['on_1b', 'on_2b', 'on_3b'], axis = 1, inplace=True)
  return df

def translate_alignments(df, type_arr1, type_arr2):
  help_arr1 = np.arange(0, len(type_arr1))
  help_arr2 = np.arange(0, len(type_arr2))
  help_arr3 = np.zeros()
  df['if_fielding_alignment'].replace(type_arr1,
                    help_arr1, inplace=True)
  df['if_fielding_alignment'].replace()
  df['of_fielding_alignment'].replace(type_arr2,
                    help_arr2, inplace=True)
  return df




def lookup_player_by_name(first, last):
  f = first.lower()
  l = last.lower()
  res = pybaseball.playerid_lookup(l, f)['key_mlbam']
  print(res)
  return int(res)


def lookup_pitcher_stats_by_id(id, start_dt, end_dt):
  res = pybaseball.statcast_pitcher(start_dt, end_dt, id)
  return res


def lookup_pitcher_statcast_by_name(name, start_dt, end_dt):
  first = name[0]
  last = name[1]
  return lookup_pitcher_stats_by_id(lookup_player_by_name(first, last), start_dt, end_dt)


def lookup_hitter_statcast_by_name(name, start_dt, end_dt):
  first = name[0]
  last = name[1]
  return lookup_pitcher_stats_by_id(lookup_player_by_name(first, last), start_dt, end_dt)



def mini_analysis(df):
  cols_of_interest = ['description', 'stand', 'p_throws', 
       'type', 'balls', 'strikes', 'on_3b', 'on_2b', 'on_1b',
       'outs_when_up', 'inning', 'game_pk', 'pitcher.1',
       'at_bat_number', 'pitch_number', 'pitch_name',
       'home_score', 'away_score', 'bat_score', 'fld_score', 
       'if_fielding_alignment', 'of_fielding_alignment',
       'delta_home_win_exp', 'delta_run_exp']
  data = df[cols_of_interest]
  return data


def pre_process_pitcher_data(data):
    pitch_arr = ['Cutter', '4-Seam Fastball', 'Sinker', 'Slider', 'Changeup', \
            'Knuckle Curve', 'Curveball', 'Split-Finger', 'Pitch Out', \
            'Eephus', 'Knuckleball', 'Forkball']
    type_arr = ['ball', 'blocked_ball', 'called_strike', 'foul', 'foul_bunt',\
       'foul_tip', 'hit_by_pitch', 'hit_into_play', 'missed_bunt',\
       'pitchout', 'swinging_strike', 'swinging_strike_blocked', 'bunt_foul_tip']
    stand_arr = ['L', 'R']
    of_ass_arr = ['Standard', 'Strategic']
    if_ass_arr = ['Standard', 'Strategic', 'Infield shift']
    d = data[data['pitch_name'].isin(pitch_arr)]
    df = translate_pitch_type(d, pitch_arr)
    d = translate_outcome(df, type_arr)
    df = translate_p_throws(d, stand_arr)
    d = translate_stance(df, stand_arr)
    df = translate_runners(d)
    return df

def get_pitcher_data(df, start_dt, end_dt):
    p = pre_process_pitcher_data(df)
    pitcher = p['pitcher']
    pitchers = {}
    final = {}
    pit = unique(pitcher)
    for i in pit:
      final[i] = calc_new_features_from_pitcher_data(p[p['pitcher'] == i])
    res = pd.DataFrame.from_dict(final, 'index')
    res.index.names = ['pitcher']
    res2 = pd.merge(res, p, on='pitcher')
    return df, res2

def calc_new_features_from_pitcher_data(df):
    tendencies = calc_base_pitch_tendencies(df)
    return tendencies

def calc_base_pitch_tendencies(df):
  res = {}
  x = df['description']
  y = x.value_counts()
  tot = y.sum()
  against = np.arange(0,12)
  for i in against:
    if i in y:
      res[str(i) + '_prob'] = y[i] / tot
    else:
      res[str(i) + '_prob'] = 0.0
  return res

def calc_pitch_successes(df):
  res = []
  return res



def load_data_wo_tendencies(new_data):
  y_cols = ['description', 'off_speed']
  y = new_data[y_cols[1]]
  x_cols = ['stand', 'p_throws', 'balls', 'strikes', 'outs_when_up', 'inning', 
        'at_bat_number', 'pitch_number', 'bat_score',
       'fld_score', 'on_1b_bin', 'on_2b_bin', 'on_3b_bin']
  x = new_data[x_cols]
  newer_data = new_data[x_cols + y_cols]
  return newer_data


def load_tendencies(new_data):
  y_cols = ['description', 'off_speed']
  y = pitcher_data_added[y_cols[1]]
  x_cols = ['stand', 'p_throws', 'balls', 'strikes', 'outs_when_up', 'inning', 
        'at_bat_number', 'pitch_number', 'bat_score',
       'fld_score', 'on_1b_bin', 'on_2b_bin', 'on_3b_bin', '0_prob', '1_prob', 
       '2_prob', '3_prob', '4_prob', '5_prob', '6_prob', '7_prob', '8_prob', 
       '9_prob', '10_prob', '11_prob']
  x = pitcher_data_added[x_cols]
  newer_data = pitcher_data_added[x_cols + y_cols]
  return newer_data


def print_corr_map(data):
  plt.figure(figsize=(32, 10))
  # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
  # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
  heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
  # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
  heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)