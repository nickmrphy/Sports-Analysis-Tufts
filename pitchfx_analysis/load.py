import pybaseball
import process

def load(data):
    new_data = process.pre_process(data)
    new_data = process.load_tendencies(new_data)
    y = new_data['description']
    x_cols = ['stand', 'p_throws', 'balls', 'strikes', 'outs_when_up', 'inning', 
        'at_bat_number', 'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score',
       'fld_score', 'off_speed', 'on_1b_bin',
       'on_2b_bin', 'on_3b_bin']
    x = new_data[x_cols]
    return new_data, x, y
