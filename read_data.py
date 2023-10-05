import pandas as pd
import numpy as np

def read_data(data_dir):
    expert_data = pd.read_csv(data_dir, header=None)
    state = expert_data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ') for item in state])  # 将state变为(?, 14)的格式，一行代表一个state
    action = expert_data.iloc[1].to_numpy()
    npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
    return npstate, npaction
