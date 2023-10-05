import dogfight_client as df
import time
from DemoEnvAi_new import *
import pickle
import numpy as np
import csv

df.connect("10.243.58.131", 50888) # 连接环境
time.sleep(2) # 等待连接
df.disable_log() # 关闭日志？
render = False
df.set_renderless_mode(render) # 开启渲染模式
df.set_client_update_mode(True) # 确保数据传输正确
planes = df.get_planes_list()

plane_id = planes[0]
oppo_id = planes[3]

env = DemoEnv()
env.Plane_ID_ally = plane_id
env.Plane_ID_oppo = oppo_id

state = env.reset()
df.update_scene()
state_dim = state.shape[0]
action_dim = 4
state_list = []
action_list = []
temp_state_list = [] # 通过临时state-action删去无效轨迹
temp_action_list = []

df.activate_IA(plane_id)

episode = 0
step = 0
episode_step = 0
health = env._get_health()
invalid_data = 0
while episode <= 0:
    while health > 0:
        time.sleep(1/20)
        df.update_scene()
        
        state = env._get_observation()
        temp_state_list.append(state) 
        action = env._get_action()
        temp_action_list.append(action)
        health = env._get_health()
        step += 1
        episode_step += 1
        if episode_step > 4000:
            invalid_data += 1
            step -= episode_step
            break
    if episode_step <= 4000:
        state_list.extend(temp_state_list)
        action_list.extend(temp_action_list)
    print(f"episode = {episode}  step = {episode_step}  ")
    env.reset()
    df.activate_IA(plane_id)
    health = env._get_health()
    episode_step = 0
    episode += 1

print("episode", episode, " step", step, " state dim", state_dim, " invalid date", invalid_data)

action_array = np.array(action_list)
state_array = np.array(state_list)
data = [state_array, action_array]

filename = 'expert_data_ai.csv'

# with open(filename, 'wb') as file: # 打开pickle文件
#     pickle.dump(data, file) # 写入pickle文件

with open(filename, 'w', newline='') as file:  # 打开CSV文件，注意要指定newline=''以避免空行
    writer = csv.writer(file)
    writer.writerows(data)  # 将数据写入CSV文件

df.set_client_update_mode(False)
df.disconnect()