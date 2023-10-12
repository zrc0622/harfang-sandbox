#IMPORTS
from NeuralNetwork import Agent as Agent
from ROTNeuralNetwork2 import Agent as ROTAgent
from read_data import read_data
from ReplayMemory  import *
import numpy as np
import time
import sys
import math
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from HarfangEnv_GYM import *
import dogfight_client as df

import datetime
import os
from pathlib import Path

from plot import draw_dif, draw_pos

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log1.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

print(torch.cuda.is_available())

df.connect("10.243.58.131", 11111) #TODO:Change IP and PORT values

start = time.time() #STARTING TIME
df.disable_log()

# PARAMETERS
trainingEpisodes = 6000
validationEpisodes = 50 # 100
explorationEpisodes = 200 # 200

Test = False
if Test:
    render = False
else:
    render = True
    
df.set_renderless_mode(render)
df.set_client_update_mode(True)

bufferSize = (10**6)
gamma = 0.99
criticLR = 1e-4
actorLR = 1e-4
tau = 0.005
checkpointRate = 50 # 100
highScore = -math.inf
successRate = -math.inf
batchSize = 128
maxStep = 5000
validatStep = 5000
hiddenLayer1 = 256
hiddenLayer2 = 512
stateDim = 14 # gai
actionDim = 4 # gai
useLayerNorm = True
bc_weight = 1 # rot

data_dir = 'C:/Users/zuo/Desktop/code/harfang/mine/harfang-sandbox/expert_data_ai.csv'
expert_states, expert_actions = read_data(data_dir)
print(expert_states.shape)
print(expert_actions.shape)

name = "Harfang_GYM"


#INITIALIZATION
env = HarfangEnv()

agent = 'rot'

if agent == 'rot':
    agent = ROTAgent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name, expert_states, expert_actions, bc_weight)
elif agent == 'TD3':
    agent = Agent(actorLR, criticLR, stateDim, actionDim, hiddenLayer1, hiddenLayer2, tau, gamma, bufferSize, batchSize, useLayerNorm, name)

if not Test:
    start_time = datetime.datetime.now()
    dir = Path.cwd() # 获取工作区路径
    log_dir = str(dir) + "\\" + "new_runs\\" + str(start_time.year)+'_'+str(start_time.month)+'_'+str(start_time.day)+'_'+str(start_time.hour)+'_'+str(start_time.minute) # tensorboard文件夹路径
    plot_dir = log_dir + "\\" + "plot"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    save_parameters_to_txt(log_dir=log_dir,bufferSize=bufferSize,criticLR=criticLR,actorLR=actorLR,batchSize=batchSize,maxStep=maxStep,validatStep=validatStep,hiddenLayer1=hiddenLayer1,hiddenLayer2=hiddenLayer2)
    env.save_parameters_to_txt(log_dir)

    writer = SummaryWriter(log_dir)
arttir = 1
# agent.loadCheckpoints(f"Agent0_") # 使用未添加导弹的结果进行训练
# agent.loadCheckpoints(f"Agent7_score-1059.7389121967913") # 使用未添加导弹的结果进行训练

if not Test:
    # RANDOM EXPLORATION
    print("Exploration Started")
    for episode in range(explorationEpisodes):
        state = env.reset()
        done = False
        for step in range(maxStep):
            if not done:
                action = env.action_space.sample()                

                n_state,reward,done, info, stepsuccess = env.step(action)
                # print(n_state)
                if step is maxStep-1:
                    done = True
                agent.store(state,action,n_state,reward,done,stepsuccess)
                state=n_state

                if done:
                    break
        sys.stdout.write("\rExploration Completed: %.2f%%" % ((episode+1)/explorationEpisodes*100))
    sys.stdout.write("\n")

    print("Training Started")
    scores = []
    trainsuccess = 0
    for episode in range(trainingEpisodes):
        state = env.reset()
        totalReward = 0
        done = False
        fire = False
        bc_weight_now = bc_weight - episode/1000
        if bc_weight_now <= 0:
            bc_weight_now = 0
        for step in range(maxStep):
            if not done:
                action = agent.chooseAction(state)
                n_state,reward,done, info, stepsuccess = env.step(action)

                if step is maxStep - 1:
                    break

                agent.store(state, action, n_state, reward, done, stepsuccess) # n_state 为下一个状态
                state = n_state
                totalReward += reward

                if agent.buffer.fullEnough(agent.batchSize):
                    critic_loss, actor_loss, bc_loss, rl_loss, bc_fire_loss = agent.learn(bc_weight_now)
                    writer.add_scalar('Loss/Critic_Loss', critic_loss, step + episode * maxStep)
                    writer.add_scalar('Loss/Actor_Loss', actor_loss, step + episode * maxStep)
                    writer.add_scalar('Loss/BC_Loss', bc_loss, step + episode * maxStep)
                    writer.add_scalar('Loss/RL_Loss', rl_loss, step + episode * maxStep)     
                    writer.add_scalar('Loss/BC_Fire_Loss', bc_fire_loss, step + episode * maxStep)
                    
            elif done:
                if 500 < env.Plane_Irtifa < 10000: # 改
                # if env.Ally_target_locked == True:
                    fire = True
                    trainsuccess = trainsuccess + 1
                break
               
        scores.append(totalReward)
        writer.add_scalar('Training/Episode Reward', totalReward, episode)
        writer.add_scalar('Training/Last 100 Average Reward', np.mean(scores[-100:]), episode)

        if (((episode + 1) % checkpointRate) == 0):
            writer.add_scalar('Training/Train success rate', trainsuccess/checkpointRate, episode)
            trainsuccess = 0
        
        
        now = time.time()
        seconds = int((now - start) % 60)
        minutes = int(((now - start) // 60) % 60)
        hours = int((now - start) // 3600)
        print('Episode: ', episode+1, ' Completed: %r' % done,' Success: %r' % fire, \
            ' FinalReward: %.2f' % totalReward, \
            ' Last100AverageReward: %.2f' % np.mean(scores[-100:]), \
            'RunTime: ', hours, ':',minutes,':', seconds)
            
        #VALIDATION
        if (((episode + 1) % checkpointRate) == 0):
            success = 0
            valScores = []
            dif = []
            self_pos = []
            oppo_pos = []
            for e in range(validationEpisodes):
                state = env.reset()
                totalReward = 0
                done = False
                for step in range(validatStep):
                    if not done:
                        action = agent.chooseActionNoNoise(state)
                        n_state, reward, done, info = env.step(action)
                        state = n_state
                        totalReward += reward

                        if e == validationEpisodes - 1:
                            dif.append(env.loc_diff)
                            self_pos.append(env.get_pos())
                            oppo_pos.append(env.get_oppo_pos())

                        if step is validatStep - 1:
                            break

                    elif done:
                        if 500 < env.Plane_Irtifa < 10000: # 改
                        # if env.Ally_target_locked == True:
                            success += 1
                        break

                valScores.append(totalReward)

            if mean(valScores) > highScore or success/validationEpisodes > successRate or arttir%10 == 0:
                if mean(valScores) > highScore: # 总奖励分数
                    highScore = mean(valScores)
                    agent.saveCheckpoints("Agent{}_score{}".format(arttir, highScore))
                    draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                    draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir) 

                elif success / validationEpisodes > successRate: # 追逐成功率
                    successRate = success / validationEpisodes
                    agent.saveCheckpoints("Agent{}_successRate{}".format(arttir, successRate))
                    draw_dif(f'dif_{arttir}.pdf', dif, plot_dir)
                    draw_pos(f'pos_{arttir}.pdf', self_pos, oppo_pos, plot_dir)
        
            arttir += 1

            print('Validation Episode: ', (episode//checkpointRate)+1, ' Average Reward:', mean(valScores), ' Success Rate:', success / validationEpisodes)
            writer.add_scalar('Validation/Avg Reward', mean(valScores), episode)
            writer.add_scalar('Validation/Success Rate', success/validationEpisodes, episode)
else:
    success = 0
    validationEpisodes = 1000
    for e in range(validationEpisodes):
        state = env.reset()
        totalReward = 0
        done = False
        print('before state: ', state)
        for step in range(validatStep):
            if not done:
                action = agent.chooseActionNoNoise(state)
                n_state,reward,done, info, iffire, beforeaction, afteraction, locked, reward   = env.step_test(action)
                
                if action[3]>0:
                    print(step)
                    print('reward:', reward)
                    print('action: ', action)
                    print('next state: ', n_state)
                    print('before missile: ' , beforeaction, '  if fire: ', iffire, '   after missile: ', afteraction, '    locked', locked)
                    print("+"*15)
                    if locked == True:
                        done = True
                    else:
                        break
                if step is validatStep - 1:
                    print(totalReward)
                    break

                state = n_state
                totalReward += reward
            elif done:
                if 500 < env.Plane_Irtifa < 10000: # 改
                    success += 1
                    print(success)
                break

        # print('Test  Reward:', totalReward)
    print('Success Ratio:', success / validationEpisodes)
