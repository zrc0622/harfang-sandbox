# from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def read_tensorboard_data(tensorboard_path, val_name):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址val_name是需要读取的变量名称"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val


def draw_plt(val, val_name):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name, color='darkorange')
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('reward.pdf', bbox_inches='tight')
    plt.show()


def draw_dif(file_name, dif, dir):
    plt.figure()
    plt.plot(range(len(dif)), dif, color='darkorange')
    plt.xlabel('step')
    plt.ylabel('dif')
    file_name = os.path.join(dir, file_name)
    plt.savefig(file_name, bbox_inches='tight')


def draw_pos(file_name, self_pos, oppo_pos, dir):
    ax = plt.figure().add_subplot(projection='3d')
    self_pos = np.array(self_pos)
    oppo_pos = np.array(oppo_pos)

    num_points = len(self_pos)

    colors = plt.cm.jet(np.linspace(0,1,num_points))

    for i in range(num_points - 1):
        if i != i-2:
            ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=colors[i])
            ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=colors[i])
        else:
            ax.plot(self_pos[i:i+2, 0], self_pos[i:i+2, 2], self_pos[i:i+2, 1], color=colors[i], label='self')
            ax.plot(oppo_pos[i:i+2, 0], oppo_pos[i:i+2, 2], oppo_pos[i:i+2, 1], color=colors[i], label='oppo')
    
    plt.legend()
    file_name = os.path.join(dir, file_name)
    plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":
    tensorboard_path = '/Users/wenyongyan/Downloads/logs/events.out.tfevents.1687995824.DESKTOP-6S5E44I.284.0'
    val_name = 'Training/Episode Reward'
    val = read_tensorboard_data(tensorboard_path, val_name)
    draw_plt(val, val_name)
