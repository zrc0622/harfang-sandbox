# An Imitative Reinforcement Learning Approach for Dogfight

## 模型
行为克隆、TD3、本文提出的方法训练的最佳模型存储在results文件夹下，模型验证结果如下表所示（每个模型的成功率和随机初始化成功率使用5个不同的随机种子运行50个回合得到，±表示标准差；命中率运行10个回合得到；加粗表示最佳结果）
| **模型**                                             | **算法**        | **成功率** | **随机初始化成功率** | **命中率(成功发射次数/发射次数)** |
|:--------------------------------------------------:|:-------------:|:-------:|:------------:|:------------------:|
| OurAdaptive_Agent_successRate100_Critic_Harfang_GYM | Our(adaptive) | **100.0% ± 0.0%**    | **87.6% ± 5.4%**          | 12 / 15          |
| OurLinear_Agent_successRate100_Critic_Harfang_GYM   | Our(linear)   | **100.0% ± 0.0%**    | 86.0% ± 6.0%          | **11 / 11**          |
| TD3_Agent_successRate50_Critic_Harfang_GYM        | TD3           | 49.6% ± 0.9%     | 8.0% ± 3.2%          | 0 / 70           |
| BC_Agent_successRate63_Harfang_GYM                | BC            | 62.8% ± 1.1%     | 22.4% ± 3.6%          | 12 / 13          |


## 本文方法训练得到的策略

单一炮弹场景下的策略

![image](videos/single.gif)

无限炮弹场景下的策略

![image](videos/multi.gif)

## 配置
1. 安装`Harfang3D sandbox`的[Release版本](https://github.com/harfang3d/dogfight-sandbox-hg2/releases/tag/v1.3.0)或[源代码](https://github.com/harfang3d/dogfight-sandbox-hg2)，推荐安装源代码版本，这样可以自行更改环境的端口
2. 安装本代码所需依赖
    ```
    conda env create -f environment.yaml
    ```
## 训练
1. 在`Harfang3D sandbox`下的`source`文件夹，使用以下命令打开`Harfang3D sandbox`，使用`network_port`指定端口号，打开后进入网络模式（NETWORK MODE）
    ```
    python main.py network_port 12345
    ```
2. 在`本代码`文件夹下使用以下命令进行训练（注意修改代码`train_all.py`中的IP号，使用--render开启训练渲染）
    ```
    # 自适应权重NIRL
    python train_all.py --agent ROT --port 12345 --type soft --model_name srot
    ```
    ```
    # 线性权重NIRL
    python train_all.py --agent ROT --port 12345 --type linear --bc_weight 1 --model_name lrot
    ```
    ```
    # 固定权重NIRL
    python train_all.py --agent ROT --port 12345 --type fixed --bc_weight 0.5 --model_name frot
    ```
    ```
    # TD3
    python train_all.py --agent TD3 --port 12345 --model_name td3
    ```
    ```
    # BC
    python train_all.py --agent BC --port 12345 --model_name bc
    ```
    ```
    # SAC
    python train_sac.py --type sac --port 12345 --model_name sac
    ```
    ```
    # E-SAC
    python train_sac.py --type esac --port 12345 --model_name esac
    ```
## 测试
1. 在`Harfang3D sandbox`下的`source`文件夹，使用以下命令打开`Harfang3D sandbox`，使用`network_port`指定端口号，打开后进入网络模式
    ```
    python main.py network_port 12345
    ```
2. 在`本代码`文件夹下使用以下命令进行测试（注意修改代码`train_all.py`中的IP号及导入模型名称（仅需‘xxx_Harfang_GYM之前的名称’），使用--render开启测试渲染）
    ```
    # 在相应训练代码后添加'--test --test_mode 1'即可，其中test mode 1为随机初始化，test mode 2为无限导弹，test mode 3为原始环境测试
    # 以下为一个例子
    python train_all.py --agent ROT --port 12345 --type soft --model_name srot --test --test_mode 1 --seed 1
    ```

