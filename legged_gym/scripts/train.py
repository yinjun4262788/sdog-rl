import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
import torch
import pprint
import json
import urllib.request

webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/997a83f1-4ebd-4c6e-83d4-a92ef742526f"

def send_feishu_notification(webhook_url, title, content):
    """
    发送飞书机器人消息
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "msg_type": "text",
        "content": {
            "text": f"{title}\n{content}"
        }
    }
    try:
        req = urllib.request.Request(url=webhook_url, headers=headers, data=json.dumps(data).encode("utf-8"))
        with urllib.request.urlopen(req) as response:
            print(f"Feishu notification sent: {response.status}")
    except Exception as e:
        print(f"Failed to send Feishu notification: {e}")

def save_config_to_file(env_cfg, train_cfg, log_dir, filename="this_config.txt", change_desc=None):
    """
    保存环境和训练配置到文件
    """
    if log_dir is None:
        print("Warning: log_dir is None, cannot save config file.")
        return

    # 如果目录不存在，则创建
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        except Exception as e:
            print(f"Error creating log directory {log_dir}: {e}")
            return

    filepath = os.path.join(log_dir, filename)
    
    try:
        # 转换为字典
        env_cfg_dict = class_to_dict(env_cfg)
        train_cfg_dict = class_to_dict(train_cfg)
        
        # 显式指定 utf-8 编码
        with open(filepath, 'w', encoding='utf-8') as f:
            if change_desc:
                f.write(f"Change Description: {change_desc}\n\n")
            f.write("===== Environment Config =====\n")
            pprint.pprint(env_cfg_dict, stream=f, indent=2)
            f.write("\n\n")
            f.write("===== Train Config =====\n")
            pprint.pprint(train_cfg_dict, stream=f, indent=2)
            
        print(f"Configuration saved to: {filepath}")
    except Exception as e:
        print(f"Error saving config to file: {e}")

def train(args):
    print("开始初始化环境...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 继续执行训练代码
    print("初始化PPO训练器...")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # 保存配置到文件
    save_config_to_file(env_cfg, train_cfg, ppo_runner.log_dir, change_desc=args.change_desc)
    
    print("开始训练...")
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    # 训练结束后发送飞书通知
    msg_content = f"任务名称: {args.task}\n日志路径: {ppo_runner.log_dir}"
    if args.change_desc:
        msg_content += f"\n修改描述: {args.change_desc}"
    
    send_feishu_notification(webhook_url, "【训练完成通知】", msg_content)

if __name__ == '__main__':
    args = get_args()
    train(args)