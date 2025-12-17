import os
import sys
import yaml
import subprocess
import argparse

def run_training(config_path):
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # 读取 YAML 配置
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return

    # 构建命令
    cmd = ["nohup", "python", train_script]
    
    # 添加参数
    if "task" in config:
        cmd.append(f"--task={config['task']}")
    
    if config.get("headless", False):
        cmd.append("--headless")
        
    if "num_envs" in config:
        cmd.append(f"--num_envs={config['num_envs']}")
        
    if "sim_device" in config:
        cmd.append(f"--sim_device={config['sim_device']}")
        
    if "rl_device" in config:
        cmd.append(f"--rl_device={config['rl_device']}")
        
    if "max_iterations" in config:
        cmd.append(f"--max_iterations={config['max_iterations']}")
        
    if "change_desc" in config:
        # 注意：subprocess.Popen 会自动处理参数中的空格，不需要手动加引号
        cmd.append(f"--change_desc={config['change_desc']}")

    # 其他可能的参数
    optional_args = ["resume", "experiment_name", "run_name", "seed", "load_run", "checkpoint"]
    for arg in optional_args:
        if arg in config:
            if isinstance(config[arg], bool):
                if config[arg]:
                    cmd.append(f"--{arg}")
            else:
                cmd.append(f"--{arg}={config[arg]}")

    print("Executing command:")
    print(" ".join(cmd) + " &")

    # 执行命令
    # 使用 subprocess.Popen 启动进程，类似于 nohup ... &
    with open("nohup.out", "a") as outfile:
        subprocess.Popen(cmd, stdout=outfile, stderr=subprocess.STDOUT)
    
    print("Training started in background. Logs are being written to nohup.out")

if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.yaml")
    
    parser = argparse.ArgumentParser(description="Run training with YAML config")
    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    
    run_training(args.config)