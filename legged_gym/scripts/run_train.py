import os
import sys
import yaml
import subprocess
import argparse
import shlex

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

    play_after_train = bool(config.get("play_after_train", False))
    run_in_background = bool(config.get("run_in_background", True))

    play_script = os.path.join(script_dir, "play.py")

    train_cmd = ["python", train_script]

    if "task" in config:
        train_cmd.append(f"--task={config['task']}")

    if config.get("headless", False):
        train_cmd.append("--headless")

    if "num_envs" in config:
        train_cmd.append(f"--num_envs={config['num_envs']}")

    if "sim_device" in config:
        train_cmd.append(f"--sim_device={config['sim_device']}")

    if "rl_device" in config:
        train_cmd.append(f"--rl_device={config['rl_device']}")

    if "max_iterations" in config:
        train_cmd.append(f"--max_iterations={config['max_iterations']}")

    if "change_desc" in config:
        train_cmd.append(f"--change_desc={config['change_desc']}")

    optional_args = ["resume", "experiment_name", "run_name", "seed", "load_run", "checkpoint"]
    for arg in optional_args:
        if arg in config:
            if isinstance(config[arg], bool):
                if config[arg]:
                    train_cmd.append(f"--{arg}")
            else:
                train_cmd.append(f"--{arg}={config[arg]}")

    play_cmd = None
    if play_after_train:
        play_cmd = ["python", play_script]
        if "task" in config:
            play_cmd.append(f"--task={config['task']}")

    if run_in_background:
        if play_cmd is None:
            cmd = ["nohup"] + train_cmd
        else:
            train_part = " ".join(shlex.quote(x) for x in train_cmd)
            play_part = " ".join(shlex.quote(x) for x in play_cmd)
            chained = f"{train_part}; rc=$?; if [ $rc -eq 0 ]; then {play_part}; fi; exit $rc"
            cmd = ["nohup", "bash", "-lc", chained]

        print("Executing command:")
        print(" ".join(cmd) + " &")
        with open("nohup.out", "a") as outfile:
            subprocess.Popen(cmd, stdout=outfile, stderr=subprocess.STDOUT)
        if play_cmd is None:
            print("Training started in background. Logs are being written to nohup.out")
        else:
            print("Training+play started in background (play runs only if training succeeds). Logs are being written to nohup.out")
        return

    print("Executing command:")
    print(" ".join(train_cmd))
    with open("nohup.out", "a") as outfile:
        res = subprocess.run(train_cmd, stdout=outfile, stderr=subprocess.STDOUT)

    if res.returncode != 0:
        print(f"Training failed with return code: {res.returncode}")
        return

    if play_cmd is None:
        print("Training completed successfully, not playing after training.")
        return

    print("Executing play command:")
    print(" ".join(play_cmd))
    subprocess.run(play_cmd)

if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.yaml")
    
    parser = argparse.ArgumentParser(description="Run training with YAML config")
    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config file")
    args = parser.parse_args()
    
    run_training(args.config)