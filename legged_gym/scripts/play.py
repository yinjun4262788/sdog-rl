import os
import shutil
import subprocess
from datetime import datetime

import isaacgym
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry

import numpy as np
import torch


def _camera_vec3(xyz):
    return gymapi.Vec3(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _yaw_from_quat_xyzw(q):
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _rotate_z(v, yaw: float):
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float32)


def _set_record_camera(env, camera_handle, env_id: int, camera_cfg: dict):
    follow = bool(camera_cfg.get("follow", False))
    if follow:
        base_pos = env.root_states[env_id, 0:3].detach().cpu().numpy()
        base_quat = env.root_states[env_id, 3:7].detach().cpu().numpy()
        yaw = _yaw_from_quat_xyzw(base_quat)

        follow_offset = np.array(camera_cfg["follow_offset"], dtype=np.float32)
        lookat_offset = np.array(camera_cfg["lookat_offset"], dtype=np.float32)

        cam_pos = base_pos + _rotate_z(follow_offset, yaw)
        cam_target = base_pos + _rotate_z(lookat_offset, yaw)
    else:
        cam_pos = np.array(camera_cfg["pos"], dtype=np.float32)
        cam_target = np.array(camera_cfg["lookat"], dtype=np.float32)

    env.gym.set_camera_location(
        camera_handle,
        env.envs[env_id],
        _camera_vec3(cam_pos),
        _camera_vec3(cam_target),
    )


def _make_video_from_frames(frames_dir: str, fps: int, out_path: str):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    pattern = os.path.join(frames_dir, "frame_%06d.png")

    def _run(codec: str):
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(fps)),
            "-start_number",
            "0",
            "-i",
            pattern,
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    res = _run("libx264")
    if res.returncode == 0:
        return True

    res2 = _run("mpeg4")
    if res2.returncode == 0:
        return True

    print("ffmpeg failed (libx264):")
    if res.stderr:
        print(res.stderr)
    print("ffmpeg failed (mpeg4):")
    if res2.stderr:
        print(res2.stderr)
    return False


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    if RECORD_FRAMES or RECORD_VIDEO:
        setattr(env_cfg.env, "enable_cameras", True)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    cam_handles = None
    frames_dirs = None
    video_paths = None

    if RECORD_FRAMES or RECORD_VIDEO:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = VIDEO_DIR
        if base_dir is None:
            load_run_val = getattr(getattr(train_cfg, "runner", None), "load_run", -1)
            if load_run_val is None or load_run_val == -1 or str(load_run_val) == "-1":
                log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
                load_run_name = "last"
                try:
                    runs = os.listdir(log_root)
                    runs.sort()
                    if "exported" in runs:
                        runs.remove("exported")
                    if len(runs) > 0:
                        load_run_name = runs[-1]
                except Exception:
                    load_run_name = "last"
            else:
                load_run_name = str(load_run_val)

            base_dir = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "videos",
                load_run_name,
                stamp,
            )

        cam_props = gymapi.CameraProperties()
        cam_props.width = int(CAMERA_WIDTH)
        cam_props.height = int(CAMERA_HEIGHT)

        cams = CAMERAS
        if cams is None:
            cams = [
                {
                    "name": "default",
                    "follow": bool(CAMERA_FOLLOW),
                    "follow_offset": list(CAMERA_FOLLOW_OFFSET),
                    "lookat_offset": list(CAMERA_LOOKAT_OFFSET),
                    "pos": list(CAMERA_POS),
                    "lookat": list(CAMERA_LOOKAT),
                }
            ]

        cam_handles = {}
        frames_dirs = {}
        video_paths = {}

        for cfg in cams:
            name = str(cfg["name"])
            view_dir = os.path.join(base_dir, name)
            frames_dir = os.path.join(view_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            cam_handle = env.gym.create_camera_sensor(env.envs[CAMERA_ENV_ID], cam_props)
            _set_record_camera(env, cam_handle, CAMERA_ENV_ID, cfg)

            cam_handles[name] = cam_handle
            frames_dirs[name] = frames_dir
            if RECORD_VIDEO:
                video_paths[name] = os.path.join(view_dir, f"{name}.mp4")

        print("Recording views to:", base_dir)

    num_steps = 10 * int(env.max_episode_length)
    if RECORD_STEPS is not None:
        num_steps = int(RECORD_STEPS)
    elif RECORD_SECONDS is not None:
        dt = float(getattr(env, "dt", 0.0))
        if dt > 0:
            num_steps = max(1, int(float(RECORD_SECONDS) / dt))

    frame_idx = 0
    for i in range(num_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        if cam_handles is None:
            continue

        if CAPTURE_EVERY_N_STEPS is not None and int(CAPTURE_EVERY_N_STEPS) > 1:
            if (i % int(CAPTURE_EVERY_N_STEPS)) != 0:
                continue

        if MOVE_CAMERA:
            for cfg in (CAMERAS or []):
                _set_record_camera(env, cam_handles[str(cfg["name"])], CAMERA_ENV_ID, cfg)
        else:
            for cfg in (CAMERAS or []):
                if bool(cfg.get("follow", False)):
                    _set_record_camera(env, cam_handles[str(cfg["name"])], CAMERA_ENV_ID, cfg)

        if env.device != "cpu":
            env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)

        for name, cam_handle in cam_handles.items():
            frame_path = os.path.join(frames_dirs[name], f"frame_{frame_idx:06d}.png")
            env.gym.write_camera_image_to_file(
                env.sim,
                env.envs[CAMERA_ENV_ID],
                cam_handle,
                gymapi.IMAGE_COLOR,
                frame_path,
            )
        frame_idx += 1

    if RECORD_VIDEO and frames_dirs is not None and video_paths is not None:
        for name, frames_dir in frames_dirs.items():
            out_path = video_paths.get(name)
            if out_path is None:
                continue
            ok = _make_video_from_frames(frames_dir=frames_dir, fps=VIDEO_FPS, out_path=out_path)
            if ok:
                print("Saved video to:", out_path)
            else:
                print("Video synth failed; frames are saved in:", frames_dir)



if __name__ == "__main__":
    EXPORT_POLICY = True

    RECORD_FRAMES = False
    RECORD_VIDEO = True

    VIDEO_DIR = None
    VIDEO_FPS = 60

    RECORD_SECONDS = 10.0
    RECORD_STEPS = None
    CAPTURE_EVERY_N_STEPS = 1

    CAMERA_ENV_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    CAMERA_FOLLOW = True
    CAMERA_FOLLOW_OFFSET = [-2.0, 0.0, 1.0]
    CAMERA_LOOKAT_OFFSET = [0.5, 0.0, 0.5]

    CAMERA_POS = [10.0, 0.0, 6.0]
    CAMERA_LOOKAT = [0.0, 0.0, 0.5]

    CAMERAS = [
        # {
        #     "name": "follow",
        #     "follow": True,
        #     "follow_offset": [-2.0, 2.0, 0.55],
        #     "lookat_offset": [0.40, 0.0, 0.08],
        #     "pos": CAMERA_POS,
        #     "lookat": CAMERA_LOOKAT,
        # },
        {
            "name": "low",
            "follow": True,
            "follow_offset": [-1.2, 1.1, 0.03],
            "lookat_offset": [0.25, 0.0, 0.015],
            "pos": CAMERA_POS,
            "lookat": CAMERA_LOOKAT,
        },
        {
            "name": "front",
            "follow": True,
            "follow_offset": [2.2, 0.0, 0.55],
            "lookat_offset": [0.0, 0.0, 0.15],
            "pos": CAMERA_POS,
            "lookat": CAMERA_LOOKAT,
        },
        {
            "name": "overview",
            "follow": False,
            "pos": [5.05, -5.4, 0.78],
            "lookat": [5.0, 5.0, 0.25],
        },
    ]


    MOVE_CAMERA = False

    args = get_args()
    if RECORD_FRAMES or RECORD_VIDEO:
        args.headless = True

    play(args)
