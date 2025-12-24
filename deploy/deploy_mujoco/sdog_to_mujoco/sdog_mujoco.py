import argparse
import contextlib
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from tabulate import tabulate

from legged_gym import LEGGED_GYM_ROOT_DIR


def _format_joint_table(dof_names, q_before, q_after, target_pos, tau_dof, tau_applied_dof=None, precision=4):
    """整合所有关节到一个表格，移除q_before列，保留核心数据列，新增pos_error列"""
    names = list(dof_names) if dof_names is not None else [f"dof{i}" for i in range(int(len(q_before)))]
    prec = int(precision)
    float_fmt = f".{prec}f"

    def fmt(v):
        return f"{float(v): {float_fmt}}"

    all_rows = []
    # 移除q_before列，调整表头顺序
    headers = ["idx", "joint", "q_after", "target_pos", "pos_error", "tau_raw", "tau_apply", "clip_abs", "sat"]

    if tau_applied_dof is None:
        tau_applied_dof = tau_dof

    for i, n in enumerate(names):
        # 关节名称仅保留腿+部位（如FR_hip、RL_calf）
        joint_name_parts = n.split('_')
        if len(joint_name_parts) >= 2:
            joint_name = f"{joint_name_parts[0]}_{joint_name_parts[1]}"
        else:
            joint_name = n[-8:] if len(n) > 8 else n
        joint_name = joint_name.ljust(8)  # 固定8字符对齐

        tau_raw = float(tau_dof[i])
        tau_apply = float(tau_applied_dof[i])
        clip_abs = abs(tau_raw - tau_apply)
        sat = "Y" if clip_abs > 1e-6 else ""
        
        # 计算target_pos与q_after的差值（目标位置 - 当前位置）
        pos_error = target_pos[i] - q_after[i]

        all_rows.append([
            f"{i:02d}",
            joint_name,
            fmt(q_after[i]),       # 仅保留q_after，删除q_before
            fmt(target_pos[i]),    # 仅显示合并后的目标位置
            fmt(pos_error),        # 新增：位置误差列
            fmt(tau_raw),
            fmt(tau_apply),
            fmt(clip_abs),
            sat,
        ])

    full_table = tabulate(
        all_rows,
        headers=headers,
        tablefmt="grid",
        floatfmt=float_fmt,
        stralign="left"
    )
    
    return full_table


def _format_obs_table(obs, cfg, is_settle_phase, precision=4):
    """格式化观测值（模型输入）为2个表格：非关节信息 + 关节信息"""
    prec = int(precision)
    float_fmt = f".{prec}f"
    
    def fmt(v):
        return f"{float(v): {float_fmt}}"
    
    # 辅助函数：清理关节名称为「腿+部位」格式（如FR_hip、RL_calf）
    def clean_joint_name(name):
        parts = name.split('_')
        if len(parts) >= 2:
            clean_name = f"{parts[0]}_{parts[1]}"
        else:
            clean_name = name  # 兜底
        return clean_name.ljust(8)  # 固定8字符对齐，保证表格整齐
    
    num_actions = cfg["num_actions"]
    dof_names = cfg.get("dof_names", [f"dof{i}" for i in range(num_actions)])
    # 确保dof_names长度匹配，不足则补全
    if len(dof_names) < num_actions:
        dof_names += [f"dof{i}" for i in range(len(dof_names), num_actions)]
    
    # ========== 表格1：非关节信息 ==========
    non_joint_data = []
    # 1. 基础线性速度（机体坐标系）
    base_lin_vel = obs[0:3]
    non_joint_data.append([
        "base_lin_vel（基础线速度）",
        f"x: {fmt(base_lin_vel[0])}",
        f"y: {fmt(base_lin_vel[1])}",
        f"z: {fmt(base_lin_vel[2])}",
        "已缩放（机体坐标系）"
    ])
    
    # 2. 基础角速度（机体坐标系）
    base_ang_vel = obs[3:6]
    non_joint_data.append([
        "base_ang_vel（基础角速度）",
        f"roll: {fmt(base_ang_vel[0])}",
        f"pitch: {fmt(base_ang_vel[1])}",
        f"yaw: {fmt(base_ang_vel[2])}",
        "已缩放（机体坐标系）"
    ])
    
    # 3. 投影重力向量
    projected_gravity = obs[6:9]
    non_joint_data.append([
        "projected_gravity（投影重力）",
        f"x: {fmt(projected_gravity[0])}",
        f"y: {fmt(projected_gravity[1])}",
        f"z: {fmt(projected_gravity[2])}",
        "机体坐标系"
    ])
    
    # 4. 控制指令（明确wz是偏航角速度）
    cmd = obs[9:12]
    non_joint_data.append([
        "cmd（运动控制指令）",
        f"vx: {fmt(cmd[0])} (前后线速度)",
        f"vy: {fmt(cmd[1])} (左右线速度)",
        f"wz: {fmt(cmd[2])} (偏航角速度)",
        "已缩放（平面运动控制）"
    ])
    
    non_joint_table = tabulate(
        non_joint_data,
        headers=["非关节观测项", "维度1", "维度2", "维度3", "说明"],
        tablefmt="grid",
        floatfmt=float_fmt,
        stralign="left"
    )
    
    # ========== 表格2：关节信息 ==========
    # 提取关节数据
    # 关节位置（相对于默认角度，已缩放）
    qj_start = 12
    qj_end = 12 + num_actions
    qj = obs[qj_start:qj_end]
    
    # 关节速度（已缩放）
    dqj_start = qj_end
    dqj_end = dqj_start + num_actions
    dqj = obs[dqj_start:dqj_end]
    
    # 上一步动作
    action_start = dqj_end
    action_end = action_start + num_actions
    action = obs[action_start:action_end]
    
    # 构建关节信息表格：3行（位置/速度/动作），列名仅保留腿+部位
    joint_data = [
        ["关节位置（标准化后）"] + [fmt(v) for v in qj],
        ["关节速度（标准化后）"] + [fmt(v) for v in dqj],
        ["上一步动作（-1~1）"] + [fmt(v) for v in action]
    ]
    
    # 表头：行类型 + 清理后的关节名称（仅腿+部位）
    joint_headers = ["观测类型"] + [clean_joint_name(name) for name in dof_names]
    
    joint_table = tabulate(
        joint_data,
        headers=joint_headers,
        tablefmt="grid",
        floatfmt=float_fmt,
        stralign="left"
    )
    
    # ========== 组合所有表格 ==========
    phase_note = f"\n【模型输入说明】: {'静置阶段（obs未更新）' if is_settle_phase else '策略控制阶段（obs为模型输入）'}\n"
    
    full_table = (
        phase_note
        + "\n===== 表格1：非关节观测信息 =====\n"
        + non_joint_table
        + "\n\n===== 表格2：关节观测信息 =====\n"
        + joint_table
    )
    
    return full_table


def _setup_timestamp_logfile(log_dir=".", prefix="sdog_mujoco"):
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    name = f"{prefix}_{ts}.log"
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, name)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _quat_rotate(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return _quat_multiply(_quat_multiply(q, vq), _quat_conjugate(q))[1:]


def quat_rotate_inverse(q, v):
    return _quat_rotate(_quat_conjugate(q), v)


def _get_root_qpos_qvel_start(mujoco_model):
    qpos_start = 0
    qvel_start = 0
    if int(getattr(mujoco_model, "njnt", 0)) > 0:
        try:
            if int(mujoco_model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
                qpos_start = 7
                qvel_start = 6
        except Exception:
            pass
    return qpos_start, qvel_start


def _get_actuator_joint_adrs(mujoco_model):
    nu = int(getattr(mujoco_model, "nu", 0))
    qpos_adrs = np.zeros(nu, dtype=np.int32)
    dof_adrs = np.zeros(nu, dtype=np.int32)
    for i in range(nu):
        jntid = int(mujoco_model.actuator_trnid[i, 0])
        if jntid < 0:
            qpos_adrs[i] = -1
            dof_adrs[i] = -1
            continue
        qpos_adrs[i] = int(mujoco_model.jnt_qposadr[jntid])
        dof_adrs[i] = int(mujoco_model.jnt_dofadr[jntid])
    return qpos_adrs, dof_adrs


def _get_actuator_joint_names(mujoco_model):
    nu = int(getattr(mujoco_model, "nu", 0))
    names = []
    for i in range(nu):
        jntid = int(mujoco_model.actuator_trnid[i, 0])
        if jntid < 0:
            names.append(None)
            continue
        names.append(mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, jntid))
    return names


def _min_contact_dist(mujoco_data):
    try:
        ncon = int(getattr(mujoco_data, "ncon", 0))
        if ncon <= 0:
            return None
        dmin = None
        for i in range(ncon):
            d = float(mujoco_data.contact[i].dist)
            if dmin is None or d < dmin:
                dmin = d
        return dmin
    except Exception:
        return None


def _lift_root_if_penetrating(mujoco_model, mujoco_data, max_iters=300, dz=0.002, eps=-1e-4):
    if int(getattr(mujoco_model, "nq", 0)) < 3:
        return
    for _ in range(int(max_iters)):
        mujoco.mj_forward(mujoco_model, mujoco_data)
        dmin = _min_contact_dist(mujoco_data)
        if dmin is None or dmin >= float(eps):
            break
        mujoco_data.qpos[2] = float(mujoco_data.qpos[2]) + float(dz)


def _drop_root_until_contact(
    mujoco_model,
    mujoco_data,
    max_iters=800,
    dz=0.002,
    z_min=-1.0,
    penetration_eps=-1e-4,
):
    if int(getattr(mujoco_model, "nq", 0)) < 3:
        return

    for _ in range(int(max_iters)):
        mujoco.mj_forward(mujoco_model, mujoco_data)
        ncon = int(getattr(mujoco_data, "ncon", 0))
        if ncon > 0:
            dmin = _min_contact_dist(mujoco_data)
            if dmin is not None and float(dmin) < float(penetration_eps):
                _lift_root_if_penetrating(mujoco_model, mujoco_data, eps=penetration_eps)
                mujoco.mj_forward(mujoco_model, mujoco_data)
            break

        mujoco_data.qpos[2] = float(mujoco_data.qpos[2]) - float(dz)
        if float(mujoco_data.qpos[2]) <= float(z_min):
            break

    mujoco.mj_forward(mujoco_model, mujoco_data)


def load_config(config_file: str):
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])

    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)

    lin_vel_scale = float(cfg.get("lin_vel_scale", 1.0))
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

    cmd_deadzone = float(cfg.get("cmd_deadzone", 0.2))

    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)

    settle_steps = int(cfg.get("settle_steps", 100))
    clip_observations = float(cfg.get("clip_observations", 100.0))
    clip_actions = float(cfg.get("clip_actions", 100.0))

    dof_names = cfg.get("dof_names", None)
    dof_signs = cfg.get("dof_signs", None)

    expected_num_obs = 12 + 3 * num_actions
    if num_obs != expected_num_obs:
        raise ValueError(f"sdog obs expects num_obs={expected_num_obs}, got {num_obs}")

    return {
        "policy_path": policy_path,
        "xml_path": xml_path,
        "simulation_duration": simulation_duration,
        "simulation_dt": simulation_dt,
        "control_decimation": control_decimation,
        "kps": kps,
        "kds": kds,
        "default_angles": default_angles,
        "lin_vel_scale": lin_vel_scale,
        "ang_vel_scale": ang_vel_scale,
        "dof_pos_scale": dof_pos_scale,
        "dof_vel_scale": dof_vel_scale,
        "action_scale": action_scale,
        "cmd_scale": cmd_scale,
        "cmd_deadzone": cmd_deadzone,
        "num_actions": num_actions,
        "num_obs": num_obs,
        "cmd": cmd,
        "settle_steps": settle_steps,
        "clip_observations": clip_observations,
        "clip_actions": clip_actions,
        "dof_names": dof_names,
        "dof_signs": dof_signs,
    }


def build_obs_sdog(mujoco_data, cfg, action, obs):
    num_actions = cfg["num_actions"]
    default_angles = cfg["default_angles"]
    lin_vel_scale = cfg.get("lin_vel_scale", 1.0)
    ang_vel_scale = cfg["ang_vel_scale"]
    dof_pos_scale = cfg["dof_pos_scale"]
    dof_vel_scale = cfg["dof_vel_scale"]
    cmd = cfg["cmd"]
    cmd_scale = cfg["cmd_scale"]
    cmd_deadzone = float(cfg.get("cmd_deadzone", 0.2))

    qpos_start = int(cfg.get("qpos_start", 7))
    qvel_start = int(cfg.get("qvel_start", 6))

    if qpos_start == 7 and qvel_start == 6:
        quat = mujoco_data.qpos[3:7].astype(np.float32)
        base_lin_vel = quat_rotate_inverse(quat, mujoco_data.qvel[0:3].astype(np.float32)) * float(lin_vel_scale)
        base_ang_vel = quat_rotate_inverse(quat, mujoco_data.qvel[3:6].astype(np.float32)) * ang_vel_scale
        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base_lin_vel = np.zeros(3, dtype=np.float32)
        base_ang_vel = np.zeros(3, dtype=np.float32)
        projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    dof_qpos_adrs = cfg.get("dof_qpos_adrs", None)
    dof_dof_adrs = cfg.get("dof_dof_adrs", None)
    dof_signs_arr = cfg.get("dof_signs_arr", None)
    if dof_qpos_adrs is not None and dof_dof_adrs is not None:
        qj_phys = mujoco_data.qpos[dof_qpos_adrs].astype(np.float32)
        dqj_phys = mujoco_data.qvel[dof_dof_adrs].astype(np.float32)
        if dof_signs_arr is not None:
            qj = qj_phys * dof_signs_arr
            dqj = dqj_phys * dof_signs_arr
        else:
            qj = qj_phys
            dqj = dqj_phys
    else:
        qj = mujoco_data.qpos[qpos_start : qpos_start + num_actions].astype(np.float32)
        dqj = mujoco_data.qvel[qvel_start : qvel_start + num_actions].astype(np.float32)

    qj = (qj - default_angles) * dof_pos_scale
    dqj = dqj * dof_vel_scale

    obs[0:3] = base_lin_vel
    obs[3:6] = base_ang_vel
    cmd_eff = np.array(cmd, dtype=np.float32)
    if cmd_eff.shape[0] >= 2 and float(np.linalg.norm(cmd_eff[:2])) <= float(cmd_deadzone):
        cmd_eff[:2] = 0.0

    obs[6:9] = projected_gravity
    obs[9:12] = cmd_eff * cmd_scale
    obs[12 : 12 + num_actions] = qj
    obs[12 + num_actions : 12 + 2 * num_actions] = dqj
    obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action

    return obs


def setup_mujoco_and_config(config_file):
    cfg = load_config(config_file)

    mujoco_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    mujoco_data = mujoco.MjData(mujoco_model)
    mujoco_model.opt.timestep = cfg["simulation_dt"]

    qpos_start, qvel_start = _get_root_qpos_qvel_start(mujoco_model)
    cfg["qpos_start"] = int(qpos_start)
    cfg["qvel_start"] = int(qvel_start)

    act_qpos_adrs, act_dof_adrs = _get_actuator_joint_adrs(mujoco_model)
    if np.any(act_qpos_adrs < 0) or np.any(act_dof_adrs < 0):
        raise ValueError("Some actuators are not directly mapped to joints; cannot build stable joint-ordered control")
    cfg["act_qpos_adrs"] = act_qpos_adrs
    cfg["act_dof_adrs"] = act_dof_adrs

    act_joint_names = _get_actuator_joint_names(mujoco_model)
    if any(n is None for n in act_joint_names):
        raise ValueError("Some actuators are not mapped to joints; cannot build name-based dof mapping")

    dof_names = cfg.get("dof_names", None)
    if dof_names is None:
        dof_names = act_joint_names
    if len(dof_names) != int(cfg["num_actions"]):
        raise ValueError(f"dof_names must have length {cfg['num_actions']}, got {len(dof_names)}")

    # 清理dof_names为「腿+部位」格式
    cleaned_dof_names = []
    for name in dof_names:
        parts = name.split('_')
        if len(parts) >= 2:
            cleaned_name = f"{parts[0]}_{parts[1]}"
        else:
            cleaned_name = name
        cleaned_dof_names.append(cleaned_name)
    cfg["dof_names"] = cleaned_dof_names  # 更新为清理后的名称

    name_to_act = {n: i for i, n in enumerate(act_joint_names)}
    dof_to_act = np.array([name_to_act[n] for n in dof_names], dtype=np.int32)
    cfg["dof_to_act"] = dof_to_act
    cfg["dof_qpos_adrs"] = act_qpos_adrs[dof_to_act]
    cfg["dof_dof_adrs"] = act_dof_adrs[dof_to_act]

    dof_signs = cfg.get("dof_signs", None)
    if dof_signs is None:
        dof_signs_arr = np.ones(int(cfg["num_actions"]), dtype=np.float32)
    else:
        dof_signs_arr = np.array(dof_signs, dtype=np.float32)
        if dof_signs_arr.shape[0] != int(cfg["num_actions"]):
            raise ValueError(f"dof_signs must have length {cfg['num_actions']}, got {dof_signs_arr.shape[0]}")
    cfg["dof_signs_arr"] = dof_signs_arr

    # DOF映射信息改为单表格输出（名称仅保留腿+部位）
    dof_mapping_table = [
        [i, cfg["dof_names"][i], act_joint_names[dof_to_act[i]], dof_to_act[i], dof_signs_arr[i]]
        for i in range(len(cfg["dof_names"]))
    ]
    print("[sdog_mujoco] DOF映射信息：")
    print(tabulate(
        dof_mapping_table,
        headers=["dof_idx", "dof_name", "act_joint_name", "act_idx", "dof_sign"],
        tablefmt="grid",
        floatfmt=".4f"
    ))

    return cfg, mujoco_model, mujoco_data


def init_simulation_state(mujoco_model, mujoco_data, cfg):
    try:
        mujoco.mj_resetData(mujoco_model, mujoco_data)
    except Exception:
        pass

    mujoco_data.qpos[cfg["dof_qpos_adrs"]] = (cfg["default_angles"] * cfg["dof_signs_arr"]).astype(np.float32)
    mujoco_data.qvel[cfg["dof_dof_adrs"]] = 0.0
    if int(cfg.get("qpos_start", 0)) == 7 and int(cfg.get("qvel_start", 0)) == 6:
        mujoco_data.qvel[0:6] = 0.0

    mujoco.mj_forward(mujoco_model, mujoco_data)
    if int(cfg.get("qpos_start", 0)) == 7:
        _drop_root_until_contact(mujoco_model, mujoco_data)
        _lift_root_if_penetrating(mujoco_model, mujoco_data)
        mujoco.mj_forward(mujoco_model, mujoco_data)


def run_simulation_loop(
    mujoco_model,
    mujoco_data,
    cfg,
    policy,
    show_viewer=True,
    log_every=1,
):
    action = np.zeros(cfg["num_actions"], dtype=np.float32)
    target_dof_pos = cfg["default_angles"].copy()  # 仅保留这一个目标位置变量
    obs = np.zeros(cfg["num_obs"], dtype=np.float32)
    counter = 0
    ctrl_updates = 0  # 控制更新计数（每control_decimation步+1）
    settle_steps = int(cfg.get("settle_steps", 0))  # 读取静置步数配置

    dof_names = cfg.get("dof_names", None)
    if dof_names is not None:
        # DOF名称列表改为单表格（仅显示腿+部位）
        dof_names_table = [[i, n] for i, n in enumerate(dof_names)]
        print("\n[sdog_mujoco] DOF名称列表：")
        print(tabulate(
            dof_names_table,
            headers=["idx", "dof_name"],
            tablefmt="grid"
        ))

    viewer_ctx = mujoco.viewer.launch_passive(mujoco_model, mujoco_data) if show_viewer else contextlib.nullcontext(None)
    with viewer_ctx as viewer:
        start = time.time()
        while (time.time() - start) < cfg["simulation_duration"]:
            if show_viewer and (viewer is not None) and (not viewer.is_running()):
                break
            step_start = time.time()

            # 静置阶段逻辑：强制锁死初始姿态
            if counter < settle_steps:
                action[:] = 0.0
                target_dof_pos = cfg["default_angles"].copy()

            q_phys = mujoco_data.qpos[cfg["dof_qpos_adrs"]].astype(np.float32)
            dq_phys = mujoco_data.qvel[cfg["dof_dof_adrs"]].astype(np.float32)
            q = q_phys * cfg["dof_signs_arr"]
            dq = dq_phys * cfg["dof_signs_arr"]

            # 移除冗余的target_cmd变量，直接使用target_dof_pos
            q_before = q.copy()  # 保留变量但不再输出到表格

            tau_dof = pd_control(
                target_dof_pos,  # 直接用合并后的变量
                q,
                cfg["kps"],
                np.zeros_like(cfg["kds"], dtype=np.float32),
                dq,
                cfg["kds"],
            ).astype(np.float32)

            tau_dof = tau_dof * cfg["dof_signs_arr"]
            tau = np.zeros(int(getattr(mujoco_model, "nu", 0)), dtype=np.float32)
            tau[cfg["dof_to_act"]] = tau_dof

            tau = tau.astype(np.float32)
            try:
                if int(getattr(mujoco_model, "nu", 0)) == tau.shape[0]:
                    if hasattr(mujoco_model, "actuator_ctrllimited") and hasattr(mujoco_model, "actuator_ctrlrange"):
                        limited = mujoco_model.actuator_ctrllimited.astype(bool)
                        if np.any(limited):
                            lo = mujoco_model.actuator_ctrlrange[:, 0]
                            hi = mujoco_model.actuator_ctrlrange[:, 1]
                            tau = np.clip(tau, lo, hi)
                    elif hasattr(mujoco_model, "actuator_forcerange"):
                        lo = mujoco_model.actuator_forcerange[:, 0]
                        hi = mujoco_model.actuator_forcerange[:, 1]
                        tau = np.clip(tau, lo, hi)
            except Exception:
                pass

            tau_applied_dof = None
            try:
                if tau.shape[0] > 0:
                    tau_applied_dof = tau[cfg["dof_to_act"]].astype(np.float32)
            except Exception:
                tau_applied_dof = None

            mujoco_data.ctrl[:] = tau
            mujoco.mj_step(mujoco_model, mujoco_data)
            counter += 1

            q_phys_after = mujoco_data.qpos[cfg["dof_qpos_adrs"]].astype(np.float32)
            q_after = q_phys_after * cfg["dof_signs_arr"]

            # 按仿真总步数输出日志（解决step间隔为4的问题）
            do_log = int(log_every) > 0 and (counter % int(log_every) == 0)
            if do_log:
                ncon = int(getattr(mujoco_data, "ncon", 0))
                # ========== 核心修改：同时标注静置阶段+控制更新步 ==========
                # 1. 判断是否在静置阶段
                is_settle_phase = counter < settle_steps
                phase_tag = "静置阶段" if is_settle_phase else "策略控制阶段"
                
                # 2. 判断是否是控制更新步
                is_ctrl_step = (counter % cfg["control_decimation"]) == 0
                ctrl_tag = f"控制更新步 | ctrl_updates={ctrl_updates + 1}" if is_ctrl_step else "普通仿真步"
                
                # 3. 构建最终日志头
                header = (
                    f"[sdog_mujoco] t={float(time.time() - start):.3f}s "
                    f"step={counter} [{phase_tag} | {ctrl_tag}] ncon={ncon}"
                )
                print(header)
                
                # 调用表格函数时仅传合并后的target_dof_pos
                table_str = _format_joint_table(
                    cfg.get("dof_names", None),
                    q_before,  # 仍传入但表格中不再显示
                    q_after,
                    target_dof_pos,
                    tau_dof,
                    tau_applied_dof=tau_applied_dof,
                    precision=4,
                )
                print(table_str)
                
                # ========== 新增：打印模型输入（观测值） ==========
                obs_table_str = _format_obs_table(obs, cfg, is_settle_phase, precision=4)
                print("\n[模型输入/Observation]")
                print(obs_table_str)
                
                print("-" * 120)

            # 控制更新逻辑（仅在control_decimation步触发）
            if counter % cfg["control_decimation"] == 0:
                ctrl_updates += 1
                # 仅当超出静置阶段时，才执行策略推理
                if counter >= settle_steps:
                    obs = build_obs_sdog(mujoco_data, cfg, action, obs)
                    np.clip(obs, -cfg["clip_observations"], cfg["clip_observations"], out=obs)
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().cpu().numpy().squeeze().astype(np.float32)
                    action = np.clip(action, -cfg["clip_actions"], cfg["clip_actions"]).astype(np.float32)
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    target_dof_pos = action * cfg["action_scale"] + cfg["default_angles"]

            if show_viewer and (viewer is not None):
                viewer.sync()

            time_until_next_step = mujoco_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SDOG MuJoCo Simulation")
    parser.add_argument("--config", type=str, default="sdog.yaml", help="配置文件路径")
    parser.add_argument("--no_viewer", action="store_true", help="不显示可视化窗口")
    parser.add_argument("--log_every", type=int, default=1, help="日志输出间隔(仿真总步数)")
    parser.add_argument("--log_dir", type=str, default="./sdog_logs", help="日志保存目录")
    
    args = parser.parse_args()

    config_file = args.config
    no_viewer = args.no_viewer
    log_every = args.log_every
    log_dir = args.log_dir

    # 创建日志文件（自动创建sdog_logs文件夹）
    log_path = _setup_timestamp_logfile(log_dir=log_dir, prefix="sdog_mujoco")
    # 仅重定向标准输出到日志文件，标准错误保留到终端
    log_f = open(log_path, "w", buffering=1, encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = log_f

    # 终端打印日志文件路径（stderr输出）
    print(f"[sdog_mujoco] 日志文件将保存至: {log_path}", file=sys.stderr)
    
    try:
        # 提前加载配置，打印关键参数
        cfg = load_config(config_file)
        settle_steps = int(cfg.get("settle_steps", 0))
        control_decimation = int(cfg.get("control_decimation", 4))
        
        # 日志文件中记录基础信息
        print(f"[sdog_mujoco] 日志文件路径: {log_path}")
        print(f"[sdog_mujoco] 配置文件: {config_file}")
        print(f"[sdog_mujoco] 日志输出间隔: {log_every}步（仿真总步数）")
        print(f"[sdog_mujoco] 控制降频系数(control_decimation): {control_decimation}")
        print(f"[sdog_mujoco] 静置步数(settle_steps): {settle_steps}")
        print("-" * 120)

        # 初始化配置和仿真环境
        cfg, mujoco_model, mujoco_data = setup_mujoco_and_config(config_file)
        policy = torch.jit.load(cfg["policy_path"])

        # 初始化仿真状态
        init_simulation_state(mujoco_model, mujoco_data, cfg)
        
        # 运行仿真循环
        run_simulation_loop(
            mujoco_model,
            mujoco_data,
            cfg,
            policy,
            show_viewer=not no_viewer,
            log_every=log_every,
        )
    except Exception as e:
        # 异常信息输出到终端（stderr）
        import traceback
        print(f"[sdog_mujoco] 运行出错: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # 恢复标准输出
        try:
            sys.stdout = orig_stdout
        except Exception as e:
            print(f"[sdog_mujoco] 恢复标准输出失败: {e}", file=sys.stderr)
        
        # 关闭日志文件
        try:
            log_f.close()
            print(f"[sdog_mujoco] 日志已成功保存至: {log_path}", file=sys.stderr)
        except Exception as e:
            print(f"[sdog_mujoco] 关闭日志文件失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()