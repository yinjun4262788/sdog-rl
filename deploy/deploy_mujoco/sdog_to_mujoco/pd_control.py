import argparse
import contextlib
import time
import os
import datetime

import mujoco
import mujoco.viewer
import numpy as np
from tabulate import tabulate

MODEL_PATH = "/home/vkrobot/yinjun/sdog-rl/sdog-rl/resources/robots/sdog/scene.xml"

LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

TARGET_ANGLES = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
KP = np.array([20.0] * 12, dtype=np.float64)
KD = np.array([0.5] * 12, dtype=np.float64)


def _build_mappings(model, joint_names):
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    qpos_adrs = np.array([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int32)
    dof_adrs = np.array([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=np.int32)

    act_ids = np.full(len(joint_ids), -1, dtype=np.int32)
    for a in range(int(getattr(model, "nu", 0))):
        jntid = int(model.actuator_trnid[a, 0])
        for i, jid in enumerate(joint_ids):
            if jntid == int(jid):
                act_ids[i] = a
                break

    if np.any(act_ids < 0):
        missing = [joint_names[i] for i in range(len(joint_names)) if act_ids[i] < 0]
        raise ValueError(f"Some joints have no direct motor actuator mapping: {missing}")

    return qpos_adrs, dof_adrs, act_ids


def _clip_ctrl(model, ctrl):
    ctrl = ctrl.astype(np.float64)
    try:
        if hasattr(model, "actuator_ctrllimited") and hasattr(model, "actuator_ctrlrange"):
            limited = model.actuator_ctrllimited.astype(bool)
            if np.any(limited):
                lo = model.actuator_ctrlrange[:, 0]
                hi = model.actuator_ctrlrange[:, 1]
                ctrl = np.clip(ctrl, lo, hi)
        elif hasattr(model, "actuator_forcerange"):
            lo = model.actuator_forcerange[:, 0]
            hi = model.actuator_forcerange[:, 1]
            ctrl = np.clip(ctrl, lo, hi)
    except Exception:
        pass
    return ctrl


# 精准计算字符宽度（中文字符算2个，英文字符算1个）
def get_char_width(s):
    width = 0
    for c in s:
        if '\u4e00' <= c <= '\u9fff':  # 中文字符范围
            width += 2
        else:
            width += 1
    return width


# 填充字符串到指定宽度（兼容中英文）
def pad_to_width(s, target_width, align='left'):
    current_width = get_char_width(s)
    pad_len = target_width - current_width
    if pad_len <= 0:
        return s
    if align == 'left':
        return s + ' ' * pad_len
    elif align == 'right':
        return ' ' * pad_len + s
    else:
        return ' ' * (pad_len // 2) + s + ' ' * (pad_len - pad_len // 2)


def run_hardcoded_servo(model, data, duration_s, ramp_s, show_viewer=True, print_table=True, log_file=None):
    qpos_adrs, dof_adrs, act_ids = _build_mappings(model, LEG_JOINT_NAMES)

    # 初始化仿真数据
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    q_start = data.qpos[qpos_adrs].astype(np.float64).copy()

    # 统计变量初始化
    sat_steps = 0
    sat_sum = 0
    max_abs_tau_cmd = 0.0
    max_abs_act_force = 0.0
    ncon_sum = 0
    ncon_max = 0

    # ========== 核心调整：恢复5列表格配置，移除ncon列 ==========
    COLUMN_WIDTHS = {
        "关节名称": 14,        # 列1：14字符宽
        "期望角度(rad)": 14,  # 列2：14字符宽
        "控制后实际角度(rad)": 18,  # 列3：18字符宽
        "PD计算力矩(N·m)": 18,      # 列4：18字符宽
        "限幅后力矩(N·m)": 18       # 列5：18字符宽
    }
    # 原始表头（仅保留5列）
    RAW_HEADERS = [
        "关节名称", "期望角度(rad)", "控制后实际角度(rad)",
        "PD计算力矩(N·m)", "限幅后力矩(N·m)"
    ]
    # 填充表头到固定宽度
    PADDED_HEADERS = [
        pad_to_width(header, COLUMN_WIDTHS[header], align='center')
        for header in RAW_HEADERS
    ]

    step_count = 0

    # 启动Viewer
    viewer_ctx = mujoco.viewer.launch_passive(model, data) if show_viewer else contextlib.nullcontext(None)
    with viewer_ctx as viewer:
        start = time.time()
        log_file.write("===== PD控制仿真开始 =====\n")
        log_file.write(f"仿真时长：{duration_s}s，斜坡过渡时长：{ramp_s}s\n\n")
        log_file.flush()

        while (time.time() - start) < float(duration_s):
            if show_viewer and (viewer is not None) and (not viewer.is_running()):
                break
            step_start = time.time()

            # 1. 计算当前时刻的期望角度
            t = time.time() - start
            alpha = 1.0 if ramp_s <= 0 else float(np.clip(t / float(ramp_s), 0.0, 1.0))
            q_target = (1.0 - alpha) * q_start + alpha * TARGET_ANGLES

            # 2. 读取Step执行前的关节状态
            q_pre = data.qpos[qpos_adrs].astype(np.float64)
            dq_pre = data.qvel[dof_adrs].astype(np.float64)

            # 3. 计算PD控制力矩
            tau_pd = (q_target - q_pre) * KP + (0.0 - dq_pre) * KD

            # 4. 构造控制指令并限幅
            ctrl = np.zeros(int(getattr(model, "nu", 0)), dtype=np.float64)
            ctrl[act_ids] = tau_pd
            ctrl_clipped = _clip_ctrl(model, ctrl)
            tau_clipped = ctrl_clipped[act_ids]

            # 5. 施加控制指令并执行仿真Step
            data.ctrl[:] = ctrl_clipped.astype(np.float32)
            mujoco.mj_step(model, data)

            # 6. 读取Step执行后的关节角度
            q_post = data.qpos[qpos_adrs].astype(np.float64)

            # 7. 统计数据更新
            sat_mask = np.abs(tau_clipped - tau_pd) > 1e-12
            sat_steps += 1
            sat_sum += int(np.sum(sat_mask))
            max_abs_tau_cmd = max(float(max_abs_tau_cmd), float(np.max(np.abs(tau_clipped))))

            try:
                af = data.actuator_force.astype(np.float64)
                max_abs_act_force = max(float(max_abs_act_force), float(np.max(np.abs(af[act_ids]))))
            except Exception:
                pass

            # 获取当前Step的ncon值（仅用于Step标识行显示）
            try:
                current_ncon = int(getattr(data, "ncon", 0))  # 当前step的接触数量
                ncon_sum += current_ncon
                ncon_max = max(int(ncon_max), current_ncon)
            except Exception:
                current_ncon = 0  # 异常时默认0

            # ========== 表格输出（移除ncon列，仅保留5列） ==========
            if print_table:
                step_table_data = []
                for i, joint_name in enumerate(LEG_JOINT_NAMES):
                    # 1. 格式化数值（保留4位小数）
                    q_target_str = f"{q_target[i]:.4f}"
                    q_post_str = f"{q_post[i]:.4f}"
                    tau_pd_str = f"{tau_pd[i]:.4f}"
                    tau_clipped_str = f"{tau_clipped[i]:.4f}"

                    # 2. 填充每个单元格到固定宽度
                    col1 = pad_to_width(joint_name, COLUMN_WIDTHS["关节名称"], align='left')
                    col2 = pad_to_width(q_target_str, COLUMN_WIDTHS["期望角度(rad)"], align='right')
                    col3 = pad_to_width(q_post_str, COLUMN_WIDTHS["控制后实际角度(rad)"], align='right')
                    col4 = pad_to_width(tau_pd_str, COLUMN_WIDTHS["PD计算力矩(N·m)"], align='right')
                    col5 = pad_to_width(tau_clipped_str, COLUMN_WIDTHS["限幅后力矩(N·m)"], align='right')

                    # 仅保留5列数据，移除ncon列
                    step_table_data.append([col1, col2, col3, col4, col5])

                # 生成表格（恢复5列对齐方式）
                table_str = tabulate(
                    step_table_data,
                    headers=PADDED_HEADERS,
                    tablefmt="grid",
                    colalign=["left", "right", "right", "right", "right"],  # 恢复5列对齐
                    disable_numparse=True  # 禁用数字解析，避免干扰
                )

                # ========== 关键保留：Step标识行显示ncon（表头级） ==========
                log_file.write(f"\n{'='*90}\n")
                log_file.write(f"Step {step_count} | 时间：{t:.3f}s | 接触数量(ncon)：{current_ncon}\n")
                log_file.write(f"{'='*90}\n")
                log_file.write(table_str + "\n")
                log_file.flush()

            # 同步Viewer和控制步频
            if show_viewer and (viewer is not None):
                viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            step_count += 1

    # 仿真结束统计（保留ncon汇总）
    log_file.write(f"\n{'='*90}\n")
    log_file.write("===== 仿真结束汇总统计 =====\n")
    log_file.write(f"{'='*90}\n")
    q_final = data.qpos[qpos_adrs].astype(np.float64).copy()
    np.set_printoptions(precision=4, suppress=True)
    log_file.write("TARGET_ANGLES（最终期望角度）：\n")
    log_file.write(f"{TARGET_ANGLES}\n")
    log_file.write("\nq_final（最终控制后实际角度）：\n")
    log_file.write(f"{q_final}\n")
    log_file.write("\n角度误差（期望 - 实际）：\n")
    log_file.write(f"{TARGET_ANGLES - q_final}\n")

    if sat_steps > 0:
        sat_ratio = float(sat_sum) / float(sat_steps * len(act_ids))
        log_file.write("\n扭矩限幅统计：\n")
        log_file.write(f"max_abs_tau_cmd = {max_abs_tau_cmd:.3f} N·m\n")
        log_file.write(f"max_abs_act_force = {max_abs_act_force:.3f} N·m\n")
        log_file.write(f"saturation_ratio = {sat_ratio:.3%}\n")
        log_file.write(f"ncon_avg = {float(ncon_sum) / float(sat_steps):.3f}\n")
        log_file.write(f"ncon_max = {int(ncon_max)}\n")
        log_file.write(f"总Step数 = {step_count}\n")

    log_file.flush()
    print(f"PD控制仿真输出已写入日志文件：{log_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0, help="仿真总时长（秒）")
    parser.add_argument("--ramp", type=float, default=0.0, help="期望角度斜坡过渡时长（秒）")
    parser.add_argument("--no_gravity", action="store_true", help="禁用重力")
    parser.add_argument("--no_contact", action="store_true", help="禁用接触检测")
    parser.add_argument("--no_viewer", action="store_true", help="禁用仿真可视化窗口")
    parser.add_argument("--no_table", action="store_true", help="禁用每Step表格输出")
    parser.add_argument("--log-path", type=str, default=None,
                        help="日志文件路径（默认：./sdog_pd_ctrl_YYYYMMDD_HHMMSS.log）")
    args = parser.parse_args()

    # 初始化日志文件
    if args.log_path:
        log_file = open(args.log_path, "w", encoding="utf-8")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"sdog_pd_ctrl_{timestamp}.log"
        log_file = open(log_filename, "w", encoding="utf-8")

    try:
        # 加载模型
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        if args.no_contact:
            bit = getattr(getattr(mujoco, "mjtDisableBit", None), "mjDSBL_CONTACT", None)
            if bit is not None:
                model.opt.disableflags |= int(bit)
        if args.no_gravity:
            model.opt.gravity[:] = 0.0

        data = mujoco.MjData(model)

        # 执行PD控制
        run_hardcoded_servo(
            model,
            data,
            duration_s=args.duration,
            ramp_s=args.ramp,
            show_viewer=not args.no_viewer,
            print_table=not args.no_table,
            log_file=log_file
        )
    finally:
        log_file.close()
        print(f"日志文件已关闭：{log_file.name}")


if __name__ == "__main__":
    main()