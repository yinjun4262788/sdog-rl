import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "/home/vkrobot/yinjun/sdog-rl/sdog-rl/resources/robots/sdog/scene.xml"

# 需要调的 12 个关节名，顺序和 sdog.yaml 里的 dof_names 一致
LEG_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 关掉重力，让机器人悬空，便于调姿态
    model.opt.gravity[:] = 0.0

    # 建立每个关节在 qpos 里的索引，方便后面读角度
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINT_NAMES]
    qpos_ids = [model.jnt_qposadr[jid] for jid in joint_ids]

    print("已加载模型，可以在右侧 Joint 面板里拖动这 12 个关节：")
    print(LEG_JOINT_NAMES)

    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)

    # 窗口关闭后，打印当前 12 个关节的角度
    angles = data.qpos[qpos_ids].copy()
    np.set_printoptions(precision=4, suppress=True)
    print("\n当前关节角（按 sdog.yaml 的顺序）为：")
    for name, val in zip(LEG_JOINT_NAMES, angles):
        print(f"{name:>15s}: {val: .4f}")
    print("\n可以把下面这行直接拷进 sdog.yaml 的 default_angles：")
    print("default_angles: [")
    for i, v in enumerate(angles):
        end = ",\n" if (i + 1) % 3 == 0 and i != len(angles) - 1 else ", "
        print(f"  {v: .4f}{end}", end="")
    print("]\n")

if __name__ == "__main__":
    main()