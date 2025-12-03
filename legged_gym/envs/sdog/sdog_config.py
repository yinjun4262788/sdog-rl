from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SDOGRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5 ] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,   # [rad]
            'RL_hip_joint': 0,   # [rad]
            'FR_hip_joint': 0 ,  # [rad]
            'RR_hip_joint': 0,   # [rad]

            'FL_thigh_joint': 0,     # [rad]
            'RL_thigh_joint': 0,   # [rad]
            'FR_thigh_joint': 0,     # [rad]
            'RR_thigh_joint': 0,   # [rad]

            'FL_calf_joint': 0,   # [rad]
            'RL_calf_joint': 0,    # [rad]
            'FR_calf_joint': 0,  # [rad]
            'RR_calf_joint': 0,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # 新增 commands 类来覆盖默认配置
    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0] # 保持 X 轴移动能力
            lin_vel_y = [0.0, 0.0]  # 禁止 Y 轴移动
            ang_vel_yaw = [0.0, 0.0]# 禁止旋转
            heading = [0.0, 0.0]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/sdog/urdf/sdogHP.urdf'
        name = "sdog"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.45
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            orientation = 0.
            torques = -0.00001
            dof_vel = 0.
            dof_acc = -2.5e-7
            base_height = 0. 
            collision = 0.
            feet_stumble = 0. 
            action_rate = 0
            stand_still = 0.
            

class SDOGRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_sdog'

  
