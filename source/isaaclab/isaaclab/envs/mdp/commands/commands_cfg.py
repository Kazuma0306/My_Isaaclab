# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .null_command import NullCommand, StepFRToBlockCommand, MultiLegBaseCommand, MultiLegBaseCommand2, MultiLegBaseCommand3, FootstepFromHighLevel
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""















# --- 設定クラス: CommandTermCfg を継承 ---
@configclass
class StepFRToBlockCommandCfg(CommandTermCfg):
    """FR用 単一ブロックターゲット (ux, uy) を出すコマンド設定"""
    class_type: type = StepFRToBlockCommand
    # リサンプリング周期 [s]
    resampling_time_range: tuple[float, float] = (2.0, 3.0)
    debug_vis: bool = False
    # ブロック中心からのローカル・オフセット範囲（m）
    local_offset_range: tuple[float, float] = (-0.05, 0.05)

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)







# --- 設定クラス: CommandTermCfg を継承 ---
@configclass
class MultiLegBaseCommandCfg(CommandTermCfg):

    """各脚のベース座標系ターゲットを出すコマンド。
       - FR: ブロックのローカル (ux,uy) をサンプル→World→Base に変換
       - FL/RL/RR: ベース座標の固定ターゲット
    """
    # 紐づけるコマンドクラス（実装側で MultiLegBaseTargetsCommand を用意）
    class_type: type = MultiLegBaseCommand

    # リサンプリング周期 [s]
    resampling_time_range: tuple[float, float] = (2.0, 3.0)
    debug_vis: bool = True

    # --- FR = ブロックから生成するターゲット ---
    block_name: str = "stone2"
    fr_local_offset_range: tuple[float, float] = (-0.05, 0.05)  # ブロック上面ローカル (ux,uy) の一様分布
    block_top_offset: float = 0.15  # ブロック上面からのZオフセット（可視化/誘導用）

    # --- 他脚の固定ターゲット（ベース座標系, 単位[m]）---
    #   例は GO2 くらいの寸法を想定。実機/モデルに合わせて調整してください。
    fixed_targets_b: dict[str, tuple[float, float, float]] = {
        "FL_foot": ( 0.25,  0.18, 0.0),
        "RL_foot": (-0.25,  0.18, 0.0),
        "RR_foot": (-0.25, -0.18, 0.0),
    }

    # ロボットの脚ボディ名（順序は環境に合わせて）
    leg_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    # --- 可視化（Wxyz で渡す）---
    # FR の目標姿勢を表示
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # 各脚ターゲットの表示（小さめ）
    feet_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    feet_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)



# --- 設定クラス: CommandTermCfg を継承 ---
@configclass
class MultiLegBaseCommand2Cfg(CommandTermCfg):

    """各脚のベース座標系ターゲットを出すコマンド。
       - FR: ブロックのローカル (ux,uy) をサンプル→World→Base に変換
       - FL/RL/RR: ベース座標の固定ターゲット
    """
    # 紐づけるコマンドクラス（実装側で MultiLegBaseTargetsCommand を用意）
    class_type: type = MultiLegBaseCommand2

    # リサンプリング周期 [s]
    resampling_time_range: tuple[float, float] = (2.0, 3.0)
    debug_vis: bool = True

    
    # --- 可視化（Wxyz で渡す）---
    # FR の目標姿勢を表示
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # 各脚ターゲットの表示（小さめ）
    feet_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    feet_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)



# --- 設定クラス: CommandTermCfg を継承 ---
@configclass
class MultiLegBaseCommand3Cfg(CommandTermCfg):

    """各脚のベース座標系ターゲットを出すコマンド。
       - FR: ブロックのローカル (ux,uy) をサンプル→World→Base に変換
       - FL/RL/RR: ベース座標の固定ターゲット
    """
    # 紐づけるコマンドクラス（実装側で MultiLegBaseTargetsCommand を用意）
    class_type: type = MultiLegBaseCommand3

    # リサンプリング周期 [s]
    resampling_time_range: tuple[float, float] = (2.0, 3.0)
    debug_vis: bool = True

    
    # --- 可視化（Wxyz で渡す）---
    # FR の目標姿勢を表示
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # 各脚ターゲットの表示（小さめ）
    feet_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    feet_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)


    params=dict(
    # phase_block_keys={
    #     "0": dict(FL_foot="stone3", FR_foot="stone6", RL_foot="stone4", RR_foot="stone5"),
    #     "1": dict(FL_foot="stone3", FR_foot="stone2", RL_foot="stone4", RR_foot="stone6"),
    # },

    phase_block_keys = {
    # --- Phase 0: 初期状態 (全足が初期ブロックに乗っている) ---
    "0": {
        "FL_foot": "stone5",
        "FR_foot": "stone6",  # 右前: 現在地
        "RL_foot": "stone7",
        "RR_foot": "stone8",  # 右後: 現在地
    },
    
    # --- Phase 1: 右前足 (FR) だけ前に出す ---
    # 右後ろ足 (RR) のターゲットはまだ "stone5" (現在地) のままにしておくのが重要です！
    "1": {
        "FL_foot": "stone5",
        "FR_foot": "stone2",  # ★右前: 新しいブロックへ！
        "RL_foot": "stone7",
        "RR_foot": "stone8",  # 右後: まだ動かない（支えになる）
    },

    # --- Phase 2: 右後ろ足 (RR) を前に出す ---
    # 右前足 (FR) が着地したあと、遅れて右後ろ足を動かします
    # 通常、右後ろ足は「さっき右前足があった場所 (stone6)」などを狙うことが多いです
    "2": {
        "FL_foot": "stone5",
        "FR_foot": "stone2",  # 右前: 既に移動済み
        "RL_foot": "stone3",
        "RR_foot": "stone8",  # ★右後: 右前が居た場所へ移動！
    },

    "3": {
        "FL_foot": "stone1",
        "FR_foot": "stone2",  # 右前: 既に移動済み
        "RL_foot": "stone3",
        "RR_foot": "stone8",  # ★右後: 右前が居た場所へ移動！
    },

    "4": {
        "FL_foot": "stone1",
        "FR_foot": "stone2",  # 右前: 既に移動済み
        "RL_foot": "stone3",
        "RR_foot": "stone4",  # ★右後: 右前が居た場所へ移動！
    },


    "5": {
        "FL_foot": "stone1",
        "FR_foot": "stone10",  
        "RL_foot": "stone3",
        "RR_foot": "stone4",  
    },

    "6": {
        "FL_foot": "stone1",
        "FR_foot": "stone10",  
        "RL_foot": "stone5",
        "RR_foot": "stone4",  
    },

    "7": {
        "FL_foot": "stone9",
        "FR_foot": "stone10",  # 右前: 既に移動済み
        "RL_foot": "stone5",
        "RR_foot": "stone4",  # ★右後: 右前が居た場所へ移動！
    },

    "8": {
        "FL_foot": "stone9",
        "FR_foot": "stone10",  # 右前: 既に移動済み
        "RL_foot": "stone5",
        "RR_foot": "stone6",  # ★右後: 右前が居た場所へ移動！
    },


    }
)




#For High layer

@configclass
class FootstepFromHighLevelCfg(CommandTermCfg):
    # class_type: typeFor High layer[CommandTerm] = FootstepFromHighLevel

    class_type: type = FootstepFromHighLevel

    command_dim: int = 12   # 4 脚 x 3 (xyz)
    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # ほぼ resample しない