# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generator that does nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from .commands_cfg import NullCommandCfg

import torch




from isaaclab.markers import VisualizationMarkers



class NullCommand(CommandTerm):
    """Command generator that does nothing.

    This command generator does not generate any commands. It is used for environments that do not
    require any commands.
    """

    cfg: NullCommandCfg
    """Configuration for the command generator."""

    def __str__(self) -> str:
        msg = "NullCommand:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self):
        """Null command.

        Raises:
            RuntimeError: No command is generated. Always raises this error.
        """
        raise RuntimeError("NullCommandTerm does not generate any commands.")

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return {}

    def compute(self, dt: float):
        pass

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass



class StepFRToBlockCommand(CommandTerm):
    """FRを単一ブロック上の指定ローカル(ux,uy)へ誘導するための2次元コマンド。
    command.shape = (num_envs, 2) で [ux, uy] （ブロック座標系の上面平面）を出す。
    """

    def __init__(self, cfg: "StepFRToBlockCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        # メトリクス例（任意）
        self.metrics["resample_count"] = torch.zeros(self.num_envs, device=self.device)

        self.env = env

    # --- 必須: コマンド・プロパティ ---
    @property
    def command(self) -> torch.Tensor:
        return self._command

    # --- 必須: 実装フック ---
    def _update_metrics(self):
        # ここでは特に何もしない（任意でログを更新可）
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # [ux, uy] を一様乱数で再サンプル
        lo, hi = self.cfg.local_offset_range
        self._command[env_ids, 0] = torch.empty(len(self._command[env_ids]), device=self.device).uniform_(lo, hi)
        self._command[env_ids, 1] = torch.empty(len(self._command[env_ids]), device=self.device).uniform_(lo, hi)
        # メトリクス更新（任意）
        self.metrics["resample_count"][env_ids] += 1.0

    def _update_command(self):
        # このコマンドは「保持型（保持しておくだけ）」なので更新なし
        # （必要ならここでスムージング等）
        pass

    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.fr_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.fr_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.fr_pose_visualizer.set_visibility(False)
    



    def _debug_vis_callback(self, event):
        # check if robot is initialize

        robot = self.env.scene.articulations["robot"]
        block = self.env.scene.rigid_objects["stone2"]
        if (not robot.is_initialized) or (not block.is_initialized):
            return

        # --- 1) ターゲット姿勢（world）を計算 ---
        # コマンド: [ux, uy]（ブロック座標系の上面ローカル）
        cmd = self.command  # [B,2]
        ux, uy = cmd[..., 0], cmd[..., 1]

        # ブロック中心とyawを取得（単一インスタンスなので [:,0,*] でもOKだが形状違いに配慮）
        def _blk_pos_w():
            p = block.data.root_pos_w
            return p[:, 0, :] if p.ndim == 3 else p  # [B,3]
        def _blk_quat_w():
            q = block.data.root_quat_w
            return q[:, 0, :] if q.ndim == 3 else q  # [B,4] (w,x,y,z)
        def _yaw_from_quat(q):
            w,x,y,z = q.unbind(-1)
            return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        blk_pos_w = _blk_pos_w()              # [B,3]
        yaw_blk   = _yaw_from_quat(_blk_quat_w())  # [B]

        cy, sy = torch.cos(yaw_blk), torch.sin(yaw_blk)
        R = torch.stack(
            [torch.stack([cy, -sy], dim=-1), torch.stack([sy, cy], dim=-1)],
            dim=-2,
        )  # [B,2,2]
        t_xy = torch.stack([ux, uy], dim=-1)                  # [B,2]
        tgt_xy_w = (R @ t_xy.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]  # [B,2]
        tgt_z = blk_pos_w[..., 2] + 0.15   # ブロック天面より少し上に表示（必要なら調整）

        # 目標姿勢（位置+クォータニオン）。ここでは向き=ブロックyaw（ロール/ピッチ0）
        yaw = yaw_blk
        qw = torch.cos(0.5*yaw)
        qx = torch.zeros_like(qw)
        qy = torch.zeros_like(qw)
        qz = torch.sin(0.5*yaw)
        goal_pos = torch.stack([tgt_xy_w[..., 0], tgt_xy_w[..., 1], tgt_z], dim=-1)  # [B,3]
        goal_quat = torch.stack([qw, qx, qy, qz], dim=-1)                             # [B,4]

        # --- 2) FRフットの現在姿勢（world） ---
        fr_name = "FR_foot"
        fr_id = robot.body_names.index(fr_name)
        # ある版では body_link_pose_w[:, idx] が [B,7] で入っています
        if hasattr(robot.data, "body_link_pose_w"):
            fr_pose_w = robot.data.body_link_pose_w[:, fr_id]  # [B,7]
            fr_pos = fr_pose_w[:, :3]
            fr_quat = fr_pose_w[:, 3:7]
        else:
            fr_pos = robot.data.body_pos_w[:, fr_id, :3]       # [B,3]
            # クォータニオン名は版差あり（body_quat_w / body_orient_w など）
            fr_quat = getattr(robot.data, "body_quat_w", robot.data.body_orient_w)[:, fr_id, :4]  # [B,4]

        # --- 3) 可視化呼び出し ---
        self.goal_pose_visualizer.visualize(goal_pos, goal_quat)
        self.fr_pose_visualizer.visualize(fr_pos, fr_quat)



