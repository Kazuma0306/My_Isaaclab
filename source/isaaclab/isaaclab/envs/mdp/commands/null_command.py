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










from typing import Sequence, Dict, Tuple



LEG_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# ---- math utils (wxyz) ----
def _yaw_from_quat_wxyz(q):  # [B,4]
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def _rot2d(yaw):             # [B,2,2]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([cy, -sy], -1),
                        torch.stack([sy,  cy], -1)], -2)

def _rot3_from_quat_wxyz(q): # [B,3,3]
    w,x,y,z = q.unbind(-1)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    r00 = 1 - 2*(yy+zz); r01 = 2*(xy-wz);   r02 = 2*(xz+wy)
    r10 = 2*(xy+wz);     r11 = 1 - 2*(xx+zz); r12 = 2*(yz-wx)
    r20 = 2*(xz-wy);     r21 = 2*(yz+wx);   r22 = 1 - 2*(xx+yy)
    return torch.stack([torch.stack([r00,r01,r02], -1),
                        torch.stack([r10,r11,r12], -1),
                        torch.stack([r20,r21,r22], -1)], -2)

def _rigid_pos_quat_w(robj):
    p = robj.data.root_pos_w
    q = robj.data.root_quat_w
    pos = p[:,0,:] if p.ndim==3 else p         # [B,3]
    quat= q[:,0,:] if q.ndim==3 else q         # [B,4] wxyz
    return pos, quat



def _quat_from_yaw(yaw):  # yaw→(wxyz)
    qw = torch.cos(0.5*yaw)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    qz = torch.sin(0.5*yaw)
    return torch.stack([qw,qx,qy,qz], dim=-1)



class MultiLegBaseCommand(CommandTerm):
    """
    出力: command[num_envs, 12] = [FL(xb,yb,zb), FR(...), RL(...), RR(...)] （すべてベース座標）
    - FR: ブロック 'fr_support_key' のローカル [ux,uy] を block->world->base へ
    - FL/RL/RR: 地形の「ワールド矩形」から [xw,yw] をサンプル → (zw は固定 or 関数) → world->base
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        B, dev = self.num_envs, self.device

        self._command = torch.zeros(B, 3*len(LEG_ORDER), device=dev)
        # FR: ブロック上のローカル[ux,uy]
        self._fr_local = torch.zeros(B, 2, device=dev)

        # ---- 設定 ----
        self.fr_support_key: str = getattr(cfg, "fr_support_key", "stone2")
        self.block_local_offset_range: Tuple[float,float] = getattr(cfg, "block_local_offset_range", (-0.08, 0.08))
        self.block_top_offset: float = getattr(cfg, "block_top_offset", 0.15)

        # 他脚：ワールド矩形定義（env原点ローカルで与え、originを足してワールドへ）
        # 例: {"FL_foot": {"center_xy":(0.30, 0.20), "half":(0.06,0.06), "top_z":0.01}, ...}
        self.ped_areas: Dict[str, Dict] = getattr(cfg, "ped_areas", {
            "FL_foot": {"center_xy": (0.30,  0.20), "half": (0.06,0.06), "top_z": 0.01},
            "RL_foot": {"center_xy": (-0.10,  0.20), "half": (0.06,0.06), "top_z": 0.01},
            "RR_foot": {"center_xy": (-0.10, -0.20), "half": (0.06,0.06), "top_z": 0.01},
        })
        # top_z を一定値でなく地形高さ関数で決めたい場合は、ここに関数を差す:
        # self.height_fn: callable | None = your_height_function  # (B,2)->(B,)
        self.height_fn = None


        self._goal_markers: Dict[str, VisualizationMarkers] = {}
        self._foot_markers: Dict[str, VisualizationMarkers] = {}



        self._resample_command(range(B))

    

    # --- 必須: 実装フック ---
    def _update_metrics(self):
        # ここでは特に何もしない（任意でログを更新可）
        pass



    @property
    def command(self) -> torch.Tensor:
        return self._command

    # --------- サンプリング ---------
    def _sample_square(self, n, lo, hi):
        x = torch.empty(n, device=self.device).uniform_(lo, hi)
        y = torch.empty(n, device=self.device).uniform_(lo, hi)
        return torch.stack([x,y], -1)

   

    def _resample_command(self, env_ids=None):
        """
        env_ids:
        - None -> 全env
        - list / range / torch.Tensor
        """
        dev = self.device

        # IsaacLab の他コマンドに合わせて、None なら全 env
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=dev, dtype=torch.long)
        else:
            # list や range の場合は 1D long tensor に変換
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)

        blo_lo, blo_hi = self.block_local_offset_range
        self._fr_local[env_ids] = self._sample_square(env_ids.numel(), blo_lo, blo_hi)

        self._update_command()

        self.metrics.setdefault("resample_count", torch.zeros(self.num_envs, device=dev))
        self.metrics["resample_count"][env_ids] += 1.0


    # --------- 更新（world→base 変換を含む）---------
    def _update_command(self):
        B, dev = self.num_envs, self.device
        scene = self.env.scene

        # ベース姿勢
        robot = scene.articulations["robot"]
        base_p = robot.data.root_pos_w                  # [B,3]
        base_q = robot.data.root_quat_w                 # [B,4] wxyz
        R_wb3  = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # world->base

        # 各envの原点（envごとに台座中心をずらす場合に必要）
        # IsaacLab には env.origins 相当があるはず。無ければ (0,0,0) を使う
        origins = getattr(scene, "env_origins", torch.zeros(B,3, device=dev))
        o_xy = origins[...,:2]  # [B,2]

        # ---- FR: ブロック剛体 → world → base
        block = scene.rigid_objects[self.fr_support_key]
        blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)  # [B,3], [B,4]
        R_bw2 = _rot2d(_yaw_from_quat_wxyz(blk_quat_w))   # block->world (XY)
        t_fr_xy_w = (R_bw2 @ self._fr_local.unsqueeze(-1)).squeeze(-1) + blk_pos_w[...,:2]
        t_fr_z_w  = blk_pos_w[...,2] + self.block_top_offset
        t_fr_w    = torch.cat([t_fr_xy_w, t_fr_z_w.unsqueeze(-1)], -1)         # [B,3]
        fr_b      = (R_wb3 @ (t_fr_w - base_p).unsqueeze(-1)).squeeze(-1)      # [B,3]

        # ---- 他脚：地形のワールド矩形 → base
        leg_targets_b = {"FR_foot": fr_b}
        for leg in ("FL_foot","RL_foot","RR_foot"):
            spec = self.ped_areas[leg]
            c_xy = torch.as_tensor(spec["center_xy"], device=dev, dtype=base_p.dtype)  # [2]
            hx, hy = spec["half"]
            # env原点を考慮したワールド中心
            ctr = o_xy + c_xy  # [B,2]
            # 矩形内からサンプル（固定したいなら _update_command ではサンプルせず _resample で保持）
            # 固定にしたい場合は以下2行を「ゼロ（0,0）」にして ctr をそのまま使う
            # off = self._sample_square(B, -hx, hx)  # [B,2]
            t_xy_w = ctr #+ off                     # [B,2]

            if self.height_fn is not None:
                t_z_w = self.height_fn(t_xy_w) + 0.0  # 必要なら+クリアランス
            else:
                # 地形が静的で高さが既知なら固定値でもOK（例: 0.0 や spec["top_z"]）
                t_z_w = torch.full((B,), float(spec.get("top_z", 0.0)), device=dev, dtype=base_p.dtype)

            t_w = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)                # [B,3]
            leg_targets_b[leg] = (R_wb3 @ (t_w - base_p).unsqueeze(-1)).squeeze(-1)

        # 出力（FL,FR,RL,RR）order
        out = torch.zeros(B, 3*len(LEG_ORDER), device=dev, dtype=base_p.dtype)
        for i, leg in enumerate(LEG_ORDER):
            out[:, 3*i:3*(i+1)] = leg_targets_b[leg]
        self._command = out  # [B,12]

    


    
    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     self._debug = debug_vis

    #     # ★ 親 __init__ から先に呼ばれても落ちないようにする
    #     if not hasattr(self, "_goal_markers"):
    #         self._goal_markers: Dict[str, VisualizationMarkers] = {}
    #     if not hasattr(self, "_foot_markers"):
    #         self._foot_markers: Dict[str, VisualizationMarkers] = {}

    #     if not debug_vis:
    #         for d in (self._goal_markers, self._foot_markers):
    #             for m in d.values():
    #                 m.set_visibility(False)
    #         return

    #     # ★ 初回作成
    #     if not self._goal_markers:
    #         for leg in LEG_ORDER:
    #             self._goal_markers[leg] = VisualizationMarkers(
    #                 self.cfg.goal_pose_visualizer_cfg
    #             )
    #             self._foot_markers[leg] = VisualizationMarkers(
    #                 self.cfg.feet_pose_visualizer_cfg
    #             )

    #     for d in (self._goal_markers, self._foot_markers):
    #         for m in d.values():
    #             m.set_visibility(True)


    
    def _set_debug_vis_impl(self, debug_vis: bool):
        self._debug = debug_vis

        # 親 __init__ から呼ばれたときでも安全なように、必ず dict を持たせる
        if not hasattr(self, "_goal_markers"):
            self._goal_markers: Dict[str, VisualizationMarkers] = {}
        if not hasattr(self, "_foot_markers"):
            self._foot_markers: Dict[str, VisualizationMarkers] = {}

        # 可視化 OFF → あるものは全部非表示にして終了
        if not debug_vis:
            for d in (self._goal_markers, self._foot_markers):
                for m in d.values():
                    m.set_visibility(False)
            return

        # 可視化 ON → 全ての脚について marker を「存在させる」
        for leg in LEG_ORDER:
            if leg not in self._goal_markers:
                self._goal_markers[leg] = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            if leg not in self._foot_markers:
                self._foot_markers[leg] = VisualizationMarkers(
                    self.cfg.feet_pose_visualizer_cfg
                )

        # ON にしたタイミングで全部可視化
        for d in (self._goal_markers, self._foot_markers):
            for m in d.values():
                m.set_visibility(True)
    



    def _debug_vis_callback(self, event):
        # check if robot is initialize

        # markers がまだなければここで必ず作る
        if not hasattr(self, "_goal_markers") or not self._goal_markers:
            self._set_debug_vis_impl(True)
            

        robot = self.env.scene.articulations["robot"]
        block = self.env.scene.rigid_objects["stone2"]
        if (not robot.is_initialized) or (not block.is_initialized):
            return

        # --- 1) ターゲット姿勢（world）を計算 ---




        B, dev = self.num_envs, self.device

        # ---- FR 目標 (world) ----
        blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)      # [B,3], [B,4](wxyz)
        yaw_blk = _yaw_from_quat_wxyz(blk_quat_w)
        R_bw2   = _rot2d(yaw_blk)
        t_fr_xy_w = (R_bw2 @ self._fr_local.unsqueeze(-1)).squeeze(-1) + blk_pos_w[...,:2]
        t_fr_z_w  = blk_pos_w[...,2] + self.block_top_offset
        goal_fr_w = torch.cat([t_fr_xy_w, t_fr_z_w.unsqueeze(-1)], -1)   # [B,3]
        goal_fr_q = _quat_from_yaw(yaw_blk)                              # [B,4]（向き＝ブロックyaw）

        # ---- 他脚 目標 (world) ----
        origins = getattr(self.env.scene, "env_origins", torch.zeros(B,3, device=dev))
        o_xy = origins[...,:2]
        goal_w = {"FR_foot": (goal_fr_w, goal_fr_q)}
        for leg in ("FL_foot","RL_foot","RR_foot"):
            spec = self.ped_areas[leg]
            xy_env = torch.as_tensor(spec["center_xy"], device=dev, dtype=goal_fr_w.dtype)
            t_xy_w = o_xy + xy_env
            t_z_w  = torch.full((B,), float(spec.get("top_z", 0.0)), device=dev, dtype=goal_fr_w.dtype)
            pos = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)
            quat= torch.tensor([1.0,0.0,0.0,0.0], device=dev, dtype=goal_fr_w.dtype).expand(B,4)  # 無回転
            goal_w[leg] = (pos, quat)


        # ---- 足先 現在値 (world) ----
        def _foot_pose_w(leg_name):
            idx = robot.body_names.index(leg_name)
            if hasattr(robot.data, "body_link_pose_w"):
                pose = robot.data.body_link_pose_w[:, idx]  # [B,7]
                return pose[:, :3], pose[:, 3:7]
            else:
                pos = robot.data.body_pos_w[:, idx, :3]
                quat = getattr(robot.data, "body_quat_w",
                               getattr(robot.data, "body_orient_w"))[:, idx, :4]
                return pos, quat

        # ---- 描画 ----
        for leg in LEG_ORDER:
            gp, gq = goal_w[leg]        # [B,3], [B,4]
            fp, fq = _foot_pose_w(leg)  # [B,3], [B,4]
            self._goal_markers[leg].visualize(gp, gq)
            self._foot_markers[leg].visualize(fp, fq)
    



