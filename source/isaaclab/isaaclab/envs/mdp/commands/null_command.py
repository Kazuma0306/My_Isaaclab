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



# class MultiLegBaseCommand(CommandTerm):
#     """
#     出力: command[num_envs, 12] = [FL(xb,yb,zb), FR(...), RL(...), RR(...)] （すべてベース座標）
#     - FR: ブロック 'fr_support_key' のローカル [ux,uy] を block->world->base へ
#     - FL/RL/RR: 地形の「ワールド矩形」から [xw,yw] をサンプル → (zw は固定 or 関数) → world->base
#     """
#     def __init__(self, cfg, env):
#         super().__init__(cfg, env)
#         self.env = env
#         B, dev = self.num_envs, self.device

#         self._command = torch.zeros(B, 3*len(LEG_ORDER), device=dev)
#         # FR: ブロック上のローカル[ux,uy]
#         self._fr_local = torch.zeros(B, 2, device=dev)

#         # ---- 設定 ----
#         self.fr_support_key: str = getattr(cfg, "fr_support_key", "stone2")
#         self.block_local_offset_range: Tuple[float,float] = getattr(cfg, "block_local_offset_range", (-0.08, 0.08))
#         self.block_top_offset: float = getattr(cfg, "block_top_offset", 0.15)

#         # 他脚：ワールド矩形定義（env原点ローカルで与え、originを足してワールドへ）
#         # 例: {"FL_foot": {"center_xy":(0.30, 0.20), "half":(0.06,0.06), "top_z":0.01}, ...}
#         self.ped_areas: Dict[str, Dict] = getattr(cfg, "ped_areas", {
#             "FL_foot": {"center_xy": (0.25,  0.15), "half": (0.06,0.06), "top_z": 0.01},
#             "RL_foot": {"center_xy": (-0.15,  0.15), "half": (0.06,0.06), "top_z": 0.01},
#             "RR_foot": {"center_xy": (-0.15, -0.15), "half": (0.06,0.06), "top_z": 0.01},
#         })
#         # top_z を一定値でなく地形高さ関数で決めたい場合は、ここに関数を差す:
#         # self.height_fn: callable | None = your_height_function  # (B,2)->(B,)
#         self.height_fn = None


#         self._goal_markers: Dict[str, VisualizationMarkers] = {}
#         self._foot_markers: Dict[str, VisualizationMarkers] = {}



#         self._resample_command(range(B))

    

#     # --- 必須: 実装フック ---
#     def _update_metrics(self):
#         # ここでは特に何もしない（任意でログを更新可）
#         pass



#     @property
#     def command(self) -> torch.Tensor:
#         return self._command

#     # --------- サンプリング ---------
#     def _sample_square(self, n, lo, hi):
#         x = torch.empty(n, device=self.device).uniform_(lo, hi)
#         y = torch.empty(n, device=self.device).uniform_(lo, hi)
#         return torch.stack([x,y], -1)

   

#     def _resample_command(self, env_ids=None):
#         """
#         env_ids:
#         - None -> 全env
#         - list / range / torch.Tensor
#         """
#         dev = self.device

#         # IsaacLab の他コマンドに合わせて、None なら全 env
#         if env_ids is None:
#             env_ids = torch.arange(self.num_envs, device=dev, dtype=torch.long)
#         else:
#             # list や range の場合は 1D long tensor に変換
#             if not isinstance(env_ids, torch.Tensor):
#                 env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)

#         blo_lo, blo_hi = self.block_local_offset_range
#         self._fr_local[env_ids] = self._sample_square(env_ids.numel(), blo_lo, blo_hi)

#         self._update_command()

#         self.metrics.setdefault("resample_count", torch.zeros(self.num_envs, device=dev))
#         self.metrics["resample_count"][env_ids] += 1.0


#     # --------- 更新（world→base 変換を含む）---------
#     def _update_command(self):
#         B, dev = self.num_envs, self.device
#         scene = self.env.scene

#         # ベース姿勢
#         robot = scene.articulations["robot"]
#         base_p = robot.data.root_pos_w                  # [B,3]
#         base_q = robot.data.root_quat_w                 # [B,4] wxyz
#         R_wb3  = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # world->base

#         # 各envの原点（envごとに台座中心をずらす場合に必要）
#         # IsaacLab には env.origins 相当があるはず。無ければ (0,0,0) を使う
#         origins = getattr(scene, "env_origins", torch.zeros(B,3, device=dev))
#         o_xy = origins[...,:2]  # [B,2]

#         # ---- FR: ブロック剛体 → world → base
#         block = scene.rigid_objects[self.fr_support_key]
#         blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)  # [B,3], [B,4]
#         R_bw2 = _rot2d(_yaw_from_quat_wxyz(blk_quat_w))   # block->world (XY)
#         t_fr_xy_w = (R_bw2 @ self._fr_local.unsqueeze(-1)).squeeze(-1) + blk_pos_w[...,:2]
#         t_fr_z_w  = blk_pos_w[...,2] + self.block_top_offset
#         t_fr_w    = torch.cat([t_fr_xy_w, t_fr_z_w.unsqueeze(-1)], -1)         # [B,3]
#         fr_b      = (R_wb3 @ (t_fr_w - base_p).unsqueeze(-1)).squeeze(-1)      # [B,3]

#         # ---- 他脚：地形のワールド矩形 → base
#         leg_targets_b = {"FR_foot": fr_b}
#         for leg in ("FL_foot","RL_foot","RR_foot"):
#             spec = self.ped_areas[leg]
#             c_xy = torch.as_tensor(spec["center_xy"], device=dev, dtype=base_p.dtype)  # [2]
#             hx, hy = spec["half"]
#             # env原点を考慮したワールド中心
#             ctr = o_xy + c_xy  # [B,2]
#             # 矩形内からサンプル（固定したいなら _update_command ではサンプルせず _resample で保持）
#             # 固定にしたい場合は以下2行を「ゼロ（0,0）」にして ctr をそのまま使う
#             # off = self._sample_square(B, -hx, hx)  # [B,2]
#             t_xy_w = ctr #+ off                     # [B,2]

#             if self.height_fn is not None:
#                 t_z_w = self.height_fn(t_xy_w) + 0.0  # 必要なら+クリアランス
#             else:
#                 # 地形が静的で高さが既知なら固定値でもOK（例: 0.0 や spec["top_z"]）
#                 t_z_w = torch.full((B,), float(spec.get("top_z", 0.0)), device=dev, dtype=base_p.dtype)

#             t_w = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)                # [B,3]
#             leg_targets_b[leg] = (R_wb3 @ (t_w - base_p).unsqueeze(-1)).squeeze(-1)

#         # 出力（FL,FR,RL,RR）order
#         out = torch.zeros(B, 3*len(LEG_ORDER), device=dev, dtype=base_p.dtype)
#         for i, leg in enumerate(LEG_ORDER):
#             out[:, 3*i:3*(i+1)] = leg_targets_b[leg]
#         self._command = out  # [B,12]



    
#     def _set_debug_vis_impl(self, debug_vis: bool):
#         self._debug = debug_vis

#         # 親 __init__ から呼ばれたときでも安全なように、必ず dict を持たせる
#         if not hasattr(self, "_goal_markers"):
#             self._goal_markers: Dict[str, VisualizationMarkers] = {}
#         if not hasattr(self, "_foot_markers"):
#             self._foot_markers: Dict[str, VisualizationMarkers] = {}

#         # 可視化 OFF → あるものは全部非表示にして終了
#         if not debug_vis:
#             for d in (self._goal_markers, self._foot_markers):
#                 for m in d.values():
#                     m.set_visibility(False)
#             return

#         # 可視化 ON → 全ての脚について marker を「存在させる」
#         for leg in LEG_ORDER:
#             if leg not in self._goal_markers:
#                 self._goal_markers[leg] = VisualizationMarkers(
#                     self.cfg.goal_pose_visualizer_cfg
#                 )
#             if leg not in self._foot_markers:
#                 self._foot_markers[leg] = VisualizationMarkers(
#                     self.cfg.feet_pose_visualizer_cfg
#                 )

#         # ON にしたタイミングで全部可視化
#         for d in (self._goal_markers, self._foot_markers):
#             for m in d.values():
#                 m.set_visibility(True)
    



#     def _debug_vis_callback(self, event):
#         # check if robot is initialize

#         # markers がまだなければここで必ず作る
#         if not hasattr(self, "_goal_markers") or not self._goal_markers:
#             self._set_debug_vis_impl(True)
            

#         robot = self.env.scene.articulations["robot"]
#         block = self.env.scene.rigid_objects["stone2"]
#         if (not robot.is_initialized) or (not block.is_initialized):
#             return

#         # --- 1) ターゲット姿勢（world）を計算 ---




#         B, dev = self.num_envs, self.device

#         # ---- FR 目標 (world) ----
#         blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)      # [B,3], [B,4](wxyz)
#         yaw_blk = _yaw_from_quat_wxyz(blk_quat_w)
#         R_bw2   = _rot2d(yaw_blk)
#         t_fr_xy_w = (R_bw2 @ self._fr_local.unsqueeze(-1)).squeeze(-1) + blk_pos_w[...,:2]
#         t_fr_z_w  = blk_pos_w[...,2] + self.block_top_offset
#         goal_fr_w = torch.cat([t_fr_xy_w, t_fr_z_w.unsqueeze(-1)], -1)   # [B,3]
#         goal_fr_q = _quat_from_yaw(yaw_blk)                              # [B,4]（向き＝ブロックyaw）

#         # ---- 他脚 目標 (world) ----
#         origins = getattr(self.env.scene, "env_origins", torch.zeros(B,3, device=dev))
#         o_xy = origins[...,:2]
#         goal_w = {"FR_foot": (goal_fr_w, goal_fr_q)}
#         for leg in ("FL_foot","RL_foot","RR_foot"):
#             spec = self.ped_areas[leg]
#             xy_env = torch.as_tensor(spec["center_xy"], device=dev, dtype=goal_fr_w.dtype)
#             t_xy_w = o_xy + xy_env
#             t_z_w  = torch.full((B,), float(spec.get("top_z", 0.0)), device=dev, dtype=goal_fr_w.dtype)
#             pos = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)
#             quat= torch.tensor([1.0,0.0,0.0,0.0], device=dev, dtype=goal_fr_w.dtype).expand(B,4)  # 無回転
#             goal_w[leg] = (pos, quat)


#         # ---- 足先 現在値 (world) ----
#         def _foot_pose_w(leg_name):
#             idx = robot.body_names.index(leg_name)
#             if hasattr(robot.data, "body_link_pose_w"):
#                 pose = robot.data.body_link_pose_w[:, idx]  # [B,7]
#                 return pose[:, :3], pose[:, 3:7]
#             else:
#                 pos = robot.data.body_pos_w[:, idx, :3]
#                 quat = getattr(robot.data, "body_quat_w",
#                                getattr(robot.data, "body_orient_w"))[:, idx, :4]
#                 return pos, quat

#         # ---- 描画 ----
#         for leg in LEG_ORDER:
#             gp, gq = goal_w[leg]        # [B,3], [B,4]
#             fp, fq = _foot_pose_w(leg)  # [B,3], [B,4]
#             self._goal_markers[leg].visualize(gp, gq)
#             self._foot_markers[leg].visualize(fp, fq)
    





class MultiLegBaseCommand2(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        B, dev = self.num_envs, self.device

        self._command = torch.zeros(B, 3*len(LEG_ORDER), device=dev)

        # ---- 設定 ----
        # ブロック上のローカル [ux,uy] の範囲と高さオフセット（全脚共通）
        self.block_local_offset_range: Tuple[float,float] = getattr(
            cfg, "block_local_offset_range", (-0.08, 0.08)
        )
        self.block_top_offset: float = getattr(cfg, "block_top_offset", 0.15)

        # ★ 脚ごとの対応ブロック名
        #   Stone3/4/5/6 はシーン側の rigid_objects のキーに合わせて適宜変えてください
        self.leg_block_keys: Dict[str, str] = getattr(cfg, "leg_block_keys", {
            "FL_foot": "stone3",
            "FR_foot": "stone6",
            "RL_foot": "stone4",
            "RR_foot": "stone5",
        })

        # ★ 各脚のブロックローカル座標 [ux, uy]
        self._local_xy: Dict[str, torch.Tensor] = {
            leg: torch.zeros(B, 2, device=dev) for leg in LEG_ORDER
        }

        # （もう ped_areas/height_fn は使わないなら削除してOK）
        self.height_fn = None

        self._goal_markers: Dict[str, VisualizationMarkers] = {}
        self._foot_markers: Dict[str, VisualizationMarkers] = {}

        # 初回サンプル
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
        dev = self.device

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=dev, dtype=torch.long)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)

        blo_lo, blo_hi = self.block_local_offset_range

        # ★ 全脚分のブロックローカル座標をサンプル
        for leg in LEG_ORDER:
            self._local_xy[leg][env_ids] = self._sample_square(env_ids.numel(), blo_lo, blo_hi)

        self._update_command()

        self.metrics.setdefault("resample_count", torch.zeros(self.num_envs, device=dev))
        self.metrics["resample_count"][env_ids] += 1.0

    
    def _update_command(self):
        B, dev = self.num_envs, self.device
        scene = self.env.scene

        # ベース姿勢
        robot = scene.articulations["robot"]
        base_p = robot.data.root_pos_w                  # [B,3]
        base_q = robot.data.root_quat_w                 # [B,4] wxyz
        R_wb3  = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # world->base

        leg_targets_b: Dict[str, torch.Tensor] = {}

        # ★ 各脚ごとに対応ブロックから target を計算
        for leg in LEG_ORDER:
            block_name = self.leg_block_keys[leg]
            block = scene.rigid_objects[block_name]

            blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)   # [B,3], [B,4]
            yaw_blk = _yaw_from_quat_wxyz(blk_quat_w)          # [B]
            R_bw2   = _rot2d(yaw_blk)                          # [B,2,2]

            # ローカル [ux,uy] -> world XY
            local_xy = self._local_xy[leg]                     # [B,2]
            t_xy_w = (R_bw2 @ local_xy.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
            t_z_w  = blk_pos_w[..., 2] + self.block_top_offset

            t_w = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], dim=-1)   # [B,3]

            # world -> base
            leg_targets_b[leg] = (R_wb3 @ (t_w - base_p).unsqueeze(-1)).squeeze(-1)  # [B,3]

        # 出力（FL,FR,RL,RR）order
        out = torch.zeros(B, 3*len(LEG_ORDER), device=dev, dtype=base_p.dtype)
        for i, leg in enumerate(LEG_ORDER):
            out[:, 3*i:3*(i+1)] = leg_targets_b[leg]

        self._command = out  # [B,12]

    
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
        # markers がまだなければここで必ず作る
        if not hasattr(self, "_goal_markers") or not self._goal_markers:
            self._set_debug_vis_impl(True)

        robot = self.env.scene.articulations["robot"]

        # 対応ブロックを全部取得
        blocks = {
            name: self.env.scene.rigid_objects[name]
            for name in set(self.leg_block_keys.values())
        }

        # 初期化待ち
        if (not robot.is_initialized) or any(not b.is_initialized for b in blocks.values()):
            return

        B, dev = self.num_envs, self.device

        goal_w: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # ★ 各脚の目標 (world) を計算
        for leg in LEG_ORDER:
            block_name = self.leg_block_keys[leg]
            block = blocks[block_name]

            blk_pos_w, blk_quat_w = _rigid_pos_quat_w(block)      # [B,3], [B,4](wxyz)
            yaw_blk = _yaw_from_quat_wxyz(blk_quat_w)             # [B]
            R_bw2   = _rot2d(yaw_blk)                             # [B,2,2]

            local_xy = self._local_xy[leg]                        # [B,2]
            t_xy_w = (R_bw2 @ local_xy.unsqueeze(-1)).squeeze(-1) + blk_pos_w[..., :2]
            t_z_w  = blk_pos_w[..., 2] + self.block_top_offset

            pos = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], dim=-1)         # [B,3]
            quat = _quat_from_yaw(yaw_blk)                                  # [B,4] ブロックと同じ yaw

            goal_w[leg] = (pos, quat)

        # ---- 足先 現在値 (world) ----
        def _foot_pose_w(leg_name):
            idx = robot.body_names.index(leg_name)
            if hasattr(robot.data, "body_link_pose_w"):
                pose = robot.data.body_link_pose_w[:, idx]  # [B,7]
                return pose[:, :3], pose[:, 3:7]
            else:
                pos = robot.data.body_pos_w[:, idx, :3]
                quat = getattr(
                    robot.data,
                    "body_quat_w",
                    getattr(robot.data, "body_orient_w")
                )[:, idx, :4]
                return pos, quat

        # ---- 描画 ----
        for leg in LEG_ORDER:
            gp, gq = goal_w[leg]        # [B,3], [B,4]
            fp, fq = _foot_pose_w(leg)  # [B,3], [B,4]
            self._goal_markers[leg].visualize(gp, gq)
            self._foot_markers[leg].visualize(fp, fq)










class MultiLegBaseCommand3(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        B, dev = self.num_envs, self.device

        self._command = torch.zeros(B, 3*len(LEG_ORDER), device=dev)

        # ローカル [ux,uy] と高さオフセット
        self.block_local_offset_range: Tuple[float,float] = getattr(
            cfg, "block_local_offset_range", (-0.00, 0.00)
        )
        self.block_top_offset: float = getattr(cfg, "block_top_offset", 0.15)

        # --- 設定の読み込み ---
        P = getattr(cfg, "params", {})
        phase_block_keys_cfg = P.get("phase_block_keys", None)
        leg_block_keys_cfg   = P.get("leg_block_keys", None)

        if phase_block_keys_cfg is None:
            print("Warning: phase_block_keys not found, falling back to single phase.")
            # if leg_block_keys_cfg is None:
            #     leg_block_keys_cfg = {
            #         "FL_foot": "stone3", "FR_foot": "stone6",
            #         "RL_foot": "stone4", "RR_foot": "stone5",
            #     }
            # phase_block_keys_cfg = {0: leg_block_keys_cfg}

        # ★修正1: キーを強制的に int に変換し、誤って文字列キーが入るのを防ぐ
        self.phase_block_keys: dict[int, dict[str, str]] = {
            int(k): v for k, v in phase_block_keys_cfg.items()
        }

        # フェーズ番号
        self.phases = sorted(self.phase_block_keys.keys())
        self.num_phases = len(self.phases)

        # チェック
        for ph, mp in self.phase_block_keys.items():
            for leg in LEG_ORDER:
                if leg not in mp:
                    raise RuntimeError(f"phase_block_keys[{ph}] missing leg '{leg}'")

        # 現在フェーズ
        self.phase = torch.zeros(B, dtype=torch.long, device=dev)

        # ブロック参照キャッシュ
        scene = env.scene
        self.blocks_by_phase = {}
        for ph, mp in self.phase_block_keys.items():
            self.blocks_by_phase[ph] = {
                leg: scene.rigid_objects[blk_name] for leg, blk_name in mp.items()
            }

        self._local_xy = {leg: torch.zeros(B, 2, device=dev) for leg in LEG_ORDER}

        # ★修正2: デバッグ用キャッシュの初期化
        # _debug_vis_callback で計算済みの値を使うために用意
        self._debug_goal_w_cache = {}
        self._debug_goal_quat_cache = {}

        self._goal_markers = {}
        self._foot_markers = {}

        # 初回実行
        self._resample_command(range(B))



    def set_phase(self, env_ids, phase: int):
        dev = self.device
        if phase not in self.phase_block_keys:
            raise RuntimeError(f"phase={phase} not found.")
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        self.phase[env_ids] = phase
        # self._update_command()

        self._resample_command(env_ids)



    
    def advance_phase(self, env_ids):
        dev = self.device
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        self.phase[env_ids] += 1
        self.phase[env_ids] = self.phase[env_ids].clamp(0, self.num_phases - 1)

        # self._update_command()

        self._resample_command(env_ids)

        # 1000番目の環境の情報を代表して表示 (10ステップに1回)
        # if self.env.common_step_counter % 10 == 0:
        env_0_phase = self.phase[0].item()
        print(f"[Check] Phase: {env_0_phase}")

    

    def _update_metrics(self):
        pass




    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _sample_square(self, n, lo, hi):
        x = torch.empty(n, device=self.device).uniform_(lo, hi)
        y = torch.empty(n, device=self.device).uniform_(lo, hi)
        return torch.stack([x, y], -1)

    def _resample_command(self, env_ids=None):
        dev = self.device
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=dev, dtype=torch.long)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)

        blo_lo, blo_hi = self.block_local_offset_range
        for leg in LEG_ORDER:
            self._local_xy[leg][env_ids] = self._sample_square(env_ids.numel(), blo_lo, blo_hi)

        self._update_command()
        
        # metricsへのアクセスがある場合のみ
        if hasattr(self, "metrics"):
            self.metrics.setdefault("resample_count", torch.zeros(self.num_envs, device=dev))
            self.metrics["resample_count"][env_ids] += 1.0

    def _update_command(self):
        # 安全策: phaseが消えていたら再作成
        if (not hasattr(self, "phase")) or (not isinstance(self.phase, torch.Tensor)) \
           or (self.phase.shape[0] != self.num_envs):
            self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        B, dev = self.num_envs, self.device
        scene = self.env.scene
        robot = scene.articulations["robot"]
        
        base_p = robot.data.root_pos_w
        base_q = robot.data.root_quat_w
        R_wb3  = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)

        out = torch.zeros(B, 3*len(LEG_ORDER), device=dev, dtype=base_p.dtype)

        # デバッグキャッシュの準備
        if not self._debug_goal_w_cache:
            for leg in LEG_ORDER:
                self._debug_goal_w_cache[leg] = torch.zeros(B, 3, device=dev)
                self._debug_goal_quat_cache[leg] = torch.zeros(B, 4, device=dev)

        for i, leg in enumerate(LEG_ORDER):
            leg_pos_w  = torch.zeros_like(base_p)
            leg_quat_w = torch.zeros_like(base_q)

            for ph in self.phases:
                # ★ここがエラー箇所でした。
                # ph は int, self.phase も LongTensor なので正常に比較可能になります。
                idxs = torch.nonzero(self.phase == ph, as_tuple=False).squeeze(-1)
                
                if idxs.numel() == 0:
                    continue

                block = self.blocks_by_phase[ph][leg]#フェーズごとに変わるブロック配列を取得、そこから座標を取得
                pos_w, quat_w = _rigid_pos_quat_w(block)

                leg_pos_w[idxs]  = pos_w[idxs]
                leg_quat_w[idxs] = quat_w[idxs]

            # World座標計算
            yaw_blk = _yaw_from_quat_wxyz(leg_quat_w)
            R_bw2   = _rot2d(yaw_blk)
            local_xy = self._local_xy[leg]
            
            t_xy_w = (R_bw2 @ local_xy.unsqueeze(-1)).squeeze(-1) + leg_pos_w[..., :2]
            t_z_w  = leg_pos_w[..., 2] + self.block_top_offset
            t_w = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], dim=-1)

            # ★修正3: 計算したWorld座標を保存（_debug_vis_callback用）
            self._debug_goal_w_cache[leg] = t_w.clone()
            self._debug_goal_quat_cache[leg] = _quat_from_yaw(yaw_blk)

            # Base座標へ変換してコマンド出力
            leg_b = (R_wb3 @ (t_w - base_p).unsqueeze(-1)).squeeze(-1)
            out[:, 3*i:3*(i+1)] = leg_b




            

        self._command = out

    # --- ★修正4: 可視化関数を書き換え ---
    # 古い leg_block_keys 依存を削除し、計算済みキャッシュ(_debug_goal_w_cache)を使用
    def _set_debug_vis_impl(self, debug_vis: bool):
        self._debug = debug_vis
        if not hasattr(self, "_goal_markers"): self._goal_markers = {}
        if not hasattr(self, "_foot_markers"): self._foot_markers = {}

        if not debug_vis:
            for d in (self._goal_markers, self._foot_markers):
                for m in d.values(): m.set_visibility(False)
            return

        for leg in LEG_ORDER:
            if leg not in self._goal_markers:
                self._goal_markers[leg] = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            if leg not in self._foot_markers:
                self._foot_markers[leg] = VisualizationMarkers(self.cfg.feet_pose_visualizer_cfg)

        for d in (self._goal_markers, self._foot_markers):
            for m in d.values(): m.set_visibility(True)




    def _debug_vis_callback(self, event):
        if not hasattr(self, "_goal_markers") or not self._goal_markers:
            self._set_debug_vis_impl(True)
        
        robot = self.env.scene.articulations["robot"]
        if not robot.is_initialized: return

        for leg in LEG_ORDER:
            # update_command で計算して保存した値をここで描画
            gp = self._debug_goal_w_cache.get(leg, None)
            gq = self._debug_goal_quat_cache.get(leg, None)
            
            if gp is not None and gq is not None:
                self._goal_markers[leg].visualize(gp, gq)

            # 足の現在位置
            idx = robot.body_names.index(leg)
            if hasattr(robot.data, "body_link_pose_w"):
                pose = robot.data.body_link_pose_w[:, idx]
                fp, fq = pose[:, :3], pose[:, 3:7]
            else:
                fp = robot.data.body_pos_w[:, idx, :3]
                fq = getattr(robot.data, "body_quat_w", getattr(robot.data, "body_orient_w"))[:, idx, :4]
            
            self._foot_markers[leg].visualize(fp, fq)





from isaaclab.utils.math import quat_apply

class FootstepFromHighLevel(CommandTerm):
    def __init__(self, cfg: FootstepFromHighLevelCfg, env):
        super().__init__(cfg, env)
        self.env = env
        self._command = torch.zeros(self.num_envs, cfg.command_dim, device=self.device)

        # ★修正2: デバッグ用キャッシュの初期化
        # _debug_vis_callback で計算済みの値を使うために用意
        self._debug_goal_w_cache = {}
        self._debug_goal_quat_cache = {}

        self._goal_markers = {}
        self._foot_markers = {}

    @property
    def command(self):
        return self._command

    # 上位から書き換えて使うので、resample/update は何もしないでOK
    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    
    def set_foot_targets_base(self, flat_or_batched: torch.Tensor):
        """
        flat_or_batched: [B,12] または [B,4,3]
        ベース座標系の足ターゲットとして解釈する。
        """
        B = self.num_envs
        dev = self.device

        # --- 1) 入力を [B,4,3] の base座標にそろえる ---
        if flat_or_batched.ndim == 2:
            assert flat_or_batched.shape[1] == 12
            foot_targets_b = flat_or_batched.view(B, 4, 3)
        else:
            assert flat_or_batched.shape == (B, 4, 3)
            foot_targets_b = flat_or_batched

        # ★ ここが超重要：commandには **base座標のまま** を入れる
        self._command[:, :] = foot_targets_b.view(B, -1)

        # --- 2) ここから下は「可視化用に world に直すだけ」 ---

        robot = self.env.scene.articulations["robot"]
        base_pos_w = robot.data.root_pos_w      # [B,3]
        base_quat_w = robot.data.root_quat_w    # [B,4]

        from isaaclab.utils.math import quat_apply

        # base -> world
        vec_b  = foot_targets_b.view(B * 4, 3)       # [B*4,3]
        quat_b = base_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(B * 4, 4)

        vec_w = quat_apply(quat_b, vec_b)            # [B*4,3]
        base_pos_rep = base_pos_w.unsqueeze(1).expand(-1, 4, -1)  # [B,4,3]
        targets_w = vec_w.view(B, 4, 3) + base_pos_rep            # [B,4,3]

        # デバッグ用キャッシュ (world)
        for leg_i, leg in enumerate(LEG_ORDER):
            gp = targets_w[:, leg_i, :]                      # [B,3]
            gq = torch.zeros(B, 4, device=dev); gq[:, 3] = 1
            self._debug_goal_w_cache[leg] = gp
            self._debug_goal_quat_cache[leg] = gq

    # def set_foot_targets_base(self, foot_targets_b):
    #     """
    #     上位ポリシーが出した足置き目標を base 座標系で受け取る。
    #     foot_targets_b: [num_envs, 4, 3]  (FR, FL, RR, RL)
    #     """
    #     B = self.num_envs
    #     # assert foot_targets_b.shape == (B, len(LEG_ORDER), 3)

    #     # 1) そのまま command に突っ込む（下位への観測用）
    #     #    （自分のエンコード仕様に合わせて書き換えてOK）
    #     self._command[:] = foot_targets_b

    #     # 2) デバッグ用にワールド座標へ変換してキャッシュ
    #     robot = self.env.scene.articulations["robot"]
    #     base_pos_w = robot.data.root_state_w[:, 0:3]   # [B,3]
    #     base_quat_w = robot.data.root_state_w[:, 3:7]  # [B,4]

    #     # base -> world
    #     # foot_targets_b: [B,4,3]
        
    #     # targets_w = quat_apply(base_quat_w.unsqueeze(1), foot_targets_b) \
    #     #             + base_pos_w.unsqueeze(1)           # [B,4,3]

    #     # # 各脚ごとにキャッシュ
    #     # for leg_i, leg in enumerate(LEG_ORDER):
    #     #     gp = targets_w[:, leg_i, :]        # [B,3]
    #     #     # ここでは向きは「上向き」固定でOKなら単位クォータニオン
    #     #     gq = torch.zeros(B, 4, device=self.device)
    #     #     gq[:, 3] = 1.0

    #     #     self._debug_goal_w_cache[leg] = gp
    #     #     self._debug_goal_quat_cache[leg] = gq

        
    #     # [B,4,3] -> [B*4,3]
    #     vec_b = foot_targets_b.reshape(B * len(LEG_ORDER), 3)
    #     targets_w = foot_targets_b.reshape(B , len(LEG_ORDER), 3)


    #     # [B,4] -> [B,1,4] -> [B,4,4] -> [B*4,4]
    #     # quat_b = base_quat_w.unsqueeze(1).expand(-1, len(LEG_ORDER), -1)
    #     # quat_b = quat_b.reshape(B * len(LEG_ORDER), 4)

    #     # base -> world
    #     # [B*4,3]
    #     # vec_w = quat_apply(quat_b, vec_b)

    #     # ベース位置を足して [B*4,3] -> [B,4,3]
    #     # base_pos_rep = base_pos_w.unsqueeze(1).expand(-1, len(LEG_ORDER), -1)  # [B,4,3]
    #     # targets_w = (vec_w.reshape(B, len(LEG_ORDER), 3) + base_pos_rep)       # [B,4,3]

    #     # 各脚ごとにキャッシュ
    #     for leg_i, leg in enumerate(LEG_ORDER):
    #         gp = targets_w[:, leg_i, :]        # [B,3]
    #         gq = torch.zeros(B, 4, device=self.device)
    #         gq[:, 3] = 1.0                     # 向きはとりあえず単位 quat

    #         self._debug_goal_w_cache[leg] = gp
    #         self._debug_goal_quat_cache[leg] = gq




    def _set_debug_vis_impl(self, debug_vis: bool):
        self._debug = debug_vis
        if not hasattr(self, "_goal_markers"): self._goal_markers = {}
        if not hasattr(self, "_foot_markers"): self._foot_markers = {}

        if not debug_vis:
            for d in (self._goal_markers, self._foot_markers):
                for m in d.values(): m.set_visibility(False)
            return

        for leg in LEG_ORDER:
            if leg not in self._goal_markers:
                self._goal_markers[leg] = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            if leg not in self._foot_markers:
                self._foot_markers[leg] = VisualizationMarkers(self.cfg.feet_pose_visualizer_cfg)

        for d in (self._goal_markers, self._foot_markers):
            for m in d.values(): m.set_visibility(True)




    def _debug_vis_callback(self, event):
        if not hasattr(self, "_goal_markers") or not self._goal_markers:
            self._set_debug_vis_impl(True)
        
        robot = self.env.scene.articulations["robot"]
        if not robot.is_initialized: return

        for leg in LEG_ORDER:
            # update_command で計算して保存した値をここで描画
            gp = self._debug_goal_w_cache.get(leg, None)
            gq = self._debug_goal_quat_cache.get(leg, None)
            
            if gp is not None and gq is not None:
                self._goal_markers[leg].visualize(gp, gq)

            # 足の現在位置
            idx = robot.body_names.index(leg)
            if hasattr(robot.data, "body_link_pose_w"):
                pose = robot.data.body_link_pose_w[:, idx]
                fp, fq = pose[:, :3], pose[:, 3:7]
            else:
                fp = robot.data.body_pos_w[:, idx, :3]
                fq = getattr(robot.data, "body_quat_w", getattr(robot.data, "body_orient_w"))[:, idx, :4]
            
            self._foot_markers[leg].visualize(fp, fq)






class MultiLegBaseCommand(CommandTerm):
    """
    出力: command[num_envs, 12] = [FL(xb,yb,zb), FR(...), RL(...), RR(...)] （すべてベース座標）

    - 4脚すべて、指定された「ワールド矩形」から [xw,yw] をランダムサンプル
      （矩形は ped_areas[leg] で脚ごとに指定）
    - zw は固定値 top_z（または height_fn で決定）
    - サンプルは _resample_command でのみ行い、_update_command では保持した値を使う
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        B, dev = self.num_envs, self.device

        # 出力コマンド [B, 12] = [FL(3), FR(3), RL(3), RR(3)]
        self._command = torch.zeros(B, 3 * len(LEG_ORDER), device=dev)

        # 各脚ごとのローカルオフセット（矩形内をランダムサンプルして保持）
        #   _ped_local[leg]: [B,2]  (中心 center_xy からのオフセット)
        self._ped_local: Dict[str, torch.Tensor] = {
            leg: torch.zeros(B, 2, device=dev) for leg in LEG_ORDER
        }

        # ---- 設定 ----
        # 例:
        # ped_areas = {
        #   "FL_foot": {"center_xy": (0.25,  0.15), "half": (0.06,0.06), "top_z": 0.01},
        #   "FR_foot": {"center_xy": (0.25, -0.15), "half": (0.06,0.06), "top_z": 0.01},
        #   "RL_foot": {"center_xy": (-0.15,  0.15), "half": (0.06,0.06), "top_z": 0.01},
        #   "RR_foot": {"center_xy": (-0.15, -0.15), "half": (0.06,0.06), "top_z": 0.01},
        # }
        default_ped_areas = {
            "FL_foot": {"center_xy": (0.25,  0.15), "half": (0.1, 0.1), "top_z": 0.01},
            "FR_foot": {"center_xy": (0.25, -0.15), "half": (0.1, 0.1), "top_z": 0.01},
            "RL_foot": {"center_xy": (-0.15,  0.15), "half": (0.1, 0.1), "top_z": 0.01},
            "RR_foot": {"center_xy": (-0.15, -0.15), "half": (0.1, 0.1), "top_z": 0.01},
        }
        self.ped_areas: Dict[str, Dict] = getattr(cfg, "ped_areas", default_ped_areas)

        # 地形高さ関数を使いたい場合は (B,2)->(B,) を与える
        # 例: cfg.height_fn を別でセットしておき、ここで getattr で取る
        self.height_fn = getattr(cfg, "height_fn", None)

        # デバッグ可視化用
        self._goal_markers: Dict[str, VisualizationMarkers] = {}
        self._foot_markers: Dict[str, VisualizationMarkers] = {}

        # 初回サンプル
        self._resample_command(range(B))


    # --- 必須: 実装フック ---
    def _update_metrics(self):
        # 必要ならここでログを更新
        pass

    @property
    def command(self) -> torch.Tensor:
        return self._command

    # --------- サンプリング関数 ---------
    def _sample_rect(self, n: int, half_xy: Tuple[float, float]):
        """
        中心 (0,0)、半径 half_xy = (half_x, half_y) の矩形
        [-half_x, half_x] x [-half_y, half_y] 内を一様サンプル
        """
        hx, hy = half_xy
        x = torch.empty(n, device=self.device).uniform_(-hx, hx)
        y = torch.empty(n, device=self.device).uniform_(-hy, hy)
        return torch.stack([x, y], -1)  # [n,2]

    # --------- コマンド再サンプル ---------
    def _resample_command(self, env_ids=None):
        """
        env_ids:
        - None -> 全env
        - list / range / torch.Tensor → long tensor に変換
        """
        dev = self.device

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=dev, dtype=torch.long)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)

        # 4脚すべてについて、指定矩形の中でランダムサンプル
        for leg in LEG_ORDER:
            spec = self.ped_areas[leg]
            half = spec["half"]  # (half_x, half_y)
            self._ped_local[leg][env_ids] = self._sample_rect(env_ids.numel(), half)

        # サンプルした値を使ってコマンド更新
        self._update_command()

        # メトリクス（任意）
        self.metrics.setdefault("resample_count", torch.zeros(self.num_envs, device=dev))
        self.metrics["resample_count"][env_ids] += 1.0

    # --------- world→base 変換を含むコマンド更新 ---------
    def _update_command(self):
        B, dev = self.num_envs, self.device
        scene = self.env.scene

        # ベース姿勢 (world)
        robot = scene.articulations["robot"]
        base_p = robot.data.root_pos_w      # [B,3]
        base_q = robot.data.root_quat_w     # [B,4] (wxyz)
        R_wb3  = _rot3_from_quat_wxyz(base_q).transpose(-1, -2)  # world->base [B,3,3]

        # env 原点 (ワールド座標; IsaacLab で env ごとにずらす時用)
        origins = getattr(scene, "env_origins", torch.zeros(B,3, device=dev))
        o_xy = origins[...,:2]  # [B,2]

        leg_targets_b: Dict[str, torch.Tensor] = {}

        # 4脚すべて同じルールでターゲット計算
        for leg in LEG_ORDER:
            spec = self.ped_areas[leg]
            center_xy = torch.as_tensor(
                spec["center_xy"], device=dev, dtype=base_p.dtype
            )  # [2]

            # env原点 + 指定中心 + ランダムオフセット
            ctr = o_xy + center_xy                 # [B,2]
            off = self._ped_local[leg]            # [B,2] (矩形内ランダム)
            t_xy_w = ctr + off                    # [B,2]

            # 高さ
            if self.height_fn is not None:
                t_z_w = self.height_fn(t_xy_w)                     # [B,]
            else:
                t_z_w = torch.full(
                    (B,), float(spec.get("top_z", 0.0)),
                    device=dev, dtype=base_p.dtype
                )

            # world 座標 [B,3]
            t_w = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)

            # base 座標へ変換: (R_wb3 @ (t_w - base_p))
            leg_targets_b[leg] = (R_wb3 @ (t_w - base_p).unsqueeze(-1)).squeeze(-1)

        # 出力（FL,FR,RL,RR の順番）
        out = torch.zeros(B, 3 * len(LEG_ORDER), device=dev, dtype=base_p.dtype)
        for i, leg in enumerate(LEG_ORDER):
            out[:, 3*i:3*(i+1)] = leg_targets_b[leg]
        self._command = out   # [B,12]　ベース座標系でのコマンド

    # --------- デバッグ可視化設定 ---------
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

    # --------- デバッグ可視化コールバック ---------
    def _debug_vis_callback(self, event):
        # markers がまだなければここで必ず作る
        if not hasattr(self, "_goal_markers") or not self._goal_markers:
            self._set_debug_vis_impl(True)

        robot = self.env.scene.articulations["robot"]
        if not robot.is_initialized:
            return

        B, dev = self.num_envs, self.device

        # env 原点
        origins = getattr(self.env.scene, "env_origins", torch.zeros(B,3, device=dev))
        o_xy = origins[...,:2]  # [B,2]

        # 4脚の目標姿勢 (world)
        goal_w: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for leg in LEG_ORDER:
            spec = self.ped_areas[leg]
            center_xy = torch.as_tensor(
                spec["center_xy"], device=dev, dtype=origins.dtype
            )      # [2]
            off = self._ped_local[leg]            # [B,2]
            t_xy_w = o_xy + center_xy + off       # [B,2]

            if self.height_fn is not None:
                t_z_w = self.height_fn(t_xy_w)
            else:
                t_z_w = torch.full(
                    (B,), float(spec.get("top_z", 0.0)),
                    device=dev, dtype=origins.dtype
                )

            pos = torch.cat([t_xy_w, t_z_w.unsqueeze(-1)], -1)   # [B,3]
            # 向きはとりあえず無回転
            quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=dev, dtype=origins.dtype
            ).expand(B, 4)
            goal_w[leg] = (pos, quat)

        # 足先 現在値 (world)
        def _foot_pose_w(leg_name):
            idx = robot.body_names.index(leg_name)
            if hasattr(robot.data, "body_link_pose_w"):
                pose = robot.data.body_link_pose_w[:, idx]  # [B,7]
                return pose[:, :3], pose[:, 3:7]
            else:
                pos = robot.data.body_pos_w[:, idx, :3]
                quat = getattr(
                    robot.data, "body_quat_w",
                    getattr(robot.data, "body_orient_w")
                )[:, idx, :4]
                return pos, quat

        # 描画
        for leg in LEG_ORDER:
            gp, gq = goal_w[leg]        # [B,3], [B,4]
            fp, fq = _foot_pose_w(leg)  # [B,3], [B,4]
            self._goal_markers[leg].visualize(gp, gq)
            self._foot_markers[leg].visualize(fp, fq)
