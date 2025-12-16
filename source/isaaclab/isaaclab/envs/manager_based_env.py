# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import torch
from collections.abc import Sequence
from typing import Any

import isaacsim.core.utils.torch as torch_utils
import omni.log
import omni.physx
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version

from isaaclab.managers import ActionManager, EventManager, ObservationManager, RecorderManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import attach_stage_to_usd_context, use_stage
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.utils.timer import Timer

from .common import VecEnvObs
from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ViewportCameraController
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import os
import omni.usd
import numpy as np
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg


from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema
from isaaclab.sim import utils as sim_utils



# 中央の台
PLATFORM_SIZE = 1.0          # 1.0 x 1.0
PLATFORM_HALF = PLATFORM_SIZE * 0.5

# Go2 のサイズ感（大雑把）
GO2_LENGTH = 0.70            # だいたい 70 cm
GO2_WIDTH  = 0.31            # だいたい 31 cm

# 石幅のレンジ [m]（例：簡単な時は 0.35、大変になると 0.22 まで細く）
STONE_WIDTH_RANGE = (0.22, 0.35)     # (min, max)

# 石同士のギャップのレンジ [m]（例：最初ほぼ 0、難しくなると ~0.3）
STONE_GAP_RANGE   = (0.0, 0.30)      # (min, max)

# リングの内側半径（台の縁+数 mm）
INNER_MARGIN = 0.01                  # 台とのギャップほぼ 0
INNER_HALF   = PLATFORM_HALF + INNER_MARGIN   # ≒ 0.51

# リングの厚さ（どこまで外側に石を置くか）
RING_WIDTH   = GO2_LENGTH *  2# 体長の8割くらい外に広げる
OUTER_HALF   = INNER_HALF + RING_WIDTH
OUTER_HALF   = 5


Y_LIMIT      = 1.2                   # 左右にどこまで石を出すか
STONE_H      = 0.30                  # 高さは一定



def generate_stepping_stone_ring_xy(
    stone_w: float,
    inner_half: float,
    outer_half: float,
    gap: float = 1e-4,
    margin: float = 1e-3,
    y_limit: float = 1.5,
    front_only: bool = True,
):
    """
    - stone_w   : ブロック一辺
    - inner_half: 中央台の半サイズ＋α (この内側には石を置かない)
    - outer_half: リングの外側半径
    - gap       : ブロック同士の隙間
    - margin    : inner_half から石までの隙間（ほぼ0でOK）
    - y_limit   : |y| <= y_limit の範囲だけ使う
    - front_only: True なら +x 側だけ（ロボット前方だけ）石を置く
    戻り値: [(x, y), ...]  （env原点基準のローカル座標）
    """
    pitch = stone_w + gap
    half_w = stone_w * 0.5

    # 台の縁 inner_half から margin だけ外側に、
    # ブロックの内側がほぼ接するように中心半径 r0 を決める
    #   inner_edge ≒ inner_half + margin
    #   center    = inner_edge + half_w
    r0 = inner_half + margin + half_w

    max_center = outer_half - half_w
    if r0 > max_center:
        return []

    # 正側の中心位置を r0 から pitch ごとに並べる
    coords_pos = np.arange(r0, max_center + 1e-6, pitch)
    if coords_pos.size == 0:
        return []

    coords_neg = -coords_pos[::-1]

    xs = np.concatenate((coords_neg, coords_pos))
    ys = np.concatenate((coords_neg, coords_pos))

    xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

    # L∞ノルムでリング判定
    max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

    m_inner = (max_dist_center >= r0 - half_w)   # 中央台＋margin から外側
    m_outer = (max_dist_center <= outer_half - half_w)

    # 前方だけにするかどうか
    if front_only:
        m_x = (xs_grid - half_w > 0.0)
    else:
        m_x = np.ones_like(xs_grid, dtype=bool)

    # y 範囲制限
    m_y_upper = (ys_grid + half_w < y_limit)
    m_y_lower = (ys_grid - half_w > -y_limit)

    m = m_inner & m_outer & m_x & m_y_upper & m_y_lower

    xs_flat, ys_flat = xs_grid[m], ys_grid[m]
    return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]



def resolve_stone_params(difficulty: float):
    """
    IsaacLab公式 stepping_stones_terrain と同じ補間で
    石幅とギャップを決める。
    difficulty: 0.0(易) ～ 1.0(難)
    """
    d = float(np.clip(difficulty, 0.0, 1.0))

    w_min, w_max = STONE_WIDTH_RANGE
    g_min, g_max = STONE_GAP_RANGE

    # stone_width = max - d*(max-min)
    stone_w = w_max - d * (w_max - w_min)
    # stone_distance = min + d*(max-min)
    stone_gap = g_min + d * (g_max - g_min)

    return stone_w, stone_gap




def generate_ring_xy_isaac(
    difficulty: float,
    inner_half: float = INNER_HALF,
    outer_half: float = OUTER_HALF,
    y_limit: float = Y_LIMIT,
    front_only: bool = True,
):
    """
    IsaacLab公式と同じ difficulty 補間で
    stone_w / gap を決めてリング座標を返す。
    戻り値: [(x, y), ...] （env原点基準）
    """
    stone_w, stone_gap = resolve_stone_params(difficulty)
    return generate_stepping_stone_ring_xy(
        stone_w=stone_w,
        inner_half=inner_half,
        outer_half=outer_half,
        gap=stone_gap,
        margin=1e-4,      # 台との隙間ほぼ0
        y_limit=y_limit,
        front_only=front_only,
    )




def reset_stones_ring(
    env,
    env_ids: torch.Tensor,
    stone_xy_local: torch.Tensor,
    stone_z_local: float,
    collection_name: str = "stones",
):
    """
    env.scene.env_origins を足して、ローカル座標のリングを
    各 env の world 座標に配置し直す。
    """
    device = env.device
    stones = env.scene.rigid_object_collections[collection_name]

    env_origins = env.scene.env_origins[env_ids]  # [N_env, 3]

    n_env = env_ids.shape[0]
    n_block = stone_xy_local.shape[0]

    state = stones.data.object_state_w[env_ids].clone()
    pos = state[..., 0:3]
    quat = state[..., 3:7]
    linv = state[..., 7:10]
    angv = state[..., 10:13]

    xy_local = stone_xy_local.to(device)   # [N_block, 2]
    x_local = xy_local[:, 0]               # [N_block]
    y_local = xy_local[:, 1]

    x_origin = env_origins[:, 0:1]         # [N_env, 1]
    y_origin = env_origins[:, 1:2]
    z_origin = env_origins[:, 2:3]

    pos[:, :, 0] = x_origin + x_local[None, :]
    pos[:, :, 1] = y_origin + y_local[None, :]
    pos[:, :, 2] = z_origin + stone_z_local

    quat[:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    linv.zero_()
    angv.zero_()

    stones.write_object_state_to_sim(object_state=state, env_ids=env_ids)


# def add_breakable_spherical_joint4(stage, stone_path,
#                                    break_force=200.0, break_torque=30.0, cone_limit_deg=8.0):
#     if not stone_path or not stone_path.startswith("/"):
#         raise ValueError(f"stone_path must be absolute: {stone_path!r}")
#     stone = stage.GetPrimAtPath(stone_path)
#     if not stone.IsValid():
#         raise RuntimeError(f"stone prim not found: {stone_path}")

#     joint_path = f"{stone_path}/Joint"
#     joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
#     joint.CreateBody0Rel().SetTargets([stone_path])   # World 拘束（片側のみ）
#     joint.CreateBody1Rel().SetTargets(["/World"])

#     # 石の半厚を見積もる（extentが無い時はフォールバック値）
#     half_h = 0.15
#     try:
#         b = UsdGeom.Boundable(stone)
#         ext = b.GetExtentAttr().Get()
#         if ext:
#             half_h = 0.5 * (float(ext[1][2]) - float(ext[0][2]))
#     except Exception:
#         pass

#     # 石ローカルの底面中心をアンカーに
#     local_anchor = Gf.Vec3f(0.0, 0.0, -0.01)

#     # ワールド座標へ変換（★ここを修正）
#     time = Usd.TimeCode.Default()  # もしくは Usd.TimeCode(stage.GetStartTimeCode())
#     xf = UsdGeom.Xformable(stone).ComputeLocalToWorldTransform(time)
#     world_anchor = Gf.Matrix4d(xf).Transform(Gf.Vec3d(0.0, 0.0, 0))

#     # アンカー整合
#     joint.CreateLocalPos0Attr().Set(local_anchor)            # 石ローカル
#     joint.CreateLocalRot0Attr().Set(Gf.Quatf(1,0,0,0))
#     joint.CreateLocalPos1Attr().Set(Gf.Vec3f(world_anchor))  # ワールド側
#     joint.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))

#     # 破断＆コーン角
#     joint.CreateBreakForceAttr().Set(float(break_force))
#     joint.CreateBreakTorqueAttr().Set(float(break_torque))

#     lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "angular")
#     lim.CreateLowAttr().Set(0.0)
#     lim.CreateHighAttr().Set(float(cone_limit_deg))
#     if hasattr(lim, "CreateEnabledAttr"):
#         lim.CreateEnabledAttr().Set(True)






# def add_breakable_spherical_joint7(stage, stone_path, env_origin_vec: Gf.Vec3f,
#                                    break_force=5000.0, break_torque=500.0,
#                                    cone_limit_deg=180.0):
    
#     stone = stage.GetPrimAtPath(stone_path)
#     if not stone.IsValid():
#         print(f"Warning: stone prim not found: {stone_path}")
#         return 

#     joint_path = f"{stone_path}_Joint"
#     if stage.GetPrimAtPath(joint_path).IsValid():
#         stage.RemovePrim(joint_path)

#     joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)

#     # World拘束: Body0=石、Body1=/World (★これは必須)
#     joint.CreateBody0Rel().SetTargets([stone_path])
#     joint.CreateBody1Rel().SetTargets(["/World"])

#     local_bottom_z =0

#     # === アンカー計算 (★ここが重要) ===
    
#     # 1. 石の「ローカル」変換を取得 (親である env_X からの相対)
#     stone_xform = UsdGeom.Xformable(stone)
#     # GetLocalTransformation() は、親を含まない、純粋な石自身のxformOpを読む
#     local_transform_matrix = stone_xform.GetLocalTransformation(Usd.TimeCode.Default())
#     # 石のローカル原点(0,0,0)が、env_X の中でどの座標にあるか
#     local_p_vec3d = local_transform_matrix.Transform(Gf.Vec3d(0, 0, local_bottom_z))
#     local_p = Gf.Vec3f(local_p_vec3d) # (例: env_X の (1, 2, 0.15) など)
    
#     # 2. 正しいワールド座標を「手動で計算」する
#     #    p_world = (envのワールド座標) + (石のenv内ローカル座標)
#     p_world = env_origin_vec + local_p

#     # 3. アンカーを設定 (吹っ飛ばない「0基準」アンカー)
#     # ローカル側 (Body0 = 石)
#     joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0)) # 石の原点
#     joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    
#     # ワールド側 (Body1 = /World)
#     joint.CreateLocalPos1Attr().Set(p_world) # ★計算した正しいワールド座標
#     joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

#     # 破断とリミット
#     joint.CreateBreakForceAttr().Set(float(break_force))
#     joint.CreateBreakTorqueAttr().Set(float(break_torque))

#     lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "angular")
#     lim.CreateLowAttr().Set(0.0)
#     lim.CreateHighAttr().Set(float(cone_limit_deg))
#     if hasattr(lim, "CreateEnabledAttr"):
#         lim.CreateEnabledAttr().Set(True)




def post_spawn_hook2(scene: InteractiveScene):
    """
    シーン生成直後に呼び、各ENVの石に破断ジョイントを付与。
    """
    stage = omni.usd.get_context().get_stage()
    
    # シーンからenvごとの原点（オフセット）を取得 (CPUのnumpy配列に変換)
    env_origins_np = scene.env_origins.cpu().numpy() # (num_envs, 3)

    for env_id in range(scene.num_envs):
        # このenvのワールド座標系でのオフセット (Gf.Vec3f型)
        env_origin = Gf.Vec3f(float(env_origins_np[env_id][0]), 
                              float(env_origins_np[env_id][1]), 
                              float(env_origins_np[env_id][2]))

        for i in range(3): # Stone_0, Stone_1, ...
            stone_path = f"/World/envs/env_{env_id}/Stone_{i}"
            
            # ★ env_origin を渡す新しい関数 (v7) を呼ぶ
            add_breakable_spherical_joint7(
                stage, stone_path, env_origin,
                break_force=5000.0, break_torque=500.0, cone_limit_deg=180.0
            )


def debug_paths(scene):
    stage = omni.usd.get_context().get_stage()
    for e in range(scene.num_envs):
        env_ns = f"/World/envs/env_{e}"
        print("ENV exists:", env_ns, stage.GetPrimAtPath(env_ns).IsValid())
        for i in range(3):
            stone = f"{env_ns}/Stone_{i}"   # 親なしで出しているなら f"{env_ns}/Stone_{i}"
            joint = f"{env_ns}/Stone_{i}/Joint"
            print(" stone:", stone, "->", stage.GetPrimAtPath(stone).IsValid())
            print(" joint-parent:", os.path.dirname(joint), "->",
                  stage.GetPrimAtPath(os.path.dirname(joint)).IsValid())



def ensure_parent_xform(stage, prim_path: str):
    parent = os.path.dirname(prim_path.rstrip("/"))
    if parent and not stage.GetPrimAtPath(parent).IsValid():
        UsdGeom.Xform.Define(stage, parent)


def apply_physics_material_compat(prim, static_fric: float, dynamic_fric: float, restitution: float):
    """
    prim（Geom/Collision を持つ剛体）に対して、環境差を吸収して摩擦・反発を設定する。
    優先順：
      1) PhysxSchema.PhysxMaterialAPI を prim に直付け（メソッドあれば）
      2) 直付けの raw 属性（'physxMaterial:staticFriction' など）を作成して Set
      3) UsdPhysics.MaterialAPI の“別Prim”を作成し、binding リレーションを手書きで貼る
    """
    stage = prim.GetStage()

    # --- (1) PhysxMaterialAPI で直付けできるか試す ---
    try:
        pmat = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        # メソッド名の有無で分岐（ビルド差吸収）
        if hasattr(pmat, "CreateStaticFrictionAttr") and hasattr(pmat, "CreateDynamicFrictionAttr"):
            pmat.CreateStaticFrictionAttr(float(static_fric))
            pmat.CreateDynamicFrictionAttr(float(dynamic_fric))
            if hasattr(pmat, "CreateRestitutionAttr"):
                pmat.CreateRestitutionAttr(float(restitution))
            else:
                # 古い版では 'physxMaterial:restitution' を raw で作る
                prim.CreateAttribute("physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(float(restitution))
            return
        elif hasattr(pmat, "CreateFrictionAttr"):
            # 単一 friction の版（必要なら両方に同値を入れる）
            pmat.CreateFrictionAttr(float(max(static_fric, dynamic_fric)))
            if hasattr(pmat, "CreateRestitutionAttr"):
                pmat.CreateRestitutionAttr(float(restitution))
            else:
                prim.CreateAttribute("physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(float(restitution))
            return
    except Exception:
        pass

    # --- (2) PhysX raw 属性を prim に直接作る ---
    try:
        prim.CreateAttribute("physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(float(static_fric))
        prim.CreateAttribute("physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(float(dynamic_fric))
        prim.CreateAttribute("physxMaterial:restitution",   Sdf.ValueTypeNames.Float).Set(float(restitution))
        return
    except Exception:
        pass

    # --- (3) UsdPhysics.MaterialAPI で別Primを作り、binding を手書き ---
    try:
        mat_path = prim.GetPath().pathString + "/PhysMaterial"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            UsdGeom.Xform.Define(stage, mat_path)  # ダミー Prim でOK
        mat = stage.GetPrimAtPath(mat_path)
        mapi = UsdPhysics.MaterialAPI.Apply(mat)

        if hasattr(mapi, "CreateStaticFrictionAttr"):
            mapi.CreateStaticFrictionAttr(float(static_fric))
            mapi.CreateDynamicFrictionAttr(float(dynamic_fric))
            mapi.CreateRestitutionAttr(float(restitution))
        else:
            # 最後の手段：physics:material 名前空間で raw 属性
            mat.CreateAttribute("physics:material:staticFriction", Sdf.ValueTypeNames.Float).Set(float(static_fric))
            mat.CreateAttribute("physics:material:dynamicFriction", Sdf.ValueTypeNames.Float).Set(float(dynamic_fric))
            mat.CreateAttribute("physics:material:restitution",    Sdf.ValueTypeNames.Float).Set(float(restitution))

        # MaterialBindingAPI が無い環境用：リレーションを直接作る
        rel = prim.CreateRelationship("physics:material:binding", False)
        rel.SetTargets([Sdf.Path(mat_path)])
        return
    except Exception:
        pass

    print("[material] WARNING: failed to set physics material on", prim.GetPath())






from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd, math, random

import re

def _define_xform_safely(stage, path: str):
    """
    - 既に存在すれば何もしない
    - 親階層にインスタンスが居ないかチェック
    - 書き込み可能なレイヤ（RootLayer or SessionLayer）を EditTarget にして Define
    """
    if stage.GetPrimAtPath(path).IsValid():
        return

    # 親にインスタンスがいると子は作れない
    p = Sdf.Path(path)
    a = p.GetParentPath()
    while a != Sdf.Path.absoluteRootPath:
        prim = stage.GetPrimAtPath(a.pathString)
        if prim and prim.IsInstance():
            raise RuntimeError(f"cannot define under instance: {a}")
        a = a.GetParentPath()

    # 書けるレイヤを選ぶ
    root = stage.GetRootLayer()
    if root.permissionToEdit:
        target_layer = root
    else:
        # ルートが書けない場合はセッションレイヤを書く（Omniverse ではここは書ける）
        target_layer = stage.GetSessionLayer()

    # 親から順に作る（中間パスが無いと失敗する版がある）
    to_make = []
    cur = Sdf.Path(path)
    while cur != Sdf.Path.absoluteRootPath:
        if not stage.GetPrimAtPath(cur.pathString).IsValid():
            to_make.append(cur.pathString)
        cur = cur.GetParentPath()
    to_make.reverse()

    with Usd.EditContext(stage, target_layer):
        for pth in to_make:
            UsdGeom.Xform.Define(stage, pth)


def concretize_env_path(token_or_regex_path: str, env_id: int) -> str:
    """
    受け取ったパスが:
      - "{ENV_REGEX_NS}/..."  あるいは "{{ENV_REGEX_NS}}/..."
      - "/World/envs/env_.*/..."
    のどれでも、"/World/envs/env_{env_id}/..." に具体化して返す。
    それ以外はそのまま返す。
    """
    s = token_or_regex_path

    # {ENV_REGEX_NS} / {{ENV_REGEX_NS}} → /World/envs/env_{id}
    s = re.sub(r"^\{\{?ENV_REGEX_NS\}?\}", f"/World/envs/env_{env_id}", s)

    # /World/envs/env_.*/... → /World/envs/env_{id}/...
    s = re.sub(r"^/World/envs/env_\.\*", f"/World/envs/env_{env_id}", s)

    return s






# def attach_breakable_spherical_joints(scene, stones,
#                                       break_force=(200.0, 400.0),     # [N]
#                                       break_torque=(40.0, 80.0),      # [N·m]
#                                       cone_deg=25.0,                   # 円錐角(±deg)
#                                       anchor_local_pos0=(0.0, 0.0, 0.0),  # 石側のローカル基点
#                                       seed=0):
#     """
#     - scene: InteractiveScene（envの数やenv名を持つ）
#     - stones: RigidObjectCollection インスタンス
#     - RigidObjectCollectionCfg の prim_path に {ENV_REGEX_NS} が入っている前提
#     """
#     rng = random.Random(seed)
#     stage = omni.usd.get_context().get_stage()

#     # ジョイント用のルート（インスタンス外）を用意
#     joints_root = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Joints")).GetPrim()

#     # env_i の論理名（/World/envs/env_i）を作る小ユーティリティ
#     def env_ns(i): 
#         return f"/World/envs/env_{i}"

#     # 量が多いので高速化
#     # with Sdf.ChangeBlock():
#     for ei in range(scene.num_envs):
#         # 環境ごとにサブフォルダ
#         env_joint_root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/World/Joints/env_{ei}"))

#         for idx, name in enumerate(stones.object_names):
#             # RigidObjectCollectionCfg 側のパス式を実パスに解決
#             expr = stones.cfg.rigid_objects[name].prim_path  # 例: "{ENV_REGEX_NS}/Stones/Stone_.*/Base"
#             stone_path = expr.replace("{ENV_REGEX_NS}", env_ns(ei))

#             # ジョイントprimはインスタンス外に作る
#             jpath = f"/World/Joints/env_{ei}/{name}_joint_{idx:04d}"
#             joint = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(jpath))

#             # Body0=石、Body1(未設定)=World 拘束
#             joint.CreateBody0Rel().SetTargets([Sdf.Path(stone_path)])
#             # Body1 は意図的に設定しない

#             # 破断しきい値（USD属性）
#             jf = rng.uniform(*break_force)
#             jt = rng.uniform(*break_torque)
#             joint.CreateBreakForceAttr().Set(jf)
#             joint.CreateBreakTorqueAttr().Set(jt)

#             # アンカー（ローカルフレーム）
#             joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor_local_pos0))
#             joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # 単位クォータニオン
#             # Body1はWorldなので LocalPos1/Rot1 は未設定のままでOK

#             # 円錐角制限（rotY, rotZ を ±cone_deg に）
#             for dof in ("rotY", "rotZ"):
#                 lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), dof)
#                 lim.CreateLowAttr(-float(cone_deg))
#                 lim.CreateHighAttr(+float(cone_deg))



# def attach_joints_for_all_stones(scene, breakF_range=(150.0, 300.0), breakT_range=(20.0, 40.0)):
#     import random
#     stage = omni.usd.get_context().get_stage()
#     stones = scene["stones"]

#     with Sdf.ChangeBlock():  # 大量生成をバッチ化
#         for env_id in range(scene.num_envs):
#             env_ns = f"/World/envs/env_{env_id}"
#             for name in stones.object_names:
#                 stone_expr = stones.cfg.rigid_objects[name].prim_path  # "{ENV_REGEX_NS}/Stones/Stone_XXXX"
#                 stone_path = concretize_env_path(stone_expr, env_id)

#                 print(stone_expr, stone_path)

#                 # stone_path = stone_expr.replace("{ENV_REGEX_NS}", env_ns)
#                 add_breakable_spherical_joints(
#                     stage, stone_path,
#                     break_force=random.uniform(*breakF_range),
#                     break_torque=random.uniform(*breakT_range),
#                     cone_deg=8.0,
#                 )




def concretize_env_path(path_expr: str, env_id: int) -> str:
    """{ENV_REGEX_NS} / {{ENV_REGEX_NS}} / /World/envs/env_.*/... を /World/envs/env_{env_id}/... に具体化"""
    s = re.sub(r"^\{\{?ENV_REGEX_NS\}?\}", f"/World/envs/env_{env_id}", path_expr)
    s = re.sub(r"^/World/envs/env_\.\*", f"/World/envs/env_{env_id}", s)
    return s

def ensure_prim_on_session(stage: Usd.Stage, path: str, type_name: str = "Xform"):
    """RootLayerが書けなくても SessionLayer に直に Prim を定義。親チェーンも作る。"""
    if stage.GetPrimAtPath(path).IsValid():
        return
    layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, layer):
        # 親から順に
        cur = Sdf.Path(path)
        stack = []
        while cur != Sdf.Path.absoluteRootPath:
            if not stage.GetPrimAtPath(cur.pathString).IsValid():
                stack.append(cur)
            cur = cur.GetParentPath()
        for p in reversed(stack):
            UsdGeom.Xform.Define(stage, p)
        # 目的の型に差し替え（Xformで十分なら不要）
        if type_name != "Xform":
            ps = layer.GetPrimAtPath(path)
            if ps: ps.typeName = type_name

# def add_breakable_joint_world(
#     stage, stone_path: str,
#     break_force: float, break_torque: float, cone_deg: float,
#     local_anchor: Gf.Vec3f,
#     joint_root="/Joints"
# ):
#     """Body0=石, Body1未設定(=World)。アンカーは「石ローカル(local_anchor)の世界座標」に一致させる。"""
#     stone = stage.GetPrimAtPath(stone_path)
#     if not stone or not stone.IsValid():
#         print("[joint] missing:", stone_path); return False
#     if stone.IsInstance():
#         print("[joint] instance prim—cannot joint:", stone_path); return False

#     # /Joints と /Joints/env_k を SessionLayerに作成
#     parts = stone_path.split("/")
#     # /World/envs/env_k/... を想定 → env_k を抽出
#     env_name = parts[3] if len(parts) > 3 else "env_0"
#     env_joint_dir = f"{joint_root}/{env_name}"
#     ensure_prim_on_session(stage, joint_root, "Xform")
#     ensure_prim_on_session(stage, env_joint_dir, "Xform")

#     joint_path = f"{env_joint_dir}/{stone.GetName()}_Joint"
#     if stage.GetPrimAtPath(joint_path).IsValid():
#         stage.RemovePrim(joint_path)

#     # ローカル→ワールド
#     T_w = UsdGeom.Xformable(stone).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
#     p_world = Gf.Vec3f(Gf.Matrix4d(T_w).Transform(Gf.Vec3d(local_anchor)))

#     # Joint を SessionLayer で定義
#     with Usd.EditContext(stage, stage.GetSessionLayer()):
#         j = UsdPhysics.SphericalJoint.Define(stage, joint_path)
#         j.CreateBody0Rel().SetTargets([stone_path])           # Body1 未設定 = World 拘束
#         j.CreateLocalPos0Attr().Set(local_anchor)             # 石ローカル
#         j.CreateLocalRot0Attr().Set(Gf.Quatf(1,0,0,0))
#         j.CreateLocalPos1Attr().Set(p_world)                  # World 側は世界座標をそのまま
#         j.CreateLocalRot1Attr().Set(Gf.Quatf(1,0,0,0))
#         j.CreateBreakForceAttr().Set(float(break_force))
#         j.CreateBreakTorqueAttr().Set(float(break_torque))

#         # 角度制限（環境依存だが、まず "angular" に Low/High（スカラ）で安定）
#         lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "angular")
#         lim.CreateLowAttr().Set(0.0)
#         lim.CreateHighAttr().Set(float(cone_deg))
#         if hasattr(lim, "CreateEnabledAttr"): lim.CreateEnabledAttr().Set(True)
#     return True

# def attach_breakable_spherical_joints2(scene, stones,
#                                       break_force=(200.0, 400.0),
#                                       break_torque=(40.0, 80.0),
#                                       cone_deg=25.0,
#                                       anchor_local_pos0=(0.0, 0.0, 0.0),
#                                       seed=0):
#     rng = random.Random(seed)
#     stage = omni.usd.get_context().get_stage()

#     # Joint ルートはインスタンス外（SessionLayerに強制作成）
#     ensure_prim_on_session(stage, "/Joints", "Xform")

#     with Sdf.ChangeBlock():
#         for ei in range(scene.num_envs):
#             # RigidObjectCollection の各要素名を使って cfg から具体パスを得る
#             for idx, name in enumerate(stones.object_names):
#                 expr = stones.cfg.rigid_objects[name].prim_path
#                 stone_path = concretize_env_path(expr, ei)

#                 # ローカル支点（例：石底面の中心にしたいならここを (0,0,-h/2) に）
#                 local_anchor = Gf.Vec3f(*anchor_local_pos0)

#                 ok = add_breakable_joint_world(
#                     stage,
#                     stone_path=stone_path,
#                     break_force=rng.uniform(*break_force),
#                     break_torque=rng.uniform(*break_torque),
#                     cone_deg=cone_deg,
#                     local_anchor=local_anchor,
#                     joint_root="/Joints"
#                 )
#                 if not ok:
#                     # ここでログだけ出して継続（インスタンスだった等）
#                     pass



from pxr import Usd, Sdf
import omni.usd
import omni.kit.commands

# def _deinstance_under(root_path: str):
#     stage = omni.usd.get_context().get_stage()
#     to_uninstance = []
#     for prim in stage.Traverse():
#         p = prim.GetPath().pathString
#         if p.startswith(root_path) and prim.IsInstance():
#             to_uninstance.append(prim.GetPath())
#     if to_uninstance:
#         omni.kit.commands.execute("UninstancePrims", paths=to_uninstance)

# def _create_spherical_joint(stage, joint_path, body0_path, body1_path,
#                             local_pos0=Gf.Vec3f(0,0,-0.12),
#                             local_pos1=Gf.Vec3f(0,0,0),
#                             break_force=600.0, break_torque=100.0):
#     joint = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(joint_path))
#     joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
#     joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
#     joint.CreateLocalPos0Attr().Set(local_pos0)
#     joint.CreateLocalPos1Attr().Set(local_pos1)
#     pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
#     pj.CreateBreakForceAttr().Set(break_force)
#     pj.CreateBreakTorqueAttr().Set(break_torque)

# def wire_stones_after_spawn(env_index: int,
#                             env_ns_template="/World/envs/env_{env}",
#                             ground_rel="Terrain/ground",
#                             stone_root_rel="fragile/Stones",
#                             joints_root_rel="fragile/Joints"):
#     """Call this AFTER stones are spawned."""
#     stage = omni.usd.get_context().get_stage()
#     env_ns = env_ns_template.format(env=env_index)
#     stone_root = f"{env_ns}/{stone_root_rel}".rstrip("/")
#     ground = f"{env_ns}/{ground_rel}".rstrip("/")
#     joints_root = f"{env_ns}/{joints_root_rel}".rstrip("/")

#     # 1) デインスタンス（CollectionCfgで生成された石を個別化）
#     _deinstance_under(stone_root)

#     # 2) 石のルート候補を列挙（必要なら型フィルタを調整）
#     stones = []
#     for prim in stage.Traverse():
#         p = prim.GetPath().pathString
#         if p.startswith(stone_root + "/") and prim.GetTypeName() in ("Xform","Mesh","Cube","Sphere","Capsule"):
#             stones.append(p)

#     if not stage.GetPrimAtPath(joints_root):
#         stage.DefinePrim(Sdf.Path(joints_root), "Xform")

#     # 3) 各石に Joint を作成
#     for i, stone in enumerate(stones):
#         jp = f"{joints_root}/joint_{i:04d}"
#         _create_spherical_joint(stage, jp, stone, ground)



def _has_any_rigid(prim):
    try:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
    except Exception:
        return prim.HasAPI(UsdPhysics.RigidBodyAPI)

def _world_xform(stage, path):
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)


from pxr import Sdf

def _set_if_exists(prim, attr_name, value, sdf_type):
    """その属性が prim に存在する場合だけ Set する"""
    a = prim.GetAttribute(attr_name)
    if a:
        a.Set(value)
        return True
    return False

def enable_projection_if_supported(joint_prim):
    # よくある名前を順番に試す（どれかが True になればOK）
    ok = False
    ok |= _set_if_exists(joint_prim, "physxJoint:enableProjection", True,  Sdf.ValueTypeNames.Bool)
    ok |= _set_if_exists(joint_prim, "physx:enableProjection",      True,  Sdf.ValueTypeNames.Bool)
    # まれに使われる別名（存在すれば）
    ok |= _set_if_exists(joint_prim, "physxJoint:projectionEnabled", True, Sdf.ValueTypeNames.Bool)
    ok |= _set_if_exists(joint_prim, "physx:projectionEnabled",      True, Sdf.ValueTypeNames.Bool)

    # トレランスも同様に（存在する方だけ）
    _set_if_exists(joint_prim, "physxJoint:projectionLinearTolerance",  0.05, Sdf.ValueTypeNames.Float)
    _set_if_exists(joint_prim, "physx:projectionLinearTolerance",       0.05, Sdf.ValueTypeNames.Float)
    _set_if_exists(joint_prim, "physxJoint:projectionAngularTolerance", 45.0, Sdf.ValueTypeNames.Float)
    _set_if_exists(joint_prim, "physx:projectionAngularTolerance",      45.0, Sdf.ValueTypeNames.Float)
    return ok

# def _create_locklike_spherical_joint(stage, joint_path, body0_path, body1_path=None,
#                                      # ★ 引数を追加
#                                      local0_pos: Gf.Vec3f = Gf.Vec3f(0,0,0),
#                                      local1_pos: Gf.Vec3f = Gf.Vec3f(0,0,0)):
#     # j = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(joint_path)) #sphere
#     j = UsdPhysics.PrismaticJoint.Define(stage, Sdf.Path(joint_path)) #prismatic

#     j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    
#     # ★★★ 修正点 1: Body1 を明示的に /World に設定 ★★★
#     if body1_path:
#         j.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
#     # else:
#     #     # body1_path が None の場合は、/World をターゲットにする
#         j.CreateBody1Rel().SetTargets([Sdf.Path("/World")])

#     # ★★★ 修正点 2: 引数を使う ★★★
#     # アンカー：石ローカル側 (計算済みの値)
#     j.CreateLocalPos0Attr().Set(local0_pos)
#     # アンカー：ワールド側 (計算済みの値)
#     j.CreateLocalPos1Attr().Set(local1_pos)
    
#     # 回転は (1,0,0,0) で初期化
#     # j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
#     # j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

#     # 破断設定 (この部分は元のままでOK)
#     # if hasattr(j, "CreateConeAngle0Attr"): j.CreateConeAngle0Attr().Set(5.0)
#     # if hasattr(j, "CreateConeAngle1Attr"): j.CreateConeAngle1Attr().Set(5.0)

#     j.CreateLowerLimitAttr().Set(0)
#     j.CreateUpperLimitAttr().Set(0)

#     # 1. 破断設定 (Break) は UsdPhysics.Joint (j) に直接設定します
#     j.CreateBreakForceAttr().Set(float("1e12"))
#     j.CreateBreakTorqueAttr().Set(float("1e12"))


#     # (オプション: 角度制限も UsdPhysics (j) 側で設定するのが一般的です)
#     # lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "angular")
#     # lim.CreateLowAttr().Set(0.0)
#     # lim.CreateHighAttr().Set(float(90))
#     # if hasattr(lim, "CreateEnabledAttr"):
#     #     lim.CreateEnabledAttr().Set(True)

#     # cone = UsdPhysics.ConeLimitAPI.Apply(j.GetPrim(), "angular")
#     # cone.CreateAngleAttr().Set(8.0)           # 例: 8°
#     # # ある環境では EnableAttr がある場合のみ
#     # if hasattr(cone, "CreateEnabledAttr"):
#     #     cone.CreateEnabledAttr().Set(True)

#     # stone_prim = stage.GetPrimAtPath(body0_path)
#     # stone_prim.CreateAttribute("physics:angularDamping", Sdf.ValueTypeNames.Float).Set(1.0)
#     # stone_prim.CreateAttribute("physics:linearDamping",  Sdf.ValueTypeNames.Float).Set(1.05)


# def _xform_world(stage, path):
#     return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)

# def _to_local_point(T_world_of_frame, p_world_vec3f):
#     return T_world_of_frame.GetInverse().Transform(Gf.Vec3f(p_world_vec3f))

# def _create_spherical_to_world(stage, joint_path, body0_path,
#                                local0_pos=Gf.Vec3f(0,0,-0.12)):
#     # 1) joint 定義（Body1 は作らない＝世界固定）
#     j = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(joint_path))
#     j.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])

#     # 2) アンカーの世界座標（body0 ローカル → 世界）
#     T_w_b0 = _xform_world(stage, body0_path)
#     anchor_world = T_w_b0.Transform(local0_pos)   # Gf.Vec3f

#     # 3) joint の “親” ローカルに変換して translate を設定
#     joint_prim = j.GetPrim()
#     parent_prim = joint_prim.GetParent()
#     T_w_parent = _xform_world(stage, parent_prim.GetPath().pathString)
#     translate_in_parent = _to_local_point(T_w_parent, anchor_world)  # ← これが重要
#     UsdGeom.XformCommonAPI(joint_prim).SetTranslate(Gf.Vec3d(translate_in_parent))

#     # 4) ローカルアンカー
#     j.CreateLocalPos0Attr().Set(local0_pos)       # body0 側
#     j.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))  # 世界固定側は joint 原点

#     # 角度制限（属性があれば）
#     if hasattr(j, "CreateConeAngle0Attr"): j.CreateConeAngle0Attr().Set(5.0)
#     if hasattr(j, "CreateConeAngle1Attr"): j.CreateConeAngle1Attr().Set(5.0)


GROUP = 1
MASK32 = 0xFFFFFFFE   # (= ~1 & 0xFFFFFFFF)

def _apply_group_mask_to_shapes_under(stage, rigid_root_path: str):
    n = 0
    root = stage.GetPrimAtPath(rigid_root_path)
    if not root: return 0
    for p in Usd.PrimRange.AllPrims(root):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            # 既存が Int なら 16bit へフォールバック（下のBに委ねる）
            a = p.GetAttribute("physxCollision:mask")
            if a and a.GetTypeName() == Sdf.ValueTypeNames.Int:
                _apply_group_mask_16bit(p)  # ←下で定義
            else:
                # group/mask を UInt で作成
                p.CreateAttribute("physxCollision:group", Sdf.ValueTypeNames.UInt).Set(GROUP)
                p.CreateAttribute("physxCollision:mask",  Sdf.ValueTypeNames.UInt).Set(MASK32)
                # 互換名も一応セット
                p.CreateAttribute("physxCollision:filterGroup", Sdf.ValueTypeNames.UInt).Set(GROUP)
                p.CreateAttribute("physxCollision:filterMask",  Sdf.ValueTypeNames.UInt).Set(MASK32)
            n += 1
    return n

def _apply_group_mask_16bit(shape_prim):
    """既に Int型で作られている等、UIntが使えない場合のフォールバック"""
    from pxr import Sdf
    GROUP = 1
    MASK16 = 0xFFFE      # (= ~1 & 0xFFFF) → Int で安全に入る
    shape_prim.CreateAttribute("physxCollision:group", Sdf.ValueTypeNames.Int).Set(GROUP)
    shape_prim.CreateAttribute("physxCollision:mask",  Sdf.ValueTypeNames.Int).Set(MASK16)
    # 互換名
    shape_prim.CreateAttribute("physxCollision:filterGroup", Sdf.ValueTypeNames.Int).Set(GROUP)
    shape_prim.CreateAttribute("physxCollision:filterMask",  Sdf.ValueTypeNames.Int).Set(MASK16)


# def wire_env_stones(env_index: int,
#                     env_origin_offset: Gf.Vec3f,
#                     env_ns_template="/World/envs/env_{env}",
#                     stone_name_prefix="Stone_",
#                     joints_root_rel="fragile/Joints",
#                     ground_rel=None):  # None=世界固定
#     stage = omni.usd.get_context().get_stage()
#     env_ns = env_ns_template.format(env=env_index).rstrip("/")
#     joints_root = f"{env_ns}/{joints_root_rel}".rstrip("/")


#     # 書込み先はRootLayer
#     stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

#     # 石を列挙：env直下の子で、名前が Stone_* かつ RigidBody を持つもの
#     env_prim = stage.GetPrimAtPath(env_ns)
#     stones = []
#     for c in env_prim.GetChildren():
#         name = c.GetName()
#         if not name.startswith(stone_name_prefix):
#             continue
#         if _has_any_rigid(c):
#             stones.append(c.GetPath().pathString)

#     if not stones:
#         print(f"[wire] no stones matched under {env_ns} with prefix '{stone_name_prefix}'")
#         return

#     # joints 置き場
#     if not stage.GetPrimAtPath(joints_root):
#         stage.DefinePrim(Sdf.Path(joints_root), "Xform")

#     # 1. 石のローカルアンカー (底面)
#     local0_pos_vec = Gf.Vec3f(0, 0, -0.12) # (変更なし)

#     # 2. 親である「環境」のローカルトランスフォーム（オフセット）を取得
#     env_xform = UsdGeom.Xformable(env_prim)
#     # GetLocalTransformation は親(/World)からの相対的な変位 (オフセット) を返す
#     # env_local_matrix = env_xform.GetLocalTransformation(Usd.TimeCode.Default())
#     # env_local_matrix_d = Gf.Matrix4d(env_local_matrix) # double精度に


#     # 作成
#     # ★ 渡された信頼できるオフセットを行列に変換
#     env_offset_matrix_d = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(env_origin_offset))

#     # group_path = f"{env_ns}/CollisionGroups/Stones"           # envごとに分ける例
#     # grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
#     # to_include = [] 

#     shape_count = 0 

#     # 作成
#     for i, stone_path_str in enumerate(stones):
#         jp = f"{joints_root}/joint_{i:04d}"
        
#         stone_prim = stage.GetPrimAtPath(stone_path_str)
#         if not stone_prim.IsValid(): continue

#         # 3. 石のローカルトランスフォーム (親 env_prim からの相対)
#         # (これは信頼できる)
#         stone_xform = UsdGeom.Xformable(stone_prim)
#         stone_local_matrix = stone_xform.GetLocalTransformation(Usd.TimeCode.Default())
#         stone_local_matrix_d = Gf.Matrix4d(stone_local_matrix)

#         # 4. 石の「正しい」ワールドトランスフォームを合成
#         #    world_transform = (石のローカル) * (envの信頼できるオフセット)
#         xf_matrix = stone_local_matrix_d * env_offset_matrix_d

#         # 5. 石のローカルアンカーを「正しい」ワールド座標に変換
#         world1_pos_vec_double = xf_matrix.Transform(Gf.Vec3d(local0_pos_vec))
#         world1_pos_vec = Gf.Vec3f(world1_pos_vec_double)

#         # 6. ジョイント作成 (計算した値を渡す)
#         # (前回No.42で定義した _create_locklike_spherical_joint を呼ぶ)
#         _create_locklike_spherical_joint(
#             stage = stage, 
#             joint_path = jp, 
#             body0_path = stone_path_str,                       # (None が渡される)
#             local0_pos=local0_pos_vec,   # 石のローカル側
#             local1_pos=world1_pos_vec    # ★計算した正しいワールド側
#         )

#         # ★ 衝突無効化のために、グループへ追加する対象を集める
#         # to_include.append(Sdf.Path(stone_path_str))

#         shape_count += _apply_group_mask_to_shapes_under(stage, stone_path_str)
        
#     print(f"[wire] joints created: {len(stones)}  env={env_ns} ")

#     # # ループの後で一括設定（SetTargets は上書きなので一度にやるのが安全）
#     # includes = grp.GetIncludesRel()
#     # includes.SetTargets(to_include)

#     # # 同一グループ同士の衝突を無効化（自己参照をフィルタに設定）
#     # filtered = grp.GetFilteredGroupsRel()
#     # filtered.SetTargets([Sdf.Path(group_path)])



# def disable_stone_to_stone_collision(stage, stone_paths, group_path):
#     grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))

#     # includes に石を登録（関係が未作成でも Get...Rel() で取得→SetTargets すればOK）
#     includes = grp.GetIncludesRel()
#     includes.SetTargets([Sdf.Path(p) for p in stone_paths])

#     # 同一グループ同士の衝突を無効化（自己参照）
#     filtered = grp.GetFilteredGroupsRel()
#     filtered.SetTargets([Sdf.Path(group_path)])



def disable_collisions(scene, stones, group_path="/World/CollisionGroups/stone_group"):
    """
    - scene: InteractiveScene
    - stones: RigidObjectCollection（大量の石）
    """
    stage = omni.usd.get_context().get_stage()

    # 1) 衝突グループ prim を作成（インスタンス外に置く）
    group = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
    coll_api = group.GetCollidersCollectionAPI()  # UsdCollectionAPI
    includes_rel = coll_api.GetIncludesRel()

    # 2) 全環境・全石から「CollisionAPI が付いている prim」を集め、コレクションに追加
    def env_ns(i): return f"/World/envs/env_{i}"
    with Sdf.ChangeBlock():
        for ei in range(scene.num_envs):
            for name in stones.object_names:
                # RigidObjectCollectionCfg の prim_path から実パスに解決
                expr = stones.cfg.rigid_objects[name].prim_path  # 例: "{ENV_REGEX_NS}/Stones/Stone_0001"
                stone_root = stage.GetPrimAtPath(expr.replace("{ENV_REGEX_NS}", env_ns(ei)))
                if not stone_root or not stone_root.IsValid():
                    continue
                # 石配下の「CollisionAPI を持つ prim」を列挙して includes に登録
                for p in Usd.PrimRange(stone_root):
                    if p.HasAPI(UsdPhysics.CollisionAPI):
                        includes_rel.AddTarget(p.GetPath())

        # 3) グループに「自分自身」をフィルタ対象として登録 → グループ内衝突を無効化
        group.GetFilteredGroupsRel().AddTarget(group.GetPrim().GetPath())



def _iter_colliders_under(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    # ★ instance proxy も横断
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            yield p.GetPath()

# def build_stone_collision_group(scene, stones, group_path="/World/CollisionGroups/stone_group"):
#     """
#     stones: RigidObjectCollection（大量の飛び石）
#     """
#     stage = omni.usd.get_context().get_stage()
#     group = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
#     includes_rel = group.GetCollidersCollectionAPI().GetIncludesRel()

#     def env_ns(i): return f"/World/envs/env_{i}"
#     with Usd.EditContext(stage, stage.GetEditTarget()), Sdf.ChangeBlock():
#         includes_rel.ClearTargets()
#         for ei in range(scene.num_envs):
#             for name in stones.object_names:
#                 expr = stones.cfg.rigid_objects[name].prim_path
#                 stone_root = expr.replace("{ENV_REGEX_NS}", env_ns(ei))
#                 for coll_path in _iter_colliders_under(stage, stone_root):
#                     includes_rel.AddTarget(coll_path)

#     # グループ内衝突を禁止（A vs A）
#     group.GetFilteredGroupsRel().AddTarget(group.GetPrim().GetPath())



def build_stone_collision_group(scene, stones, group_path="/World/CollisionGroups/stone_group"):
    stage = omni.usd.get_context().get_stage()
    # stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
    includes_rel = grp.GetCollidersCollectionAPI().GetIncludesRel()

    def env_ns(i): return f"/World/envs/env_{i}"

    # 収集して最後に一括 SetTargets
    targets = []
    for ei in range(scene.num_envs):
        for name in stones.object_names:
            expr = stones.cfg.rigid_objects[name].prim_path
            stone_root = expr.replace("{ENV_REGEX_NS}", env_ns(ei))

            # stone_roots = list_actual_stones(env_ns(ei))

            for coll_path in _iter_colliders_under(stage, stone_root):
                targets.append(coll_path)

            # uninstance_all(stone_roots)

            # created = sum(ensure_collision_shapes_under(p) for p in stone_roots)
            # print(f"[colliders] created={created}")

            # d) 実体の衝突Shapeを集める
            # shape_paths = collect_real_shapes(stage, stone_roots)

            # # e) CollisionGroup（効く版ならここで件数が>0になる）
            # set_collision_group(stage, "/World/CollisionGroups/stone_group", shape_paths)



    # ここが修正ポイント
    includes_rel.SetTargets(targets)         # ← ClearTargetsは不要
    # try:
    grp.GetFilteredGroupsRel().SetTargets([grp.GetPrim().GetPath()])


    # debug_count_colliders_in_group(stage, group_path)

    diag_collision_group(stage, group_path)

    # except Exception:
    #     pass


def list_actual_stones(env_path:str, name_prefix="Stone_"):
    """env直下の Stone_* で RigidBody を持つものを列挙"""
    stage = omni.usd.get_context().get_stage()
    env = stage.GetPrimAtPath(env_path)
    out = []
    for c in env.GetChildren():
        if c.GetName().startswith(name_prefix) and c.HasAPI(UsdPhysics.RigidBodyAPI):
            out.append(c.GetPath().pathString)
    print(f"[stones] env={env_path} found={len(out)}")
    return out


import omni.kit.commands as okc
from pxr import Usd, UsdPhysics, Sdf

def uninstance_all(paths):
    if paths:
        okc.execute("UninstancePrims", paths=[p for p in paths])

def ensure_collision_shapes_under(root_path:str) -> int:
    """見えるジオメトリに CollisionAPI を適用（無ければ作る）。作成数を返す"""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    made = 0
    for p in Usd.PrimRange.AllPrims(root):
        t = p.GetTypeName()
        if t in ("Mesh","Cube","Sphere","Capsule","Cylinder"):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(p)
                p.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
                made += 1
    return made








def _iter_collider_paths2(stage: Usd.Stage, root_prim_path: str):
    """root_prim_path 配下の『CollisionAPI を持つ prim』を、インスタンス配下も含めて列挙。"""
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return
    # ★ インスタンス配下も横断（TraverseInstanceProxies）
    queue = [root]
    while queue:
        p = queue.pop(0)
        # CollisionAPI が付いた prim だけを登録対象にするのが仕様
        if p.HasAPI(UsdPhysics.CollisionAPI):
            yield p.GetPath()
        # インスタンス配下まで含めて子を列挙
        for c in p.GetFilteredChildren(Usd.TraverseInstanceProxies()):
            queue.append(c)

def build_stone_collision_group2(scene, stones, group_path="/World/CollisionGroups/stone_group"):
    """
    - scene: InteractiveScene
    - stones: RigidObjectCollection（大量の飛び石）
    """
    stage = omni.usd.get_context().get_stage()
    group = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
    coll_api = group.GetCollidersCollectionAPI()
    includes_rel = coll_api.GetIncludesRel()

    # '{ENV_REGEX_NS}' を「環境正規表現」に展開（例: /World/envs/env_.*）
    env_regex = "/World/envs/env_.*"

    # まず空にしてから、全コライダ prim を includes に追加
    # includes_rel.ClearTargets()
    targets = []
    with Usd.EditContext(stage, stage.GetEditTarget()), Sdf.ChangeBlock():
        for name, ro_cfg in stones.cfg.rigid_objects.items():
            # ro_cfg.prim_path は正規表現である可能性がある
            expr = ro_cfg.prim_path.replace("{ENV_REGEX_NS}", env_regex)
            # 1) 正規表現を『実パス』群に解決（Isaac Lab のユーティリティを使用）
            stone_root_paths = sim_utils.find_matching_prim_paths(expr, stage)
            # 2) 各 root の配下から CollisionAPI を持つ prim を収集
            for root_path in stone_root_paths:
                for coll_path in _iter_collider_paths2(stage, root_path):
                    # includes_rel.AddTarget(coll_path)
                    targets.append(coll_path)

        # 3) 自己フィルタを入れる → 同一グループ内の衝突を遮断（石⇔石が切れる）
        # group.GetFilteredGroupsRel().ClearTargets()
        includes_rel.SetTargets(targets)  
        group.GetFilteredGroupsRel().SetTargets([group.GetPrim().GetPath()])
        # group.GetFilteredGroupsRel().AddTarget(group.GetPrim().GetPath())

    # --- デバッグ: 何件入ったか可視化（0 件ならどこかで見落とし）
    q = coll_api.ComputeMembershipQuery()
    count = 0
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and q.IsPathIncluded(p.GetPath()):
            count += 1
    print(f"[stone_group] registered colliders = {count}")

    diag_collision_group(stage, group_path)





def probe(root):
    exist=proxy=real_no_col=real_col=0
    for p in Usd.PrimRange.AllPrims(stage.GetPrimAtPath(root)):
        if not p or not p.IsValid(): continue
        exist+=1
        if p.IsInstanceProxy():
            proxy+=1
        else:
            if p.HasAPI(UsdPhysics.CollisionAPI): real_col+=1
            else: real_no_col+=1
    print(f"[probe] exist={exist} proxy={proxy} real_no_col={real_no_col} real+CollisionAPI={real_col}")





def collect_real_shapes(stage, stone_roots):
    shapes = []
    for rp in stone_roots:
        root = stage.GetPrimAtPath(rp)
        for p in Usd.PrimRange.AllPrims(root):
            if (not p.IsInstanceProxy()) and p.HasAPI(UsdPhysics.CollisionAPI):
                shapes.append(p.GetPath())
    print(f"[collect] shapes={len(shapes)}")
    return shapes

def set_collision_group(stage, group_path, shape_paths):
    grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
    rel = grp.GetCollidersCollectionAPI().GetIncludesRel()
    rel.SetTargets(shape_paths)  # ←一括上書き
    try:
        grp.GetFilteredGroupsRel().SetTargets([grp.GetPrim().GetPath()])  # 自己参照で同士無効
    except Exception:
        pass
    # デバッグ
    q = grp.GetCollidersCollectionAPI().ComputeMembershipQuery()
    cnt = 0
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and q.IsPathIncluded(p.GetPath()):
            cnt += 1
    print(f"[cg] membership={cnt}")

def force_physx_filtered_pairs(stage, shape_paths):
    scene_prim = stage.GetPrimAtPath("/World/physicsScene") or next(
        (p for p in stage.Traverse() if p.GetTypeName()=="PhysicsScene"), None)
    if not scene_prim:
        print("[physx] no PhysicsScene"); return
    api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    rel = api.GetPrim().GetRelationship("physx:filteredPairs") or api.GetPrim().CreateRelationship("physx:filteredPairs")
    flat=[]; N=len(shape_paths)
    for i in range(N):
        a=Sdf.Path(shape_paths[i])
        for j in range(i+1,N):
            b=Sdf.Path(shape_paths[j]); flat.extend([a,b])
    rel.SetTargets(flat)
    print(f"[physx] filteredPairs set: {len(flat)//2} pairs")




def debug_count_colliders_in_group(stage, group_path):
    group = UsdPhysics.CollisionGroup.Get(stage, Sdf.Path(group_path))
    coll_api = group.GetCollidersCollectionAPI()
    # 収集結果の解決（コレクション）
    q = coll_api.ComputeMembershipQuery()
    count = 0
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and q.IsPathIncluded(p.GetPath()):
            count += 1
    print(f"[CollisionGroup] members = {count}")




def diag_collision_group(stage, group_path):
    grp = UsdPhysics.CollisionGroup.Get(stage, Sdf.Path(group_path))
    if not grp:
        print("[diag] group not found:", group_path); return
    coll_api = grp.GetCollidersCollectionAPI()
    rel = coll_api.GetIncludesRel()
    targets = rel.GetTargets()
    print(f"[diag] includes targets authored: {len(targets)}")
    if targets[:5]:
        print("       first targets:", [str(t) for t in targets[:5]])

    # 実体/Proxy/CollisionAPI の内訳
    n_exist = n_proxy = n_real = n_collision = 0
    for t in targets:
        p = stage.GetPrimAtPath(t)
        if not p or not p.IsValid(): 
            continue
        n_exist += 1
        if p.IsInstanceProxy():
            n_proxy += 1
        else:
            n_real += 1
            if p.HasAPI(UsdPhysics.CollisionAPI):
                n_collision += 1
    print(f"[diag] exist={n_exist}, proxy={n_proxy}, real={n_real}, real+CollisionAPI={n_collision}")

    # 最終的に membership に入っている数
    q = coll_api.ComputeMembershipQuery()
    m = 0
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and q.IsPathIncluded(p.GetPath()):
            m += 1
    print(f"[diag] membership (resolved) = {m}")



# def _uninstance_stones_under(stage, env_path: str, stone_prefix="Stone_"):
#     """env 直下の Stone_* をデインスタンス（インスタンス→実体化）"""
#     env = stage.GetPrimAtPath(env_path)
#     if not env: return 0
#     to_uninst = []
#     for c in env.GetChildren():
#         if c.GetName().startswith(stone_prefix) and c.IsInstance():
#             to_uninst.append(c.GetPath())
#     if to_uninst:
#         okc.execute("UninstancePrims", paths=to_uninst)
#     return len(to_uninst)

# def _iter_real_colliders_under(stage, root_path: str):
#     """CollisionAPI を持つ“実体”shape だけ列挙（proxy 除外）"""
#     root = stage.GetPrimAtPath(root_path)
#     if not root: return []
#     out = []
#     for p in Usd.PrimRange.AllPrims(root):
#         if p and (not p.IsInstanceProxy()) and p.HasAPI(UsdPhysics.CollisionAPI):
#             out.append(p.GetPath())
#     return out

# def _force_physx_filtered_pairs(stage, shape_paths):
#     """保険：PhysX シーンの filteredPairs に石同士の全ペアをセット"""
#     scene_prim = stage.GetPrimAtPath("/World/physicsScene") or next(
#         (p for p in stage.Traverse() if p.GetTypeName()=="PhysicsScene"), None)
#     if not scene_prim: 
#         print("[physx] no PhysicsScene; skipped"); 
#         return
#     api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
#     rel = api.GetPrim().GetRelationship("physx:filteredPairs") or api.GetPrim().CreateRelationship("physx:filteredPairs")
#     pairs_flat = []
#     N = len(shape_paths)
#     for i in range(N):
#         a = Sdf.Path(shape_paths[i])
#         for j in range(i+1, N):
#             b = Sdf.Path(shape_paths[j])
#             pairs_flat.extend([a, b])
#     rel.SetTargets(pairs_flat)
#     print(f"[physx] filteredPairs set: {len(pairs_flat)//2} pairs")

# def build_stone_collision_group(scene, stones, group_path="/World/CollisionGroups/stone_group"):
#     stage = omni.usd.get_context().get_stage()
#     stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

#     # 1) 各 env をデインスタンス
#     total_uninst = 0
#     for ei in range(scene.num_envs):
#         env_path = f"/World/envs/env_{ei}"
#         total_uninst += _uninstance_stones_under(stage, env_path, stone_prefix="Stone_")
#     if total_uninst:
#         print(f"[cg] uninstanced {total_uninst} stone prims")

#     # 2) 実体コライダ（Shape）を収集
#     def env_ns(i): return f"/World/envs/env_{i}"
#     shape_targets = []
#     for ei in range(scene.num_envs):
#         for name in stones.object_names:
#             expr = stones.cfg.rigid_objects[name].prim_path  # 例 "{ENV_REGEX_NS}/Stone_0000"
#             stone_root = expr.replace("{ENV_REGEX_NS}", env_ns(ei))
#             shape_targets += _iter_real_colliders_under(stage, stone_root)

#     print(f"[cg] collected shapes: {len(shape_targets)}")

#     # 3) CollisionGroup に一括登録（この版は includes/filteredGroups が動く場合のみ有効）
#     # try:
#     grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(group_path))
#     includes_rel = grp.GetCollidersCollectionAPI().GetIncludesRel()
#     includes_rel.SetTargets(shape_targets)
#     # 自己参照で“同士を無効化”
#     grp.GetFilteredGroupsRel().SetTargets([grp.GetPrim().GetPath()])
#     # デバッグ：メンバー数を確認
#     q = grp.GetCollidersCollectionAPI().ComputeMembershipQuery()
#     cnt = 0
#     for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
#         if p.HasAPI(UsdPhysics.CollisionAPI) and q.IsPathIncluded(p.GetPath()):
#             cnt += 1
#     print(f"[cg] CollisionGroup members (resolved) = {cnt}")
#     # except Exception as e:
#     #     print(f"[cg] CollisionGroup not supported on this build: {e}")

#     # 4) 保険：必ず効かせるために PhysX filteredPairs も設定
#     _force_physx_filtered_pairs(stage, shape_targets)





# def _find_physics_scene(stage):
#     # 代表的な場所に無ければタイプ名でスキャン
#     return (stage.GetPrimAtPath("/World/physicsScene") 
#             or next((p for p in stage.Traverse() if p.GetTypeName()=="PhysicsScene"), None))

# def _list_stone_shapes_in_env(stage, env_path: str, stone_prefix="Stone_"):
#     """env 直下の Stone_* 配下にある CollisionAPI 付き shape のパスを列挙（実体のみ）"""
#     shapes = []
#     env = stage.GetPrimAtPath(env_path)
#     if not env: return shapes
#     for stone in env.GetChildren():
#         if not stone.GetName().startswith(stone_prefix):
#             continue
#         # 石配下の shape を収集
#         for p in Usd.PrimRange.AllPrims(stone):
#             if (not p.IsInstanceProxy()) and p.HasAPI(UsdPhysics.CollisionAPI):
#                 shapes.append(p.GetPath())
#     return shapes

# def disable_stone_to_stone_collisions_via_physx_pairs(env_paths):
#     stage = omni.usd.get_context().get_stage()
#     scene_prim = _find_physics_scene(stage)
#     if not scene_prim:
#         print("[physx] ERROR: PhysicsScene not found"); 
#         return

#     api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
#     rel = (api.GetPrim().GetRelationship("physx:filteredPairs") 
#            or api.GetPrim().CreateRelationship("physx:filteredPairs"))

#     # 環境ごとに“同じ環境内の石どうし”だけを無効化（ペア数を抑える）
#     targets_flat = []
#     total_pairs = 0
#     for env_path in env_paths:
#         shapes = _list_stone_shapes_in_env(stage, env_path)
#         N = len(shapes)
#         for i in range(N):
#             a = Sdf.Path(shapes[i])
#             for j in range(i+1, N):
#                 b = Sdf.Path(shapes[j])
#                 targets_flat.extend([a, b])
#         total_pairs += (N*(N-1))//2

#     # 既存ターゲットを置き換え（増殖防止）
#     rel.SetTargets(targets_flat)
#     print(f"[physx] filteredPairs set: {total_pairs} pairs across {len(env_paths)} env(s)")



from itertools import combinations


def _find_physics_scene(stage):
    # 代表的な場所に無ければタイプで探索
    return (stage.GetPrimAtPath("/World/physicsScene")
            or next((p for p in stage.Traverse() if p.GetTypeName()=="PhysicsScene"), None))

def _list_stone_shapes_in_env(stage, env_path: str, stone_prefix="Stone_"):
    """env 直下の Stone_* 配下にある '実体' の衝突形状(=CollisionAPI付きprim)を列挙"""
    shapes = []
    env = stage.GetPrimAtPath(env_path)
    if not env: 
        return shapes
    for stone in env.GetChildren():
        if not stone.GetName().startswith(stone_prefix):
            continue
        for p in Usd.PrimRange.AllPrims(stone):
            # instance proxy は除外
            if (not p.IsInstanceProxy()) and p.HasAPI(UsdPhysics.CollisionAPI):
                shapes.append(p.GetPath().pathString)
    return shapes

def _apply_group_mask_to_shapes(stage, shape_paths, group_id=1, allow_robot_ground_mask=0b110, prefer_uint=True):
    """フォールバック：各 Shape に group/mask を直書き（RigidBodyではなく Shape に！）"""
    from pxr import Sdf
    for sp in shape_paths:
        prim = stage.GetPrimAtPath(sp)
        if prefer_uint:
            prim.CreateAttribute("physxCollision:group", Sdf.ValueTypeNames.UInt).Set(group_id)
            prim.CreateAttribute("physxCollision:mask",  Sdf.ValueTypeNames.UInt).Set(allow_robot_ground_mask)
            # 互換名も一応
            prim.CreateAttribute("physxCollision:filterGroup", Sdf.ValueTypeNames.UInt).Set(group_id)
            prim.CreateAttribute("physxCollision:filterMask",  Sdf.ValueTypeNames.UInt).Set(allow_robot_ground_mask)
        else:
            prim.CreateAttribute("physxCollision:group", Sdf.ValueTypeNames.Int).Set(group_id)
            prim.CreateAttribute("physxCollision:mask",  Sdf.ValueTypeNames.Int).Set(allow_robot_ground_mask)

def disable_stone_to_stone_collisions_filtered(env_paths, stone_prefix="Stone_"):
    """
    env_paths: 例 ["/World/envs/env_0", "/World/envs/env_1", ...]
    - まず UsdPhysics.FilteredPairsAPI で Stone↔Stone の全ペアを登録（推奨手段）
    - API が無ければ、石の Shape に group/mask を直書きするフォールバックに切替
    """
    stage = omni.usd.get_context().get_stage()

    # 1) 環境ごとに石の Shape を収集（同一env内の石どうしのみを遮断してペア数を節約）
    shapes_all = []
    pairs = []
    for env in env_paths:
        shapes = _list_stone_shapes_in_env(stage, env, stone_prefix=stone_prefix)
        shapes_all.extend(shapes)
        if len(shapes) >= 2:
            pairs.extend((Sdf.Path(a), Sdf.Path(b)) for a, b in combinations(shapes, 2))

    print(f"[filtered] envs={len(env_paths)} shapes={len(shapes_all)} pairs={len(pairs)}")

    if not shapes_all:
        print("[filtered] WARNING: no stone shapes found (CollisionAPI未付与 or パス誤り)")
        return False

    # 2) 推奨：FilteredPairsAPI があればそれを使う（重複管理をAPI側に任せられる）
    scene_prim = _find_physics_scene(stage)
    if scene_prim and hasattr(UsdPhysics, "FilteredPairsAPI"):
        api = UsdPhysics.FilteredPairsAPI.Apply(scene_prim)
        try:
            if hasattr(api, "SetFilteredPairs"):
                api.SetFilteredPairs(pairs)
                print(f"[filtered] UsdPhysics.FilteredPairsAPI.SetFilteredPairs ok ({len(pairs)} pairs)")
                return True
            elif hasattr(api, "AddFilteredPairs"):
                # 既存を消してから追加（関数が無いビルドもあるので try）
                try:
                    rel = api.CreateFilteredPairsRel()  # リレーションを直で取得
                    # ClearTargets は removeSpec を要求する版があるため空セットで上書き
                    rel.SetTargets([])
                except Exception:
                    pass
                api.AddFilteredPairs(pairs)
                print(f"[filtered] UsdPhysics.FilteredPairsAPI.AddFilteredPairs ok ({len(pairs)} pairs)")
                return True
        except Exception as e:
            print(f"[filtered] FilteredPairsAPI failed ({type(e).__name__}): {e}")

    # 3) フォールバック：石の Shape に group/mask を直書き（石同士は当たらない）
    _apply_group_mask_to_shapes(stage, shapes_all, group_id=1, allow_robot_ground_mask=0b110, prefer_uint=True)
    print("[filtered] Fallback: group/mask written to stone shapes (UInt). "
          "必要なら Int/16bit に落として再適用してください。")
    return True








STONE_BIT = 0x0002  # 石のグループ用ビット

def _is_collision_prim(stage, prim):
    return bool(UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()))

def _value_type_uint_or_int():
    # UInt があれば使う。無ければ Int にフォールバック
    return Sdf.ValueTypeNames.UInt if hasattr(Sdf.ValueTypeNames, "UInt") else Sdf.ValueTypeNames.Int

def _set_attr_raw(prim, name, vtype, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, vtype, custom=False)
    attr.Set(value)

def _set_group_mask(stage, prim, group, mask):
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    if hasattr(api, "CreateCollisionGroupAttr") and hasattr(api, "CreateCollisionFilterMaskAttr"):
        api.CreateCollisionGroupAttr(int(group))
        api.CreateCollisionFilterMaskAttr(int(mask) & 0xFFFFFFFF)
    else:
        _set_attr_raw(prim, "physxCollision:collisionGroup", Sdf.ValueTypeNames.Int, int(group))
        _set_attr_raw(prim, "physxCollision:collisionFilterMask", _value_type_uint_or_int(), int(mask) & 0xFFFFFFFF)

def _force_collision_enabled_and_offsets(prim, enabled=True, rest=0.0, contact=0.02):
    # Creator があればそれを、無ければ生属性で強制セット
    coll = UsdPhysics.CollisionAPI.Apply(prim)
    if hasattr(coll, "CreateCollisionEnabledAttr"):
        coll.CreateCollisionEnabledAttr(bool(enabled))
    else:
        _set_attr_raw(prim, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, bool(enabled))

    if hasattr(coll, "CreateRestOffsetAttr") and hasattr(coll, "CreateContactOffsetAttr"):
        coll.CreateRestOffsetAttr(float(rest))
        coll.CreateContactOffsetAttr(float(contact))
    else:
        _set_attr_raw(prim, "physics:restOffset", Sdf.ValueTypeNames.Float, float(rest))
        _set_attr_raw(prim, "physics:contactOffset", Sdf.ValueTypeNames.Float, float(contact))

def set_stones_no_self_collision(stage, root="/World/envs", stone_prefix="Stone_"):
    """
    Stone_* 配下の衝突形状に:
      group = STONE_BIT
      mask  = 0xFFFFFFFF & ~STONE_BIT  （石どうし無効）
      collisionEnabled=True, rest/contact を健全値に固定
    """
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root):
            continue
        # Stone_*** 本体またはその子孫だけ対象
        names = p.split("/")
        if not any(n.startswith(stone_prefix) for n in names):
            continue
        if not _is_collision_prim(stage, prim):
            continue

        _set_group_mask(stage, prim, STONE_BIT, 0xFFFFFFFF & ~STONE_BIT)
        _force_collision_enabled_and_offsets(prim, enabled=True, rest=0.0, contact=0.02)



def dump(stage, path):
    prim = stage.GetPrimAtPath(path)
    coll = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
    g = prim.GetAttribute("physxCollision:collisionGroup").Get()
    m = prim.GetAttribute("physxCollision:collisionFilterMask").Get()
    print(f"\n== {path} ==")
    print("enabled:", coll and coll.GetCollisionEnabledAttr().Get())
    print("group  :", g, " mask:", hex(m) if m is not None else None)

















STONE_BIT  = 1 << 1
ALL_MASK32 = 0xFFFFFFFF
STONE_MASK_ALLOW_ALL_BUT_STONE = ALL_MASK32 ^ STONE_BIT  # 石以外は全部当てる（＝石↔石のみ拒否）

def _iter_collision_shapes(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    # 読み取りは proxy を含めてOK
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            yield p

def _get_authorable_shape_prim(shape_prim: Usd.Prim) -> Usd.Prim:
    """instance proxy なら Prototype 内の対応Primに切替。"""
    # 一部ビルドでは IsInstanceProxy が未定義のことがあるので hasattr チェック
    if hasattr(shape_prim, "IsInstanceProxy") and shape_prim.IsInstanceProxy():
        return shape_prim.GetPrimInPrototype()
    return shape_prim

def _set_group_mask(shape_prim: Usd.Prim, group: int, mask: int):
    # PhysX 衝突APIを prim に適用（これをやらないと無視されることがある）
    PhysxSchema.PhysxCollisionAPI.Apply(shape_prim)
    shape_prim.CreateAttribute("physxCollision:group", Sdf.ValueTypeNames.UInt).Set(int(group))
    shape_prim.CreateAttribute("physxCollision:mask",  Sdf.ValueTypeNames.UInt).Set(int(mask))
    # 互換キー（古いビルド向け）
    shape_prim.CreateAttribute("physxCollision:filterGroup", Sdf.ValueTypeNames.UInt).Set(int(group))
    shape_prim.CreateAttribute("physxCollision:filterMask",  Sdf.ValueTypeNames.UInt).Set(int(mask))

def _collect_stone_root_paths(scene, stones) -> list[str]:
    stage = omni.usd.get_context().get_stage()
    # object_prim_paths があれば使う
    opp = getattr(stones, "object_prim_paths", None)
    paths = []
    if opp:
        for p in opp:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                paths.append(p)
        if paths:
            return sorted(set(paths))
    # cfg からワイルドカード展開
    for _, ro_cfg in stones.cfg.rigid_objects.items():
        expr = ro_cfg.prim_path
        expr = expr.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        for p in sim_utils.find_matching_prim_paths(expr, stage):
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                paths.append(p)
    return sorted(set(paths))

def disable_stone_vs_stone(scene, stones_name: str = "stones"):
    stage = omni.usd.get_context().get_stage()
    stones = scene.rigid_object_collections[stones_name]
    stone_roots = _collect_stone_root_paths(scene, stones)
    if not stone_roots:
        print("[stone-filter] no stone roots found (prim_path を確認)")
        return

    wrote = 0
    seen_authorables = set()  # Prototype重複に備えて重複排除
    with Sdf.ChangeBlock():
        for root in stone_roots:
            for shp in _iter_collision_shapes(stage, root):
                tgt = _get_authorable_shape_prim(shp)
                pth = tgt.GetPath().pathString
                if pth in seen_authorables:
                    continue
                seen_authorables.add(pth)
                _set_group_mask(tgt, STONE_BIT, STONE_MASK_ALLOW_ALL_BUT_STONE)
                wrote += 1
    print(f"[stone-filter] wrote group/mask on authorable stone shapes: {wrote}")








def _iter_collider_paths(stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    # 形状(=CollisionAPI付き)の「パス」を列挙。インスタンス配下も含める
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            yield p.GetPath()

def _collect_env_stone_shape_paths(scene, env_i: int, stones_name="stones"):
    """env_i 配下の『石の形状Primパス』をすべて列挙（インスタンス対応・取りこぼし防止）。"""
    stage = omni.usd.get_context().get_stage()
    stones = scene.rigid_object_collections[stones_name]
    paths = set()
    # cfg の prim_path を /World/envs/env_i に解決してから、配下の形状を拾う
    for _, ro_cfg in stones.cfg.rigid_objects.items():
        expr = ro_cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_i}")
        for root in sim_utils.find_matching_prim_paths(expr, stage):
            for shp in _iter_collider_paths(stage, root):
                paths.add(shp.pathString)
    return sorted(paths)

def apply_stone_self_filter_per_env(scene, stones_name="stones", group_root="/World/collisions"):
    """各envに CollisionGroup を作り、グループ内(=同envの石)衝突を自己フィルタで遮断。"""
    stage = omni.usd.get_context().get_stage()
    for i in range(scene.num_envs):
        grp_path = f"{group_root}/stones_env_{i}"
        grp = UsdPhysics.CollisionGroup.Define(stage, Sdf.Path(grp_path))
        includes = grp.GetCollidersCollectionAPI().GetIncludesRel()
        # includes を石形状の実パスで上書き
        stone_shapes = _collect_env_stone_shape_paths(scene, i, stones_name=stones_name)
        includes.SetTargets([Sdf.Path(p) for p in stone_shapes])
        # 自分自身を filteredGroups に入れて「自己コリジョン無効化」
        grp.GetFilteredGroupsRel().SetTargets([Sdf.Path(grp_path)])
    print("[stone-filter] CollisionGroup self-filter applied for stones in each env")


def diag_stone_pair_filter(scene, stones_name="stones", env_i=0, sample_k=5):
    stage = omni.usd.get_context().get_stage()
    # env_i の石形状だけを診断
    stone_shapes = _collect_env_stone_shape_paths(scene, env_i, stones_name=stones_name)
    # 読みはプロキシでもOKだが、念のためプロトタイプ側も併用
    def read_gm(path):
        p = stage.GetPrimAtPath(path)
        # プロキシならプロトタイプへ
        if hasattr(p, "IsInstanceProxy") and p.IsInstanceProxy():
            p = p.GetPrimInPrototype()
        g = p.GetAttribute("physxCollision:group").Get()
        m = p.GetAttribute("physxCollision:mask").Get()
        return (g or 0), (m or 0)
    metas = [(path, *read_gm(path)) for path in stone_shapes]
    n = len(metas)
    bad = []
    for a in range(n):
        pa, ga, ma = metas[a]
        for b in range(a+1, n):
            pb, gb, mb = metas[b]
            collide = (ga & mb) != 0 and (gb & ma) != 0
            if collide:
                bad.append((pa, pb, ga, ma, gb, mb))
                if len(bad) >= sample_k:
                    break
        if len(bad) >= sample_k:
            break
    if bad:
        print(f"[diag] BAD pairs in env_{env_i}: {len(bad)} (sample below)")
        for (pa, pb, ga, ma, gb, mb) in bad[:sample_k]:
            print("  A:", pa, " g/m=", ga, ma)
            print("  B:", pb, " g/m=", gb, mb)
    else:
        print(f"[diag] All stone pairs in env_{env_i} are filtered as expected.")




import types



import numpy as np

def quantize(v: float, h: float) -> float:
    """horizontal_scale の格子に合わせて丸める"""
    return round(v / h) * h

def rects_intersect(ax0, ax1, ay0, ay1, bx0, bx1, by0, by1) -> bool:
    return (ax0 < bx1) and (ax1 > bx0) and (ay0 < by1) and (ay1 > by0)


def snap_up(v, h):   return np.ceil(v / h) * h
def snap_down(v, h): return np.floor(v / h) * h
def snap_near(v, h): return np.round(v / h) * h


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def params_from_difficulty(
    difficulty: float,
    horizontal_scale: float,
    stone_width_range_m: tuple[float, float],      # (max, min) 例: (0.40, 0.20)
    stone_distance_range_m: tuple[float, float],   # (min, max) 例: (0.10, 0.35)
) -> tuple[int, int, float, float]:
    """
    IsaacLab の stepping-stones と同じ発想:
      - 幅は difficulty↑で小さく: max -> min
      - 距離は difficulty↑で大きく: min -> max
    その後 int(.../h) でピクセル化（切り捨て）する。
    """
    d = clamp01(float(difficulty))
    w_max, w_min = stone_width_range_m
    dist_min, dist_max = stone_distance_range_m

    w_m    = lerp(w_max,  w_min,  d)      # 幅は減る
    dist_m = lerp(dist_min, dist_max, d)  # 距離は増える

    w_px    = max(1, int(w_m / horizontal_scale))
    dist_px = max(0, int(dist_m / horizontal_scale))

    # “実際に使われる（=ピクセルに落ちた）”メートル値
    w_eff_m    = w_px * horizontal_scale
    dist_eff_m = dist_px * horizontal_scale
    return w_px, dist_px, w_eff_m, dist_eff_m




# def generate_xy_list_front_isaac(
#     terrain_size_xy=(8.0, 8.0),     # (Lx, Ly) [m] そのタイルの大きさ
#     horizontal_scale=0.05,          # [m] Terrain HF と揃えたい格子
#     stone_size_xy=(0.35, 0.25),     # (sx, sy) [m] ブロック天板サイズ（XY）
#     gap_xy=(0.15, 0.15),            # (gx, gy) [m] ブロック間ギャップ
#     platform_size=1.2,              # 中央台の一辺 [m]（正方形を仮定）
#     platform_center=(0.0, 0.0),     # 台中心（普通は (0,0)）
#     x_front_ratio=0.5,              # 前半のみ = 0.5（x>0 側）
#     margin=0.10,                    # 端からの余白 [m]
#     clearance=0.02,                 # 台との追加クリアランス [m]
#     per_row_phase=True,             # 行ごとに位相ずらし（stepping-stones風）
#     jitter_xy=(0.0, 0.0),           # (jx, jy) [m] 追加ランダム
#     seed=0,
# ):
#     rng = np.random.default_rng(seed)

#     Lx, Ly = terrain_size_xy
#     sx, sy = stone_size_xy
#     gx, gy = gap_xy
#     jx, jy = jitter_xy

#     # ピッチ（中心間距離）
#     px = sx + gx
#     py = sy + gy

#     # 台（platform）のAABB
#     pcx, pcy = platform_center
#     half_p = platform_size * 0.5
#     plat_x0 = pcx - half_p - clearance
#     plat_x1 = pcx + half_p + clearance
#     plat_y0 = pcy - half_p - clearance
#     plat_y1 = pcy + half_p + clearance

#     # 配置領域（中心座標で安全に収まる範囲）
#     x_min = 0.0 + margin + sx * 0.5
#     x_max = (Lx * x_front_ratio) - margin - sx * 0.5  # x>0 側だけ使う想定（原点が中心なら Lx/2 が前端）
#     y_min = -Ly * 0.5 + margin + sy * 0.5
#     y_max = +Ly * 0.5 - margin - sy * 0.5

#     # 格子に量子化（HFと揃えるなら推奨）
#     # x_min = quantize(x_min, horizontal_scale)
#     # x_max = quantize(x_max, horizontal_scale)
#     # y_min = quantize(y_min, horizontal_scale)
#     # y_max = quantize(y_max, horizontal_scale)

#     # スナップは min=ceil / max=floor
#     x_min = snap_up(x_min, horizontal_scale)
#     x_max = snap_down(x_max, horizontal_scale)
#     y_min = snap_up(y_min, horizontal_scale)
#     y_max = snap_down(y_max, horizontal_scale)


#     # px_q  = max(horizontal_scale, quantize(px, horizontal_scale))
#     # py_q  = max(horizontal_scale, quantize(py, horizontal_scale))

#     px_q = max(horizontal_scale, snap_near(px, horizontal_scale))
#     py_q = max(horizontal_scale, snap_near(py, horizontal_scale))

#     points = []

#     # y の帯（row）を走査
#     y = y_min
#     row = 0
#     while y <= y_max + 1e-9:
#         # 行ごとに x の開始位相をずらす（完全格子にしたいなら 0 に固定）
#         phase = rng.uniform(0.0, sx) if per_row_phase else 0.0
#         x = x_min + quantize(phase, horizontal_scale)

#         while x <= x_max + 1e-9:
#             # ジッター（必要なら）
#             xx = x + (rng.uniform(-jx, jx) if jx > 0 else 0.0)
#             yy = y + (rng.uniform(-jy, jy) if jy > 0 else 0.0)

#             # 格子に戻す（HFと揃える）
#             xx = quantize(xx, horizontal_scale)
#             yy = quantize(yy, horizontal_scale)

#             # 石のAABB（中心から）
#             stone_x0 = xx - sx * 0.5
#             stone_x1 = xx + sx * 0.5
#             stone_y0 = yy - sy * 0.5
#             stone_y1 = yy + sy * 0.5

#             # 中央台と交差する石は除外
#             if not rects_intersect(stone_x0, stone_x1, stone_y0, stone_y1,
#                                    plat_x0, plat_x1, plat_y0, plat_y1):
#                 points.append((xx, yy))

#             x += px_q

#         y += py_q
#         row += 1

#     # return np.asarray(points, dtype=np.float32)

#     return points



import math
import random
from typing import List, Tuple, Optional

def rect_intersect_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (a1 > b0)

def stepping_stones_xy_front_half_pixelwise(
    size_x_m: float,
    size_y_m: float,
    horizontal_scale: float,
    platform_width_m: float,
    difficulty: float,
    stone_width_range_m: tuple[float, float] = (0.40, 0.20),     # (max, min)
    stone_distance_range_m: tuple[float, float] = (0.10, 0.35),  # (min, max)
    margin_m: float = 0.00,              # 外周の安全マージン
    platform_clearance_m: float = 0.00,  # 台からさらに離したいなら +（0で“IsaacLabの台境界ぴったり”）
    per_row_phase: bool = True,          # 行ごとに開始位相をランダム化（IsaacLab風）
    seed: int = 0,
    outer_slack_m: float = 0.2,            # ★外側だけ余白（台側には入れない）
    max_points: Optional[int] = None,    # 石数を固定したいなら指定（足りない場合は返り値が少なくなる）
) -> tuple[List[Tuple[float, float]], dict]:
    """
    返り値:
      - xy: [(x_local, y_local), ...]  ※すべて Python float（OmegaConfでも安全）
      - meta: 実効 stone_size/gap などデバッグ情報
    """
    h = float(horizontal_scale)
    rng = random.Random(seed)

    # --- 1) IsaacLab と同じくピクセルで実効サイズを決める（intで切り捨て）
    W = int(size_x_m / h)   # x方向ピクセル数
    H = int(size_y_m / h)   # y方向ピクセル数
    size_x_eff = W * h
    size_y_eff = H * h

    cx = W // 2
    cy = H // 2

    # --- 2) difficulty から stone_width / stone_distance を決めてピクセル化
    w_px, dist_px, w_eff_m, dist_eff_m = params_from_difficulty(
        difficulty, h, stone_width_range_m, stone_distance_range_m
    )
    pitch_px = max(1, w_px + dist_px)

    # --- 3) platform も IsaacLab と同じピクセル境界で切る
    pf_px = int(platform_width_m / h)
    # IsaacLabっぽい中心切り出し（整数境界）
    px1 = (W - pf_px) // 2
    px2 = (W + pf_px) // 2
    py1 = (H - pf_px) // 2
    py2 = (H + pf_px) // 2

    # clearance を “ピクセル” で拡張（台を避けすぎるのが嫌なら 0 推奨）
    clear_px = int(platform_clearance_m / h)
    px1c, px2c = px1 - clear_px, px2 + clear_px
    py1c, py2c = py1 - clear_px, py2 + clear_px

    # --- 4) 外周マージンもピクセルで（はみ出しゼロを保証するため）
    margin_px = int(margin_m / h)

    # --- 5) まず「石パッチ左下(x0,y0)」をピクセルで走査
    #   y0 は下から上へ、x0 は “前半(x>=0)” 側だけ
    #
    # 重要: RigidObject は欠けられないので、パッチが完全に入る範囲だけにする
    #
    # ピクセルiのx座標(m) は (i - cx)*h とする（cxがx=0近辺）
    # “前半”は中心より右（i >= cx）だが、パッチ幅があるので中心側を少し避ける
    #
    # パッチが完全に入る条件: x0 >= 0側境界 かつ x0+w_px <= W-margin_px
    outer_slack_px = int(outer_slack_m / h)

    y0_min = margin_px+ outer_slack_px
    y0_max = H - margin_px - w_px- outer_slack_px
    # x0_min = max(cx, m・n_px - w_・px
    # x0_min = max(px2c, cx, margin_px)
    x0_min = max(px2c + 1, cx, margin_px)

    x0_max = W - margin_px - w_px- outer_slack_px

    xy: List[Tuple[float, float]] = []

    y0 = y0_min
    row = 0
    while y0 <= y0_max:
        # 行ごとに “開始位相” をランダムにズラす（0〜w_px-1）
        phase = rng.randrange(0, w_px) if (per_row_phase and w_px > 1) else 0
        x0 = x0_min + phase

        while x0 <= x0_max:
            # --- 台との交差（ピクセル矩形で判定）
            sx0, sx1 = x0, x0 + w_px
            sy0, sy1 = y0, y0 + w_px
            hit_platform = rect_intersect_1d(sx0, sx1, px1c, px2c) and rect_intersect_1d(sy0, sy1, py1c, py2c)

            if not hit_platform:
                # 石の中心（ピクセル）→ メートル
                # ここは “ピクセル中心”に寄せておくと分かりやすい
                xc = x0 + w_px * 0.5
                yc = y0 + w_px * 0.5
                x_m = (xc - cx) * h
                y_m = (yc - cy) * h

                # --- 最終安全チェック（メートルAABBで “はみ出しゼロ”）
                # 前半のみ: x - w/2 >= 0
                if (x_m - w_eff_m * 0.5) >= 0.0 + margin_px * h and \
                   (x_m + w_eff_m * 0.5) <= (size_x_eff * 0.5 - margin_px * h) and \
                   (abs(y_m) + w_eff_m * 0.5) <= (size_y_eff * 0.5 - margin_px * h):
                    xy.append((float(x_m), float(y_m)))
                    if max_points is not None and len(xy) >= max_points:
                        meta = dict(
                            W=W, H=H, size_x_eff=size_x_eff, size_y_eff=size_y_eff,
                            stone_w_px=w_px, stone_dist_px=dist_px, pitch_px=pitch_px,
                            stone_w_eff_m=w_eff_m, stone_dist_eff_m=dist_eff_m,
                            platform_pf_px=pf_px, platform_bbox_px=(px1c, px2c, py1c, py2c),
                        )
                        return xy, meta

            x0 += pitch_px

        y0 += pitch_px
        row += 1

    meta = dict(
        W=W, H=H, size_x_eff=size_x_eff, size_y_eff=size_y_eff,
        stone_w_px=w_px, stone_dist_px=dist_px, pitch_px=pitch_px,
        stone_w_eff_m=w_eff_m, stone_dist_eff_m=dist_eff_m,
        platform_pf_px=pf_px, platform_bbox_px=(px1c, px2c, py1c, py2c),
    )
    return xy, meta



def stepping_stones_xy_front_half_pixelwise2(
    size_x_m: float,
    size_y_m: float,
    horizontal_scale: float,
    platform_width_m: float,
    difficulty: float,
    stone_width_range_m: tuple[float, float] = (0.25, 0.25),      # サイズ固定なら (w,w)
    stone_distance_range_m: tuple[float, float] = (0.10, 0.35),
    margin_m: float = 0.2,
    outer_slack_m: float = 0.2,            # ★外側だけ余白（台側には入れない）
    platform_clearance_m: float = 0.0,
    per_row_phase: bool = True,
    seed: int = 0,
    max_points: Optional[int] = None,      # ★石数固定したいなら必ず指定
    platform_gap_px: int = 0,              # ★台と石の最小ギャップ（0=隙間なし狙い）
):
    h = float(horizontal_scale)
    rng = random.Random(seed)

    W = int(size_x_m / h)
    H = int(size_y_m / h)
    cx = W // 2
    cy = H // 2

    # difficulty で pitch
    w_px, dist_px, w_eff_m, dist_eff_m = params_from_difficulty(
        difficulty, h, stone_width_range_m, stone_distance_range_m
    )
    pitch_px = max(1, w_px + dist_px)

    # 最難pitch（石数固定用）
    w_px_max, dist_px_max, _, _ = params_from_difficulty(
        1.0, h, stone_width_range_m, stone_distance_range_m
    )
    pitch_px_max = max(1, w_px_max + dist_px_max)

    # platform bbox (px)
    pf_px = int(platform_width_m / h)
    px1 = (W - pf_px) // 2
    px2 = (W + pf_px) // 2
    py1 = (H - pf_px) // 2
    py2 = (H + pf_px) // 2

    clear_px = int(platform_clearance_m / h)
    px1c, px2c = px1 - clear_px, px2 + clear_px
    py1c, py2c = py1 - clear_px, py2 + clear_px

    margin_px = int(margin_m / h)
    outer_slack_px = int(outer_slack_m / h)

    # ---- ★台側アンカー（x方向）----
    # 「隙間なし」を狙って px2c ちょうどに置く。衝突するなら1pxずつ右へ逃がす。
    x0_start = max(px2c + platform_gap_px, cx, margin_px)

    # もし台と交差する定義（rect_intersect_1dの仕様）だと当たる場合があるので、最小で交差しない位置にする
    def intersects_platform_x(sx0, sx1):
        return rect_intersect_1d(sx0, sx1, px1c, px2c)

    # x方向だけ先にチェック（yは後でチェック）
    while intersects_platform_x(x0_start, x0_start + w_px):
        x0_start += 1  # どうしてもダメなら 1px だけ隙間ができるが、最小に抑える

    # ---- 外側端（+x側）には余白を残す ----
    x0_max = W - margin_px - w_px - outer_slack_px

    # y方向も上下端に余白
    y0_min = margin_px + outer_slack_px
    y0_max = H - margin_px - w_px - outer_slack_px

    # ---- ★石数固定：最難pitchで rows/cols を決める ----
    # phaseは「2個目以降」にしか掛けないが、最大phaseがあると右端に寄るので見込みで控える
    phase_max = (w_px - 1) if (per_row_phase and w_px > 1) else 0

    # 右端までに入る列数（最難pitch基準）
    # 1列目は x0_start 固定、2列目以降は x0_start + phase + (c-1)*pitch
    # 最後の列の左下: x0_start + phase_max + (n_cols-2)*pitch_px_max
    # その右端が x0_max+w_px を超えない必要
    if x0_start > x0_max:
        return [], {"reason": "no_space_x"}

    usable_w = x0_max - (x0_start + phase_max)
    # n_cols >=1
    n_cols_max = 1 if usable_w < 0 else (2 + (usable_w // pitch_px_max))  # 2列目以降の分を数える
    usable_h = y0_max - y0_min
    n_rows_max = 1 if usable_h < 0 else (1 + (usable_h // pitch_px_max))

    if n_cols_max <= 0 or n_rows_max <= 0:
        return [], {"reason": "no_space"}

    if max_points is not None:
        n_cols = min(n_cols_max, max_points)
        n_rows = (max_points + n_cols - 1) // n_cols
        if n_rows > n_rows_max:
            n_rows = n_rows_max
            n_cols = (max_points + n_rows - 1) // n_rows
        if n_cols > n_cols_max or n_rows > n_rows_max or (n_rows * n_cols) < max_points:
            return [], {
                "reason": "cannot_fit_max_points_at_max_pitch",
                "n_cols_max": int(n_cols_max),
                "n_rows_max": int(n_rows_max),
                "requested": int(max_points),
                "pitch_px_max": int(pitch_px_max),
            }
    else:
        n_cols, n_rows = int(n_cols_max), int(n_rows_max)

    # ---- 配置 ----
    xy = []
    for r in range(n_rows):
        y0 = y0_min + r * pitch_px
        if y0 > y0_max:
            continue

        phase = rng.randrange(0, w_px) if (per_row_phase and w_px > 1) else 0

        for c in range(n_cols):
            # ★1個目は台にアンカー。2個目以降のみphaseを適用
            if c == 0:
                x0 = x0_start
            else:
                x0 = x0_start + phase + (c - 1) * pitch_px

            if x0 > x0_max:
                continue

            sx0, sx1 = x0, x0 + w_px
            sy0, sy1 = y0, y0 + w_px

            hit_platform = (
                rect_intersect_1d(sx0, sx1, px1c, px2c)
                and rect_intersect_1d(sy0, sy1, py1c, py2c)
            )
            if hit_platform:
                continue

            xc = x0 + w_px * 0.5
            yc = y0 + w_px * 0.5
            x_m = (xc - cx) * h
            y_m = (yc - cy) * h
            xy.append((float(x_m), float(y_m)))

            if max_points is not None and len(xy) >= max_points:
                return xy, {
                    "pitch_px": int(pitch_px),
                    "pitch_px_max": int(pitch_px_max),
                    "n_rows": int(n_rows),
                    "n_cols": int(n_cols),
                    "x0_start_px": int(x0_start),
                    "outer_slack_px": int(outer_slack_px),
                    "platform_gap_px": int(platform_gap_px),
                }

    return xy, {
        "pitch_px": int(pitch_px),
        "pitch_px_max": int(pitch_px_max),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "produced": int(len(xy)),
        "x0_start_px": int(x0_start),
        "outer_slack_px": int(outer_slack_px),
        "platform_gap_px": int(platform_gap_px),
    }


























class ManagerBasedEnv:
    """The base environment encapsulates the simulation scene and the environment managers for the manager-based workflow.

    While a simulation scene or world comprises of different components such as the robots, objects,
    and sensors (cameras, lidars, etc.), the environment is a higher level abstraction
    that provides an interface for interacting with the simulation. The environment is comprised of
    the following components:

    * **Scene**: The scene manager that creates and manages the virtual world in which the robot operates.
      This includes defining the robot, static and dynamic objects, sensors, etc.
    * **Observation Manager**: The observation manager that generates observations from the current simulation
      state and the data gathered from the sensors. These observations may include privileged information
      that is not available to the robot in the real world. Additionally, user-defined terms can be added
      to process the observations and generate custom observations. For example, using a network to embed
      high-dimensional observations into a lower-dimensional space.
    * **Action Manager**: The action manager that processes the raw actions sent to the environment and
      converts them to low-level commands that are sent to the simulation. It can be configured to accept
      raw actions at different levels of abstraction. For example, in case of a robotic arm, the raw actions
      can be joint torques, joint positions, or end-effector poses. Similarly for a mobile base, it can be
      the joint torques, or the desired velocity of the floating base.
    * **Event Manager**: The event manager orchestrates operations triggered based on simulation events.
      This includes resetting the scene to a default state, applying random pushes to the robot at different intervals
      of time, or randomizing properties such as mass and friction coefficients. This is useful for training
      and evaluating the robot in a variety of scenarios.
    * **Recorder Manager**: The recorder manager that handles recording data produced during different steps
      in the simulation. This includes recording in the beginning and end of a reset and a step. The recorded data
      is distinguished per episode, per environment and can be exported through a dataset file handler to a file.

    The environment provides a unified interface for interacting with the simulation. However, it does not
    include task-specific quantities such as the reward function, or the termination conditions. These
    quantities are often specific to defining Markov Decision Processes (MDPs) while the base environment
    is agnostic to the MDP definition.

    The environment steps forward in time at a fixed time-step. The physics simulation is decimated at a
    lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
    independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of simulation steps per environment step)
    and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step) parameters. Based on these parameters, the
    environment time-step is computed as the product of the two. The two time-steps can be obtained by
    querying the :attr:`physics_dt` and the :attr:`step_dt` properties respectively.
    """

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # allocate dictionary to store metrics
        self.extras = {}

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # set the stage context for scene creation steps which use the stage
            with use_stage(self.sim.get_initial_stage()):


                self.scene = InteractiveScene(self.cfg.scene)
                attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                # since the reset can trigger callbacks which use the stage,
                # we need to set the stage context here
                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # initialize observation buffers
        self.obs_buf = {}



        #changed


        # self.block_paths_by_env = None
        # self.block_xy_local = None

    
        # tiles = self.scene.terrain.terrain_origins   # [rows, cols, 3] or [num_tiles, 3]
        # if tiles.dim() == 3:
        #     tiles = tiles.reshape(-1, 3)
        # N = int(self.num_envs)
        # self._fixed_tile_centers = tiles[:N].to(self.scene.env_origins.device,
        #                                         self.scene.env_origins.dtype).clone()

        
        # coll = self.scene.rigid_object_collections["stones"]
        # # もし default_object_state がワールドなら env_origins を引いてローカル化
        # S = coll.data.default_object_state.clone()      # [N, M, 13]
        # S[..., :3] -= self.scene.env_origins.unsqueeze(1)
        # self._stones_default_local = S 

        self._alloc_custom_buffers()


        # num_levels = int(self.scene.terrain.num_levels)  # 実際の属性名に合わせて
        # num_levels = int(self.scene.terrain.terrain_levels.max().item() + 1)
        # num_levels = int(env.scene.terrain.cfg.num_levels)  # 例

        num_levels =10
        self.num_patterns = 1
        # self.num_stones = 128
        self.stone_w_m = 0.25

        self._build_xy_bank(
            num_levels=num_levels,
            num_patterns=self.num_patterns,
            # num_stones=self.num_stones,
            stone_w_m=self.stone_w_m,
        )




        # self.buffers.alive  = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
        # self.buffers.hold_t = torch.zeros(self.scene.num_envs, device=self.device)





    def _alloc_custom_buffers(self):
        B, dev = self.scene.num_envs, self.device
        # 右前足タスクで使う状態フラグ
        self._buf = types.SimpleNamespace()
        self._buf.alive  = torch.zeros(B, dtype=torch.bool, device=dev)
        self._buf.hold_t = torch.zeros(B, device=dev)
        self._buf.is_holding = torch.zeros(B, dtype=torch.bool, device=dev)

        # ★ 追加: 力履歴スタック [B, K, 4脚, 3軸]
        self._ft_K = 6               # 履歴フレーム数（必要なら変更）
        self._mass_kg = 15.0         # 体重比正規化に使う質量（あなたの機体に合わせて）
        self._normalize_grf = True   # True なら N/(m*g) に正規化
        self._buf.ft_stack = torch.zeros(B, self._ft_K, 4, 3, device=dev)






    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    """
    Operations - Setup.
    """

    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        # prepare the managers
        # -- event manager (we print it here to make the logging consistent)
        print("[INFO] Event Manager: ", self.event_manager)
        # -- recorder manager
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)

        # perform events at the start of the simulation
        # in-case a child implementation creates other managers, the randomization should happen
        # when all the other managers are created
        if self.__class__ == ManagerBasedEnv and "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
        }

    """
    Operations - MDP.
    """

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)





        #changed, proposed

        #allign number of envs and sub terrains
        # sub_origins = self.scene.terrain.terrain_origins  # 形状が [rows, cols, 3] または [num_tiles, 3]

        # # 1) [rows, cols, 3] → [rows*cols, 3] にフラット化
        # if sub_origins.dim() == 3:
        #     sub_origins = sub_origins.reshape(-1, 3)

        # # 2) env数チェック
        # N = int(self.num_envs)
        # # assert sub_origins.shape[0] >= N, f"need >= {N} tiles, got {sub_origins.shape[0]}"

        # # 3) デバイス/型を env_origins に合わせてコピー（in-place）
        # sub_origins = sub_origins.to(device=self.scene.env_origins.device,
        #                             dtype=self.scene.env_origins.dtype)
        # self.scene.env_origins[:N].copy_(sub_origins[:N]) 


        


        # self._hard_place_everything(env_ids)





        # reset state of scene
        self._reset_idx(env_ids)


        # disable_stone_vs_stone(self.scene, stones_name="stones")

        # apply_stone_self_filter_per_env(self.scene)

        # # self.debug_read_some()


        # diag_stone_pair_filter(self.scene, stones_name="stones", env_i=0)













        #changed

        # # for proposed terrain
        self.stones = self.scene.rigid_object_collections["stones"]




        # STONE_W, STONE_H, GAP= 0.2, 0.3, 0.004

        # # STONE_W = 0.3       # 0.3 x 0.3 のブロック
        # # STONE_H = 0.3
        # # GAP     = 0.01      # ピッチ = 0.31m → そこそこ詰めて並ぶ

        # # CENTER_HALF = 0.5   # 中央の台の半分の大きさ
        # # MARGIN      = 0.05  # 台の縁からブロックまでの隙間 (~5cm)

        # # INNER_HALF  = CENTER_HALF + MARGIN  # = 0.55
        # # OUTER_HALF  = 1.5   
        # stone_xy_list = self.make_ring_xy4(STONE_W, GAP, inner_half=0.7, outer_half=3.37)
        # # stone_xy_list = self.make_ring_xy5(
        # #     stone_w=STONE_W,
        # #     inner_half=CENTER_HALF,
        # #     outer_half=OUTER_HALF,
        # #     gap=1e-4,     # ブロック同士のギャップ ≒ 0
        # #     margin=1e-3,  # 台の縁とのギャップ ≒ 0
        # # )




                
        # difficulty = 0.1   # カリキュラム等で決める [0,1]


        # stone_xy_list = generate_xy_list_front_isaac(
        #     terrain_size_xy=(8.0, 8.0),
        #     horizontal_scale=0.05,
        #     stone_size_xy=(0.3, 0.3),
        #     gap_xy=(0.1, 0.1),
        #     platform_size=1.0,
        #     x_front_ratio=0.5,     # 前半（x>0 側）だけ
        #     margin=0.10,
        #     clearance=0.0,
        #     per_row_phase=False,
        #     # seed=42,
        # )

        # stone_xy_list, meta = stepping_stones_xy_front_half_pixelwise(
        #     size_x_m=8.0,
        #     size_y_m=8.0,
        #     horizontal_scale=0.02,
        #     platform_width_m=1.0,
        #     difficulty=0.67,
        #     stone_width_range_m=(0.50, 0.20),
        #     stone_distance_range_m=(0.02, 0.05),
        #     margin_m=0.2,
        #     platform_clearance_m=0.03,   # まずは 0 推奨（避けすぎを防ぐ）
        #     per_row_phase=False,
        #     # seed=123,
        #     max_points=None,            # num_stonesに合わせるなら apply 側で切る/退避が安全
        # )

        # stone_xy_list, meta = stepping_stones_xy_front_half_pixelwise2(
        #     size_x_m=8.0,
        #     size_y_m=8.0,
        #     horizontal_scale=0.02,
        #     platform_width_m=1.0,
        #     difficulty=0,
        #     stone_width_range_m=(0.25, 0.25),
        #     stone_distance_range_m=(0.05, 0.1),
        #     margin_m=0.2,
        #     edge_slack_m = 0.2,
        #     platform_clearance_m=0.0,   # まずは 0 推奨（避けすぎを防ぐ）
        #     per_row_phase=False,
        #     # seed=123,
        #     max_points=None,            # num_stonesに合わせるなら apply 側で切る/退避が安全
        # )


        stone_xy_list, meta = stepping_stones_xy_front_half_pixelwise(
            size_x_m=8.0,
            size_y_m=8.0,
            horizontal_scale=0.02,
            platform_width_m=1.0,
            difficulty=1,
            stone_width_range_m=(0.25, 0.25),
            stone_distance_range_m=(0.02, 0.05),
            margin_m=0.2,
            outer_slack_m = 0.2,
            platform_clearance_m=0.0,   # まずは 0 推奨（避けすぎを防ぐ）
            per_row_phase=False,
            # seed=123,
            max_points=None,            # num_stonesに合わせるなら apply 側で切る/退避が安全
        )



        self.xy_local = torch.tensor(stone_xy_list, dtype=torch.float32, device=self.scene.device)
        z0 = 0.3
        self.stone_z_local = z0 - STONE_H * 0.5


        # # ③ 全Env一括で初期配置
        # self._place_all_stones()

    

        # robot = self.scene.articulations["robot"]
        # print("========================================")
        # print("!!! ROBOT JOINT ORDER CHECK !!!")
        # for i, name in enumerate(robot.joint_names):
        #     print(f"Index {i:02d}: {name}")
        # print("========================================")




        reset_stones_ring(
            self,
            env_ids,
            self.xy_local,
            self.stone_z_local,
            collection_name="stones",
        )

        # self.reset_stones_by_terrain_level(env_ids, self.num_patterns)

        

        

    

        # env の個数を数える（/World/envs/env_XXX を列挙）
        
        num_envs = self.scene.num_envs
        env_origins_np = self.scene.env_origins

        # self._reset_stones(env_ids)






        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        
       


        # return observations
        return self.obs_buf, self.extras

    
    def debug_read_some(self, stones_name="stones", limit=5):
        stage = omni.usd.get_context().get_stage()
        stones = self.scene.rigid_object_collections[stones_name]
        roots = _collect_stone_root_paths(self.scene, stones)
        n=0
        for r in roots:
            for shp in _iter_collision_shapes(stage, r):
                tgt = _get_authorable_shape_prim(shp)
                g = tgt.GetAttribute("physxCollision:group").Get()
                m = tgt.GetAttribute("physxCollision:mask").Get()
                print("[dbg]", tgt.GetPath(), "group=", g, "mask=", m)
                n += 1
                if n >= limit: return

    

    

    def _hard_place_everything(self, env_ids: torch.Tensor):
        scene = self.scene
        # 1) ロボット（Articulation）
        robot = scene["robot"]  # prim_path は "{ENV_REGEX_NS}/Robot" で複製されていること
        root = robot.data.default_root_state.clone()
        root[env_ids, :3] += scene.env_origins[env_ids]            # ← env 原点を足す
        robot.write_root_pose_to_sim(root[env_ids, :7])
        robot.write_root_velocity_to_sim(root[env_ids, 7:])

        # 2) 石（RigidObjectCollection）
        stones = scene["stones"]  # 例: RigidObjectCollection 名
        obj = stones.data.default_object_state.clone()              # shape: (N_env, N_obj, 13) 相当
        obj[env_ids, :, :3] += scene.env_origins[env_ids].unsqueeze(1)
        stones.write_object_link_pose_to_sim(obj[env_ids, :, :7])
        stones.write_object_com_velocity_to_sim(obj[env_ids, :, 7:])

        # 3) 反映
        # scene.write_data_to_sim()


    




    

    def make_ring_xy4(self, stone_w, gap, inner_half, outer_half):
        """
        グリッドを使いつつ、「はみ出し」を防ぐため、
        フィルタリング時にブロックの幅を考慮する。
        """
        pitch = stone_w + gap
        half_w = stone_w / 2.0  # ★ ブロックの半分の幅

        # 1. 原点(0,0)に対称なグリッドを生成 (No. 52のロジック)
        #    (これが「均等」の基礎となります)
        coords_pos = np.arange(0, outer_half, pitch)
        coords_neg = np.arange(-pitch, -outer_half, -pitch)
        xs = np.concatenate((coords_neg, coords_pos))
        ys = np.concatenate((coords_neg, coords_pos))

        if xs.size == 0 or ys.size == 0:
            return []  # 格子点が無ければ空リストを返す

        xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

        # 2. フィルタリング (★ここが修正点)
        
        # グリッドの各点(石の中心)のL∞ノルム(チェビシェフ距離)
        max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

        # 3. 内側の境界チェック
        #    石の「中心」が「(内側の境界 + ブロックの半幅)」より外側にあるか
        m_inner = (max_dist_center > inner_half + half_w)

        # 4. 外側の境界チェック
        #    石の「中心」が「(外側の境界 - ブロックの半幅)」より内側にあるか
        m_outer = (max_dist_center < outer_half - half_w)

        m_positive_x = (xs_grid - half_w > 0)

        # ブロックの上端が y_limit より下
        m_y_upper = (ys_grid + half_w < 1.55)
        # ブロックの下端が -y_limit より上
        m_y_lower = (ys_grid - half_w > -1.55)

        # 5. 両方を満たすもの
        #    (これにより、ブロック全体がリングの内側に収まる)
        m = m_inner & m_outer & m_positive_x & m_y_upper & m_y_lower
        
        xs_flat, ys_flat = xs_grid[m], ys_grid[m]
        
        return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]

    
    # def make_ring_xy5(self, stone_w, inner_half, outer_half, gap=1e-4, margin=1e-3, y_limit=1.55):
    #     """
    #     - stone_w: ブロックの一辺
    #     - inner_half: 中央台の半サイズ（ここでは 0.5）
    #     - outer_half: 外側の半径（どこまで石を敷き詰めるか）
    #     - gap: ブロック同士の間隔（ほぼ 0）
    #     - margin: 台の縁からブロックまでのすき間（ほぼ 0）
    #     """
    #     pitch = stone_w + gap
    #     half_w = stone_w / 2.0

    #     # 台の縁 (inner_half) から見て
    #     # 「ブロック内側の縁」がほぼ接する位置にブロック中心を置きたい：
    #     #   inner_edge ≒ inner_half + margin
    #     #   center = inner_edge + half_w
    #     # => r0: 最初のリングの中心半径
    #     r0 = inner_half + margin + half_w  # 0.5 + margin + 0.15 ≒ 0.65

    #     # 正の側の中心座標を r0 から pitch 間隔で生成
    #     # ブロックの外側までが outer_half を越えない範囲で。
    #     max_center = outer_half - half_w
    #     if r0 > max_center:
    #         return []

    #     coords_pos = np.arange(r0, max_center + 1e-6, pitch)
    #     coords_neg = -coords_pos[::-1]  # 原点対称に

    #     if coords_pos.size == 0:
    #         return []

    #     xs = np.concatenate((coords_neg, coords_pos))
    #     ys = np.concatenate((coords_neg, coords_pos))

    #     xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

    #     # L∞ノルム（チェビシェフ距離）でリング帯を取る
    #     max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

    #     # 中央台＋マージンより外側
    #     m_inner = (max_dist_center >= r0 - half_w)   # だいたい inner_half + margin

    #     # outer_half からはみ出さない
    #     m_outer = (max_dist_center <= outer_half - half_w)

    #     # 前方（+x）側だけ使うなら：
    #     m_positive_x = (xs_grid - half_w > 0.0)

    #     # y 範囲制限（必要なら調整）
    #     m_y_upper = (ys_grid + half_w < y_limit)
    #     m_y_lower = (ys_grid - half_w > -y_limit)

    #     m = m_inner & m_outer & m_positive_x & m_y_upper & m_y_lower

    #     xs_flat, ys_flat = xs_grid[m], ys_grid[m]
    #     return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]


    

    # def _reset_stones(self, env_ids: torch.Tensor):
    #     """
    #     各 env の原点 self.scene.env_origins を考慮して
    #     リング状にブロックを配置する。
    #     """
    #     device = self.device
    #     stones = self.scene.rigid_object_collections["stones"]

    #     # env 原点 [N_env, 3]
    #     env_origins = self.scene.env_origins[env_ids]  # [N_env, 3]

    #     n_env = env_ids.shape[0]
    #     n_block = self.xy_local.shape[0]

    #     # 現在の状態 [N_env, N_block, 13] をコピー
    #     state = stones.data.object_state_w[env_ids].clone()
    #     pos = state[..., 0:3]
    #     quat = state[..., 3:7]
    #     linv = state[..., 7:10]
    #     angv = state[..., 10:13]

    #     # ローカル XY [N_block, 2]
    #     xy_local = self.xy_local.to(device)  # 念のため device を合わせる
    #     x_local = xy_local[:, 0]   # [N_block]
    #     y_local = xy_local[:, 1]   # [N_block]

    #     # ブロードキャストで env ごとの world 座標を作る
    #     # env_origins: [N_env, 3]
    #     # → x_origin: [N_env, 1], y_origin: [N_env, 1], z_origin: [N_env, 1]
    #     x_origin = env_origins[:, 0:1]  # [N_env, 1]
    #     y_origin = env_origins[:, 1:2]  # [N_env, 1]
    #     z_origin = env_origins[:, 2:3]  # [N_env, 1]

    #     # pos: [N_env, N_block, 3]
    #     pos[:, :, 0] = x_origin + x_local[None, :]          # [N_env, N_block]
    #     pos[:, :, 1] = y_origin + y_local[None, :]          # [N_env, N_block]
    #     pos[:, :, 2] = z_origin + self.stone_z_local        # すべて同じ高さに置く

    #     # 姿勢は水平・速度ゼロにリセット
    #     quat[:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    #     linv.zero_()
    #     angv.zero_()

    #     # まとめてシミュレータに書き込み
    #     stones.write_object_state_to_sim(
    #         object_state=state,
    #         env_ids=env_ids,
    #     )

    

    def _reset_stones(self, env_ids: torch.Tensor):
        device = self.device
        stones = self.scene.rigid_object_collections["stones"]
        env_origins = self.scene.env_origins[env_ids]  # [N_env, 3]

        state = stones.data.object_state_w[env_ids].clone()
        pos  = state[..., 0:3]
        quat = state[..., 3:7]
        linv = state[..., 7:10]
        angv = state[..., 10:13]

        n_env   = env_ids.shape[0]
        n_block = pos.shape[1]  # ★ コレクション側の個数が正

        # ローカルXY（生成側が多い/少ない両方を許容）
        xy_local = self.xy_local.to(device=device, dtype=pos.dtype)  # [N_xy, 2]
        n_xy = xy_local.shape[0]

        N = min(n_block, n_xy)  # ★ 実際に配置する数

        x_origin = env_origins[:, 0:1]  # [N_env,1]
        y_origin = env_origins[:, 1:2]
        z_origin = env_origins[:, 2:3]

        # いったん全石を “退避位置” に置く（衝突を確実に避ける）
        # ※ x,y も遠くへ飛ばしておくと、広いAABB等の副作用も減ります
        park_dx = 1000.0
        park_dy = 1000.0
        park_dz = -100.0  # 地面より十分下

        pos[:, :, 0] = x_origin + park_dx
        pos[:, :, 1] = y_origin + park_dy
        pos[:, :, 2] = z_origin + park_dz

        # 先頭 N 個だけ “狙ったXY” に配置
        if N > 0:
            x_local = xy_local[:N, 0]  # [N]
            y_local = xy_local[:N, 1]  # [N]

            pos[:, :N, 0] = x_origin + x_local[None, :]
            pos[:, :N, 1] = y_origin + y_local[None, :]
            pos[:, :N, 2] = z_origin + self.stone_z_local  # いつもの高さ

        # 姿勢は水平・速度ゼロ
        quat[:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=quat.dtype)
        linv.zero_()
        angv.zero_()

        stones.write_object_state_to_sim(object_state=state, env_ids=env_ids)

    

    def _build_xy_bank(self,
                   num_levels: int,
                   num_patterns: int,
                #    num_stones: int,
                   stone_w_m: float,
                   stone_distance_range_m=(0.02, 0.05),
                   seed0: int = 1234):
        """
        self.xy_bank[level][pattern] = torch.Tensor [N,2]
        """
        self.xy_bank = [[None for _ in range(num_patterns)] for _ in range(num_levels)]

        for lvl in range(num_levels):
            # level -> difficulty (0..1)
            diff = 0.0 if num_levels == 1 else (lvl / (num_levels - 1))

            for k in range(num_patterns):
                xy_list, meta = stepping_stones_xy_front_half_pixelwise(
                    size_x_m=8.0,
                    size_y_m=8.0,
                    horizontal_scale=0.02,
                    platform_width_m=1.0,
                    difficulty=diff,

                    # ★サイズ固定
                    stone_width_range_m=(stone_w_m, stone_w_m),
                    stone_distance_range_m=stone_distance_range_m,

                    # ★石数固定（ここが重要）
                    # max_points=num_stones,

                    # パターン＝seed違い（同レベル内のバリエーション）
                    seed=seed0 + 10000 * lvl + k,

                    # ここはあなたの既存設定に合わせてOK
                    margin_m=0.2,
                    platform_clearance_m=0.0,
                    per_row_phase=False,
                )

                # # ★不足はサイレントに流すと地獄なので、ここで止めるのを推奨
                # if len(xy_list) < num_stones:
                #     raise RuntimeError(
                #         f"[xy_bank] level={lvl} pattern={k}: only {len(xy_list)}/{num_stones} stones. "
                #         f"Try reducing max gap / margin / clearance or num_stones."
                #     )

                self.xy_bank[lvl][k] = torch.tensor(
                    xy_list, device=self.device, dtype=torch.float32
                )


    

    def reset_stones_by_terrain_level(self, env_ids: torch.Tensor, num_patterns: int):
        terrain = self.scene.terrain

        # envごとのレベル（Terrain Generatorが管理しているやつ）
        levels = terrain.terrain_levels[env_ids].to(torch.long)  # [N_env]

        # envごとにパターンをランダム選択（これが「パターンあり」）
        pats = torch.randint(0, num_patterns, (env_ids.numel(),), device=self.device)  # [N_env]

        # (level, pattern) の組でグルーピングして、グループごとに self.xy_local を差し替えて _reset_stones
        # ※ self.xy_local は全env共通参照なので、同じxyを使うenv群ごとに呼ぶ必要がある
        for lvl in torch.unique(levels):
            mask_lvl = (levels == lvl)
            env_lvl = env_ids[mask_lvl]
            pats_lvl = pats[mask_lvl]

            for pat in torch.unique(pats_lvl):
                mask = (pats_lvl == pat)
                env_grp = env_lvl[mask]
                if env_grp.numel() == 0:
                    continue

                # ★ここでxy_localを選ぶ
                self.xy_local = self.xy_bank[int(lvl)][int(pat)]  # [Nstones,2]

                # あなたの既存の配置関数
                self._reset_stones(env_grp)





    


    def reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)







        
        #changed, proposed

        #allign number of envs and sub terrains
        # sub_origins = self.scene.terrain.terrain_origins  # 形状が [rows, cols, 3] または [num_tiles, 3]

        # # 1) [rows, cols, 3] → [rows*cols, 3] にフラット化
        # if sub_origins.dim() == 3:
        #     sub_origins = sub_origins.reshape(-1, 3)

        # # 2) env数チェック
        # N = int(self.num_envs)
        # # assert sub_origins.shape[0] >= N, f"need >= {N} tiles, got {sub_origins.shape[0]}"

        # # 3) デバイス/型を env_origins に合わせてコピー（in-place）
        # sub_origins = sub_origins.to(device=self.scene.env_origins.device,
        #                             dtype=self.scene.env_origins.dtype)
        # self.scene.env_origins[:N].copy_(sub_origins[:N]) 


        # self._hard_place_everything(env_ids)







        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)



        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras






    # def _update_hold_flags(self, T_hold=0.5, fz_contact=15.0, theta_lim=0.20, w_lim=1.0):
    #     inside = (fr_on_block_rect(self, margin=0.02) > 0.5)   # あなたの関数
    #     fz_ok  = (fr_fz(self) > fz_contact)
    #     theta  = _block_theta(self)
    #     wmag   = _block_wmag(self)
    #     stable = (theta <= theta_lim) & (wmag <= w_lim)

    #     alive = inside & fz_ok & stable
    #     self._buf.alive = alive
    #     self._buf.hold_t = torch.where(alive, self._buf.hold_t + self.dt,
    #                                    torch.zeros_like(self._buf.hold_t))

    # def _update_hold_flags(self, T_hold=0.2, fz_contact=5.0, theta_lim=0.20, w_lim=1.0):
    #     # 1. 各条件の判定
    #     inside = (fr_on_block_rect(self, margin=0.02) > 0.5)
    #     fz_ok  = (fr_fz(self) > fz_contact)
        
    #     # ブロックの安定性 (ここはそのままでOK)
    #     theta  = _block_theta(self)
    #     wmag   = _block_wmag(self)
    #     stable = (theta <= theta_lim) & (wmag <= w_lim)

    #     # 2. フラグの結合
    #     # 変数名を is_holding に変更 (alive と混同しないため)
    #     is_holding = inside & fz_ok & stable
        
    #     # バッファへの保存 (報酬計算などで使うため)
    #     self._buf.is_holding = is_holding 

    #     # 3. タイマーの更新 (チャタリング対策版)
    #     # holdingなら時間を足す。
    #     # holdingじゃないなら、「即0」ではなく「少し減らす」や「即0」などポリシーによるが、
    #     # ここでは厳密にやるなら「即0」で良いが、fzの閾値を少し緩める等の対策を推奨。
        
    #     # 例: 制御周期を使う
    #     dt = self.step_dt  
        
    #     self._buf.hold_t = torch.where(
    #         is_holding, 
    #         self._buf.hold_t + dt,          # 条件を満たせば加算
    #         torch.zeros_like(self._buf.hold_t) # 満たさなければリセット
    #     )

    def _update_ft_stack(self):
        # センサから今フレームの力を取得
        ft = self.scene.sensors["contact_forces"].data.net_forces_w  # [B,4,3]
        if self._normalize_grf:
            ft = ft / (self._mass_kg * 9.81)

        self._buf.ft_stack = torch.roll(self._buf.ft_stack, shifts=-1, dims=1)
        self._buf.ft_stack[:, -1] = ft







    def step(self, action: torch.Tensor) -> tuple[VecEnvObs, dict]:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is
        decimated at a lower time-step. This is to ensure that the simulation is stable. These two
        time-steps can be configured independently using the :attr:`ManagerBasedEnvCfg.decimation` (number of
        simulation steps per environment step) and the :attr:`ManagerBasedEnvCfg.sim.dt` (physics time-step).
        Based on these parameters, the environment time-step is computed as the product of the two.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations and extras.
        """



        
        self._update_ft_stack()
        # self._update_hold_flags()









        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # -- compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_step()

        # return observations and extras
        return self.obs_buf, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.viewport_camera_controller
            del self.action_manager
            del self.observation_manager
            del self.event_manager
            del self.recorder_manager
            del self.scene

            # clear callbacks and instance
            if float(".".join(get_version()[2])) >= 5:
                if self.cfg.sim.create_stage_in_memory:
                    # detach physx stage
                    omni.physx.get_physx_simulation_interface().detach_stage()
                    self.sim.stop()
                    self.sim.clear()

            self.sim.clear_all_callbacks()
            self.sim.clear_instance()

            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    """
    Helper functions.
    """

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)





            self._buf.alive[env_ids]  = False
            self._buf.hold_t[env_ids] = 0.0


        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)












    

    

