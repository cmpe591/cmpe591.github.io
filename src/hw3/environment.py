import collections
from copy import deepcopy

import numpy as np
from dm_control import mjcf
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


class BaseEnv:
    def __init__(self, render_mode=None) -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [-np.pi/2, -np.pi/2, np.pi/2, -2.07, 0, 0, 200]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def reset(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "data"):
            del self.data
        if self.viewer is not None:
            if self._render_mode == "offscreen":
                del self.viewer
            else:
                self.viewer.close()

        scene = self._create_scene()
        xml_string = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        self.data = mujoco.MjData(self.model)
        if self._render_mode == "gui":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            self.viewer.cam.fixedcamid = 0
            self.viewer.cam.type = 2
            self.viewer._render_every_frame = False
            self.viewer._run_speed = 2
        elif self._render_mode == "offscreen":
            self.viewer = mujoco.Renderer(self.model, 128, 128)

        self.data.ctrl[:] = self._init_position
        mujoco.mj_step(self.model, self.data, nstep=2000)
        self.data.ctrl[4] = -np.pi/2
        mujoco.mj_step(self.model, self.data, nstep=2000)
        self._t = 0

    def _create_scene(self):
        return create_tabletop_scene()

    def _step(self):
        mujoco.mj_step(self.model, self.data)
        if self._render_mode == "gui":
            self.viewer.render()

    def _get_joint_position(self):
        position = np.zeros(7)
        for idx in range(len(self._joint_names)):
            position[idx] = self.data.qpos[self._joint_qpos_idxs[idx]]
            if idx == 6:
                position[idx] /= 0.721
        return position

    def _set_joint_position(self, position_dict, max_iters=10000, threshold=0.05):
        for idx in position_dict:
            if idx == 6:
                self.data.ctrl[idx] = position_dict[idx]*255
            else:
                self.data.ctrl[idx] = position_dict[idx]

        max_error = 100*threshold
        it = 0
        while max_error > threshold:
            it += 1
            self._step()
            max_error = 0
            current_position = self._get_joint_position()
            for idx in position_dict:
                error = abs(current_position[idx] - position_dict[idx])
                if error > max_error:
                    max_error = error
            if it > max_iters:
                print("Max iters reached")
                break

    def _get_ee_pose(self):
        ee_position = self.data.site(self._ee_site).xpos
        ee_rotation = self.data.site(self._ee_site).xmat
        ee_orientation = np.zeros(4)
        mujoco.mju_mat2Quat(ee_orientation, ee_rotation)
        return ee_position, ee_orientation

    def _set_ee_pose(self, position, rotation=None, orientation=None, max_iters=10000, threshold=0.04):
        if rotation is not None and orientation is not None:
            raise Exception("Only one of rotation or orientation can be set")
        quat = None
        if rotation is not None:
            quat = R.from_euler("xyz", rotation, degrees=True).as_quat()
        elif orientation is not None:
            quat = orientation
        qpos = qpos_from_site_pose(self.model, self.data, self._ee_site,
                                   position, quat, joint_names=self._joint_names[:-1]).qpos
        qdict = {i: qpos[q_idx][0] for i, q_idx in enumerate(self._joint_qpos_idxs[:-1])}

        max_error = 100*threshold
        it = 0
        while max_error > threshold:
            it += 1
            self._step()
            max_error = 0
            curr_pos, curr_quat = self._get_ee_pose()
            max_error += np.linalg.norm(np.array(position) - curr_pos)

            # this part is taken from dm_control
            # https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py#L165
            if quat is not None:
                neg_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_quat, curr_quat)
                error_quat = np.zeros(4)
                mujoco.mju_mulQuat(error_quat, quat, neg_quat)
                error_vel = np.zeros(3)
                mujoco.mju_quat2Vel(error_vel, error_quat, 1)
                max_error += np.linalg.norm(error_vel)
            for idx in qdict:
                self.data.ctrl[idx] = qpos[self._joint_qpos_idxs[idx]]
            if it > max_iters:
                print("Max iters reached")
                break

        if max_error > threshold:
            return False
        return True

    def _set_ee_in_cartesian(self, position, rotation=None, max_iters=10000, threshold=0.04, n_splits=20):
        ee_position, ee_orientation = self._get_ee_pose()
        position_traj = np.linspace(ee_position, position, n_splits+1)[1:]
        if rotation is not None:
            target_orientation = R.from_euler("xyz", rotation, degrees=True).as_quat()
            r = R.from_quat([ee_orientation, target_orientation])
            slerp = Slerp([0, 1], r)
            orientation_traj = slerp(np.linspace(0, 1, n_splits+1)[1:]).as_quat()
        else:
            orientation_traj = [ee_orientation]*n_splits

        result = self._follow_ee_trajectory(position_traj, orientation_traj,
                                   max_iters=max_iters, threshold=threshold)
        return result

    def _follow_ee_trajectory(self, position_traj, orientation_traj, max_iters=10000, threshold=0.04):
        for position, orientation in zip(position_traj, orientation_traj):
            result = self._set_ee_pose(position, orientation=orientation,
                              max_iters=max_iters, threshold=threshold)
            if not result:
                return False
        return True


def create_tabletop_scene():
    scene = create_empty_scene()
    add_camera_to_scene(scene, "frontface", [2.5, 0., 2.0], [-1.5, 0, 0])
    add_camera_to_scene(scene, "topdown", [0.73, 0., 2.3], [0.68, 0, 0])
    create_base(scene, [0, 0, 0.5], 0.5)
    create_object(scene, "box", [0.7, 0, 1], [0, 0, 0, 1], [0.5, 0.5, 0.02], [0.7, 0.7, 0.7, 1.0], friction=[0.2, 0.005, 0.0001], name="table", static=True)
    create_object(scene, "box", [0.7, 0, 0.5], [0, 0, 0, 1], [0.05, 0.05, 0.5], [0.9, 0.9, 0.9, 1.0], name="table_leg", static=True)

    wall_height = 0.03
    create_object(scene, "box", [0.7, 0.5, 1.0+wall_height], [0.7071068, 0, 0, 0.7071068], [0.01, 0.5, wall_height], [0.3, 0.3, 1.0, 1.0], name="right_wall", static=True)
    create_object(scene, "box", [0.7, -0.5, 1.0+wall_height], [0.7071068, 0, 0, 0.7071068], [0.01, 0.5, wall_height], [0.3, 0.3, 1.0, 1.0], name="left_wall", static=True)
    create_object(scene, "box", [0.2, 0.0, 1.0+wall_height], [0.7071068, 0, 0, 0.7071068], [0.5, 0.01, wall_height], [0.3, 0.3, 1.0, 1.0], name="top_wall", static=True)
    create_object(scene, "box", [1.2, 0.0, 1.0+wall_height], [0.7071068, 0, 0, 0.7071068], [0.5, 0.01, wall_height], [0.3, 0.3, 1.0, 1.0], name="bottom_wall", static=True)
    scene.find("site", "attachment_site").attach(create_ur5e_robotiq85f())
    return scene


def create_empty_scene():
    root = mjcf.RootElement()
    root.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    root.visual.headlight.ambient = [0.1, 0.1, 0.1]
    root.visual.headlight.specular = [0.0, 0.0, 0.0]
    root.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    getattr(root.visual, "global").azimuth = 120
    getattr(root.visual, "global").elevation = -20

    root.asset.add("texture", type="skybox", builtin="gradient", rgb1=[0.3, 0.5, 0.7],
                   rgb2=[0, 0, 0], width="512", height="3072")
    groundplane = root.asset.add("texture", type="2d", name="groundplane", builtin="checker", mark="edge",
                                 rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8],
                                 width="300", height="300")
    floor_mat = root.asset.add("material", name="groundplane", texture=groundplane, texrepeat=[5, 5], texuniform=True,
                               reflectance=0.2)
    root.worldbody.add("light", pos=[0., 0., 1.5], dir=[0, 0, -1], directional=True)
    root.worldbody.add("geom", type="plane", material=floor_mat, size=[0, 0, 0.05])
    return root


def create_ur5e_robotiq85f():
    robot = mjcf.from_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
    gripper = mjcf.from_path("mujoco_menagerie/robotiq_2f85/2f85.xml")
    gripper.worldbody.add("site", name="gripper_site", pos=[0, 0, 0.15], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0])
    robot.find("site", "attachment_site").attach(gripper)
    return robot


def create_object(root, obj_type, pos, quat, size, rgba, friction=[0.5, 0.005, 0.0001], density=1000,
                  name=None, static=False):
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    if not static:
        body.add("joint", type="free")
    body.add("geom", type=obj_type, size=size, rgba=rgba, friction=friction, name=name, density=density)
    return root


def create_visual(root, obj_type, pos, quat, size, rgba, name=None):
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    body.add("site", type=obj_type, size=size, rgba=rgba, name=name)
    return root


def create_base(root, position, height, rgba=[0.5, 0.5, 0.5, 1.0]):
    body = root.worldbody.add("body", pos=position, name="groundbase")
    body.add("geom", type="cylinder", size=[0.1, height], rgba=rgba, name="groundbase")
    body.add("site", pos=[0, 0, height], name="attachment_site")
    return root


def add_camera_to_scene(root, name, position, target):
    target_dummy = root.worldbody.add("body", pos=target)
    root.worldbody.add("camera", name=name, mode="targetbody", pos=position, target=target_dummy)
    return root


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def qpos_from_site_pose(model,
                        data,
                        site_name,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-14,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=100,
                        inplace=False):
    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("At least one of `target_pos` or `target_quat` must be specified.")

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        data = deepcopy(data)

    mujoco.mj_fwdPosition(model, data)
    site_id = model.site(site_name).id

    # todo: check they are indeed updated in place

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        dof_indices = [model.joint(name).id for name in joint_names]
    else:
        raise ValueError(f"`joint_names` must be either None, a list, a tuple, or a numpy array; "
                         f"got {type(joint_names)}.")

    success = False
    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            # translational error.
            err_pos[:] = target_pos - data.site(site_name).xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # rotational error.
            mujoco.mju_mat2Quat(site_xquat, data.site(site_name).xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            success = True
            break
        else:
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]
            reg_strength = regularization_strength if err_norm > regularization_threshold else 0.0
            update_joints = nullspace_method(jac_joints, err, regularization_strength=reg_strength)
            update_norm = np.linalg.norm(update_joints)

        progress_criterion = err_norm / update_norm
        if progress_criterion > progress_thresh:
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        update_nv[dof_indices] = update_joints
        mujoco.mj_integratePos(model, data.qpos, update_nv, 1)
        mujoco.mj_fwdPosition(model, data)

    if not inplace:
        qpos = data.qpos.copy()
    else:
        qpos = data.qpos

    return IKResult(qpos, err_norm, steps, success)


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
