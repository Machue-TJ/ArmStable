"""Shared demo/RL episode initialization. Distances are metres, angles radians.

CSV camera positions/directions are in base_link coordinates at the reference
base pose. Marker coordinates are (right, down, forward) relative to that CSV
camera origin; the model supplies the roll missing from the CSV Z axis.
"""
from dataclasses import dataclass
from functools import lru_cache
import importlib

import mujoco
import numpy as np

from .settings import ROOT, load_settings_section, project_path


class PlacementError(ValueError):
    """Marker layout cannot satisfy this scene's visibility constraints."""


def load_callback(value):
    """Accept a Python callable or an importable 'package.module:function'."""
    if callable(value):
        return value
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("Callback must be callable or 'package.module:function'")
    module, name = value.split(":", 1)
    callback = getattr(importlib.import_module(module), name)
    if not callable(callback):
        raise TypeError(f"{value} is not callable")
    return callback


@lru_cache(maxsize=8)
def _read_pose_table(path, modified_ns):
    """Reuse a read-only CSV table across environments until the file changes."""
    poses = np.genfromtxt(path, delimiter=",", names=True, ndmin=1, encoding="utf-8-sig")
    poses.flags.writeable = False
    return poses


def load_episode_config(value=None):
    """Load and validate the episode section of the unified project settings."""
    config = load_settings_section("episode", value)
    markers = config["markers"]
    count = markers["count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("markers.count must be a positive integer")
    if markers["diameter_m"] != 0.015:
        raise ValueError("This task requires 15 mm markers")
    fov = np.asarray(markers["marker_fovy"], dtype=float)
    if fov.shape != (2,) or not np.isfinite(fov).all() or np.any((fov <= 0) | (fov >= 180)):
        raise ValueError("markers.marker_fovy must be [horizontal, vertical] degrees in (0, 180)")
    if not np.isfinite(config["base"]["initial_height_m"]):
        raise ValueError("base.initial_height_m must be finite")
    bounds = np.asarray(markers["depth_range_m"], dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0.2 <= bounds[0] <= bounds[1] <= 2.8):
        raise ValueError("markers.depth_range_m must lie within [0.2, 2.8]")
    depth = markers["plane_depth_m"]
    if depth is not None and not bounds[0] <= depth <= bounds[1]:
        raise ValueError("plane_depth_m must lie within depth_range_m")
    if not 0 <= markers["edge_margin"] < 1:
        raise ValueError("edge_margin must lie in [0, 1)")
    if not np.isfinite(markers["min_gap_m"]) or markers["min_gap_m"] < 0:
        raise ValueError("min_gap_m must be finite and nonnegative")
    for key in ("rgba", "plane_rgba"):
        rgba = np.asarray(markers[key], dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any((rgba < 0) | (rgba > 1)):
            raise ValueError(f"markers.{key} must contain four values in [0, 1]")
    for key in ("specular", "shininess"):
        if not 0 <= markers[key] <= 1:
            raise ValueError(f"markers.{key} must lie in [0, 1]")
    return config


def make_model(config):
    """Allocate real renderable sphere geoms without adding any joint DOFs."""
    spec = mujoco.MjSpec.from_file(str(ROOT / "xml/agilex/scene.xml"))
    # Keep the original sky gradient, checker floor, lights and colored props.
    # The old extent * zfar was only 4 m. Increase the far distance so the
    # skybox remains behind the complete 2.8 m observation volume.
    spec.visual.map.zfar = 100
    spec.visual.headlight.specular = [0.6, 0.6, 0.6]
    if config["free_space"]:
        spec.geom("floor").delete()
    markers = config["markers"]
    spec.add_material(name="marker_material", rgba=markers["rgba"], emission=0,
                      specular=markers["specular"], shininess=markers["shininess"])
    for i in range(config["markers"]["count"]):
        body = spec.worldbody.add_body(name=f"marker_body_{i}", mocap=True, pos=[0, 0, -100])
        body.add_geom(
            name=f"marker_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[config["markers"]["diameter_m"] / 2, 0, 0],
            rgba=markers["rgba"],
            contype=0, conaffinity=0, group=5, material="marker_material",
        )
    plane = spec.worldbody.add_body(name="marker_plane", mocap=True, pos=[0, 0, -100])
    rgba = list(markers["plane_rgba"])
    if not markers["plane_visible"]:
        rgba[3] = 0
    plane.add_geom(name="marker_plane_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[1, 1, 0.0002], rgba=rgba, group=4, contype=0, conaffinity=0)
    return spec.compile()


def configure_base(base, settings, motion=None):
    """Configure held pose, file/pose callback, or world linear/angular velocity."""
    from config.flobase.piper_base import BasePose
    if motion is not None:
        if callable(motion):
            base.set_motion(motion)
        else:
            base.load(project_path(motion))
        return
    pose = (BasePose(settings["position_m"], settings["quat_wxyz"])
            if settings.get("quat_wxyz") is not None else
            BasePose.from_rpy(settings["position_m"], settings["rpy_rad"]))
    mode = settings["mode"]
    if mode == "fixed":
        base.set_pose(pose.position_m, quat_wxyz=pose.quat_wxyz)
    elif mode == "trajectory":
        base.load(project_path(settings["trajectory"]))
    elif mode == "pose_callback":
        base.set_motion(load_callback(settings["callback"]))
    elif mode == "velocity_callback":
        base.set_velocity_motion(load_callback(settings["callback"]), initial_pose=pose)
    else:
        raise ValueError(f"Unknown base mode: {mode}")


def configure_viewer(viewer, data, info):
    """Frame the raised arm and entire marker plane in the external scene view."""
    points = [data.body(f"link{i}").xpos.copy() for i in range(1, 7)]
    points.append(data.body("base_link").xpos.copy())
    transform = np.asarray(info["world_from_csv_camera"])
    center = np.asarray(info["marker_plane_position_world_m"])
    extent = np.asarray(info["marker_plane_half_extent_m"])
    points.extend(center + extent[0] * x * transform[:3, 0] + extent[1] * y * transform[:3, 1]
                  for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    viewer.cam.lookat[:] = (low + high) / 2
    viewer.cam.distance = max(1.4, 1.6 * np.linalg.norm(high - low))
    viewer.cam.azimuth, viewer.cam.elevation = 120, -25
    viewer.opt.geomgroup[3] = 0
    viewer.opt.geomgroup[4:6] = 1


@dataclass
class MarkerContext:
    """Passed to generator(context, rng) -> (count, 3) CSV-camera coordinates.

    All centers must share one Z plane. Use accepts() while sampling custom
    patterns to check actual RGB/stereo FOVs, separation and scene occlusion.
    """
    count: int
    plane_depth_m: float
    radius_m: float
    half_extent_m: np.ndarray
    world_from_camera: np.ndarray
    accepts: object

    def to_world(self, points_m):
        points = np.asarray(points_m, dtype=float)
        return points @ self.world_from_camera[:3, :3].T + self.world_from_camera[:3, 3]


def generate_markers(context, rng):
    """Random separated centers on a single plane perpendicular to camera +Z."""
    # Restart crowded layouts instead of keeping an early point that blocks
    # the remaining slots in the narrow near-distance stereo overlap.
    for _ in range(20):
        points = []
        for _ in range(max(500, context.count * 100)):
            candidate = np.r_[rng.uniform(-context.half_extent_m, context.half_extent_m, 2),
                              context.plane_depth_m]
            if context.accepts(candidate, points):
                points.append(candidate)
                if len(points) == context.count:
                    return np.asarray(points)
    raise PlacementError("Cannot fit visible markers on this plane; reduce count/gap or change depth/pose")


class EpisodeInitializer:
    def __init__(self, model, data, base, config=None, marker_generator=None):
        self.model, self.data, self.base = model, data, base
        self.config = load_episode_config(config)
        self.settings = self.config["markers"]
        half_angles = np.deg2rad(self.settings["marker_fovy"]) / 2
        self._tan = np.tan(half_angles)
        self._radius = self.settings["diameter_m"] / 2
        self._sphere_margin = self._radius / np.cos(half_angles)
        self._ray_groups = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        self._ray_hit = np.empty(1, dtype=np.int32)
        self.generator = marker_generator or self.settings.get("generator")
        if self.generator is not None:
            self.generator = load_callback(self.generator)
        path = project_path(self.config["init_poses_csv"]).resolve()
        self.poses = _read_pose_table(path, path.stat().st_mtime_ns)
        fields = ([f"q{i}_rad" for i in range(1, 7)] +
                  [f"camera_{axis}_m" for axis in "xyz"] +
                  [f"camera_zaxis_{axis}" for axis in "xyz"])
        if not {"sample_id", *fields} <= set(self.poses.dtype.names or ()) or not len(self.poses):
            raise ValueError("Initial pose CSV is empty or missing sample/joint/camera columns")
        if not np.isfinite(np.column_stack([self.poses[f] for f in fields + ["sample_id"]])).all():
            raise ValueError("Initial pose CSV must contain finite numbers")
        if len(np.unique(self.poses["sample_id"])) != len(self.poses):
            raise ValueError("CSV sample_id must be unique")
        self._sample_indices = {sample_id: i for i, sample_id in enumerate(self.poses["sample_id"])}
        self.qids = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.cids = np.array([model.actuator(f"joint{i}").id for i in range(1, 7)])
        self.gids = np.array([model.geom(f"marker_{i}").id for i in range(self.settings["count"])])
        self.mids = np.array([model.body(f"marker_body_{i}").mocapid[0] for i in range(len(self.gids))])
        self.plane_mid = int(model.body("marker_plane").mocapid[0])
        self.plane_gid = model.geom("marker_plane_geom").id
        self.floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        angles = np.column_stack([self.poses[f"q{i}_rad"] for i in range(1, 7)])
        ranges = np.array([model.joint(f"joint{i}").range for i in range(1, 7)])
        if np.any(angles < ranges[:, 0]) or np.any(angles > ranges[:, 1]):
            raise ValueError("CSV joint angles exceed the model joint limits")
        self._sensor_planes = []
        for name in ("d435i_rgb", "d435i_depth", "d435i_ir_right"):
            cid = model.camera(name).id
            w, h = model.cam_sensorsize[cid]
            fx, fy, ox, oy = model.cam_intrinsic[cid]
            if w <= 0 or h <= 0 or fx <= 0 or fy <= 0:
                raise ValueError("Configure D435iCamera before EpisodeInitializer")
            cx, cy = (w - 1) / 2 - ox, (h - 1) / 2 + oy
            slopes = np.array([(cx + .5) / fx, (w - .5 - cx) / fx,
                               (cy + .5) / fy, (h - .5 - cy) / fy])
            normals = np.array([[1, 0, slopes[0]], [-1, 0, slopes[1]],
                                [0, 1, slopes[2]], [0, -1, slopes[3]]])
            self._sensor_planes.append(normals / np.linalg.norm(normals, axis=1)[:, None])
        self.marker_positions_world_m = np.empty((len(self.gids), 3))

    def reset(self, rng, *, sample_id=None, marker_positions_m=None):
        """Reset physics, sample arm + matching camera row, restart base, place markers."""
        if sample_id is None:
            row = self.poses[int(rng.integers(len(self.poses)))]
        else:
            index = self._sample_indices.get(sample_id)
            if index is None:
                raise ValueError(f"Unknown CSV sample_id: {sample_id}")
            row = self.poses[index]
        return self._reset_row(row, rng, marker_positions_m)

    def _reset_row(self, row, rng, marker_positions_m):
        """Apply one CSV row at the fixed base height and place a visible layout."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        angles = np.array([row[f"q{i}_rad"] for i in range(1, 7)])
        self.data.qpos[self.qids] = angles
        self.data.ctrl[self.cids] = angles
        # Normalize only the commanded initial height. No arm bounds or trial
        # translations are needed, and trajectory/callback relative motion stays intact.
        initial = self.base.motion(0.0) if self.base.motion is not None else self.base.hold_pose
        self.base.set_episode_offset([0, 0, self.config["base"]["initial_height_m"] - initial.position_m[2]])
        base = self.data.body("base_link")
        base_rotation = base.xmat.reshape(3, 3)
        origin = base.xpos + base_rotation @ np.array([row[f"camera_{a}_m"] for a in "xyz"])
        forward = base_rotation @ np.array([row[f"camera_zaxis_{a}"] for a in "xyz"])
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise ValueError("CSV camera Z axis must be nonzero")
        forward /= norm
        rgb_id = self.model.camera("d435i_rgb").id
        actual = self.data.cam_xmat[rgb_id].reshape(3, 3)
        if not np.allclose(forward, -actual[:, 2], atol=1e-5):
            raise ValueError("CSV camera Z axis disagrees with the model at its joint angles")
        right = actual[:, 0] - forward * np.dot(actual[:, 0], forward)
        right /= np.linalg.norm(right)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
        transform[:3, 3] = origin
        self.world_from_camera = transform
        self._cameras = [(self.data.cam_xpos[self.model.camera(name).id].copy(),
                          self.data.cam_xmat[self.model.camera(name).id].reshape(3, 3) @ np.diag([1, -1, -1]))
                         for name in ("d435i_rgb", "d435i_depth", "d435i_ir_right")]
        positions = marker_positions_m if marker_positions_m is not None else self.settings["positions_m"]
        depth = self.settings["plane_depth_m"]
        if positions is not None:
            points = np.asarray(positions, dtype=float)
        else:
            depth = float(rng.uniform(*self.settings["depth_range_m"])) if depth is None else depth
            half_extent = depth * self._tan - self._sphere_margin
            if np.any(half_extent <= 0):
                raise PlacementError("marker_fovy is too narrow for a 15 mm sphere at this depth")
            context = MarkerContext(len(self.gids), depth, 0.0075,
                                    half_extent * (1 - self.settings["edge_margin"]),
                                    transform.copy(), self._accepts)
            points = np.asarray((self.generator or generate_markers)(context, rng), dtype=float)
        # The built-in sampler already checks every candidate. User layouts
        # must still be validated, including custom generators that skip accepts().
        if positions is not None or self.generator is not None:
            self._validate_points(points)
        world = points @ transform[:3, :3].T + origin
        extent = points[0, 2] * self._tan
        plane_position = origin + points[0, 2] * forward
        self.data.mocap_pos[self.mids] = world
        self.marker_positions_world_m = world.copy()
        self.data.mocap_pos[self.plane_mid] = plane_position
        mujoco.mju_mat2Quat(self.data.mocap_quat[self.plane_mid], transform[:3, :3].ravel())
        self.model.geom_size[self.plane_gid] = [*extent, 0.0002]
        self.model.geom_rbound[self.plane_gid] = np.linalg.norm(self.model.geom_size[self.plane_gid])
        mujoco.mj_forward(self.model, self.data)
        return {"sample_id": int(row["sample_id"]), "joint_angles_rad": angles.tolist(),
                "camera_position_world_m": origin.tolist(), "camera_zaxis_world": forward.tolist(),
                "world_from_csv_camera": transform.tolist(),
                "marker_positions_camera_m": points.tolist(),
                "marker_positions_world_m": world.tolist(), "marker_diameter_m": 0.015,
                "marker_plane_depth_m": float(points[0, 2]), "marker_fovy": list(self.settings["marker_fovy"]),
                "marker_plane_position_world_m": plane_position.tolist(),
                "marker_plane_normal_world": forward.tolist(), "marker_plane_half_extent_m": extent.tolist(),
                "base_height_offset_m": float(self.base.episode_offset_m[2])}

    def _accepts(self, candidate, previous):
        p = np.asarray(candidate, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            return False
        radius = self._radius
        low, high = self.settings["depth_range_m"]
        if not low <= p[2] <= high or np.any(np.abs(p[:2]) + self._sphere_margin > p[2] * self._tan):
            return False
        rotation, origin = self.world_from_camera[:3, :3], self.world_from_camera[:3, 3]
        world = rotation @ p + origin
        if self.floor_gid >= 0 and world[2] - radius <= self.data.geom_xpos[self.floor_gid, 2]:
            return False
        previous_world = np.asarray(previous).reshape(-1, 3) @ rotation.T + origin
        for index, (camera_origin, camera_rotation) in enumerate(self._cameras):
            local = camera_rotation.T @ (world - camera_origin)
            if local[2] <= radius:
                return False
            # The RGB marker window is independent of the physical stream FOV.
            if index == 0 and np.any(np.abs(local[:2]) + self._sphere_margin > local[2] * self._tan):
                return False
            if not self._inside_sensor(local, index):
                return False
            # Angular separation guarantees distinct projected spheres, also
            # after accounting for the RGB/depth optical-center offsets.
            vector = world - camera_origin
            distance = np.linalg.norm(vector)
            for other in previous_world:
                delta = other - camera_origin
                other_distance = np.linalg.norm(delta)
                separation = np.arccos(np.clip(np.dot(vector, delta) / (distance * other_distance), -1, 1))
                required = (np.arcsin(radius / distance) + np.arcsin(radius / other_distance)
                            + self.settings["min_gap_m"] / min(distance, other_distance))
                if separation < required:
                    return False
            for offset in (np.zeros(3), camera_rotation[:, 0] * radius, -camera_rotation[:, 0] * radius,
                           camera_rotation[:, 1] * radius, -camera_rotation[:, 1] * radius):
                ray = world + offset - camera_origin
                length = np.linalg.norm(ray)
                obstruction = mujoco.mj_ray(self.model, self.data, camera_origin, ray / length,
                                            self._ray_groups, 1, -1, self._ray_hit)
                if 0 <= obstruction < length + radius:
                    return False
        return True

    def _inside_sensor(self, point, camera_index):
        """Entire sphere inside the actual calibrated pinhole sensor frustum."""
        return np.all(self._sensor_planes[camera_index] @ point >= self._radius)

    def _validate_points(self, points):
        if points.shape != (len(self.gids), 3) or not np.isfinite(points).all():
            raise ValueError(f"Marker generator/positions must return finite shape ({len(self.gids)}, 3)")
        if not np.allclose(points[:, 2], points[0, 2], rtol=0, atol=1e-8):
            raise ValueError("All marker centers must share a plane perpendicular to camera Z")
        for i, point in enumerate(points):
            if not self._accepts(point, points[:i]):
                raise ValueError(f"Marker {i} violates depth/FOV/separation/visibility constraints")
