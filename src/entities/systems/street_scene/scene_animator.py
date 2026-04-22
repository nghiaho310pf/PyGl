import random
import math

from entities.registry import Registry
from entities.components.transform import Transform
from entities.components.street_scene.scene_animator_state import SceneAnimatorState
from entities.components.street_scene.scene_generator_state import SceneGeneratorState
from entities.components.street_scene.vehicle import Vehicle


from math_utils import vec3, quaternion_mul, quaternion_from_axis_angle

class SceneAnimatorSystem:
    @staticmethod
    def update(registry: Registry, dt: float):
        for scene_entity, (anim_state, gen_state) in registry.view(SceneAnimatorState, SceneGeneratorState):
            if not anim_state.animate_vehicles:
                continue

            street_length = gen_state.street_length
            half_length = street_length / 2.0
            lanes_count = gen_state.lanes_per_direction
            lane_width = (gen_state.street_width / 2) / lanes_count

            children = registry.get_children(scene_entity)
            vehicles_data = []

            # == collect all vehicle data ==
            for child in children:
                r_vehicle = registry.get_components(child, Transform, Vehicle)
                if r_vehicle:
                    transform, vehicle = r_vehicle
                    vehicles_data.append((child, transform, vehicle))

            # == compute acceleration and lane changes ==
            for i, (entity, transform, vehicle) in enumerate(vehicles_data):
                v_pos_z = transform.local.position[2]
                v_dir = 1.0 if vehicle.direction[2] > 0 else -1.0
                side = 1 if vehicle.lane_id > 0 else -1

                # --- lane following / acceleration ---
                # a vehicle is "in our way" if it's currently in our lane OR moving into it
                min_dist = float('inf')
                front_vehicle = None

                for j, (o_entity, o_transform, o_vehicle) in enumerate(vehicles_data):
                    if i == j:
                        continue

                    # check if other vehicle occupies our current lane or is moving into it
                    # also check our target lane if we are currently changing lanes
                    is_in_way = (o_vehicle.lane_id == vehicle.lane_id or o_vehicle.target_lane_id == vehicle.lane_id)
                    if vehicle.lane_id != vehicle.target_lane_id:
                        is_in_way = is_in_way or (o_vehicle.lane_id == vehicle.target_lane_id or o_vehicle.target_lane_id == vehicle.target_lane_id)

                    if not is_in_way:
                        continue

                    o_pos_z = o_transform.local.position[2]
                    dist = (o_pos_z - v_pos_z) * v_dir
                    if dist <= 0: dist += street_length

                    if dist < min_dist:
                        min_dist = dist
                        front_vehicle = o_vehicle

                target_accel = 0.0
                safe_distance = 7.0 + vehicle.current_speed * 0.7

                if front_vehicle:
                    if min_dist < safe_distance:
                        braking_factor = (safe_distance - min_dist) / safe_distance
                        target_accel = -vehicle.max_braking * braking_factor * 1.5
                        speed_diff = vehicle.current_speed - front_vehicle.current_speed
                        if speed_diff > 0:
                            target_accel -= speed_diff * 4.0
                    else:
                        if vehicle.current_speed < vehicle.target_speed:
                            target_accel = vehicle.max_acceleration
                else:
                    if vehicle.current_speed < vehicle.target_speed:
                        target_accel = vehicle.max_acceleration

                vehicle.acceleration = target_accel

                # --- lane changing decision ---
                if vehicle.lane_id == vehicle.target_lane_id:
                    if front_vehicle and min_dist < safe_distance * 2.0 and vehicle.current_speed < vehicle.target_speed * 0.95:
                        current_lane_idx = abs(vehicle.lane_id) - 1
                        possible_lanes = []
                        if current_lane_idx > 0: possible_lanes.append(side * current_lane_idx)
                        if current_lane_idx < lanes_count - 1: possible_lanes.append(side * (current_lane_idx + 2))

                        for target_l_id in possible_lanes:
                            is_clear = True
                            for j, (o_entity, o_transform, o_vehicle) in enumerate(vehicles_data):
                                if i == j: continue
                                if o_vehicle.lane_id == target_l_id or o_vehicle.target_lane_id == target_l_id:
                                    o_pos_z = o_transform.local.position[2]
                                    dist_f = (o_pos_z - v_pos_z) * v_dir
                                    if dist_f <= 0: dist_f += street_length
                                    dist_b = (v_pos_z - o_pos_z) * v_dir
                                    if dist_b <= 0: dist_b += street_length

                                    min_gap_f = 15.0 + vehicle.current_speed * 0.5
                                    min_gap_b = 10.0 + o_vehicle.current_speed * 0.5
                                    if dist_f < min_gap_f or dist_b < min_gap_b:
                                        is_clear = False
                                        break

                            if is_clear:
                                vehicle.target_lane_id = target_l_id
                                vehicle.lane_change_progress = 0.0
                                break

            # == update physics and lane change progress ==
            for entity, transform, vehicle in vehicles_data:
                vehicle.current_speed += vehicle.acceleration * dt
                vehicle.current_speed = max(0.0, vehicle.current_speed)

                # update Z (forward)
                v_dir = 1.0 if vehicle.direction[2] > 0 else -1.0
                transform.local.position[2] += v_dir * vehicle.current_speed * dt

                # wrap vehicles back onto other end of street
                if transform.local.position[2] > half_length:
                    transform.local.position[2] -= street_length
                elif transform.local.position[2] < -half_length:
                    transform.local.position[2] += street_length

                if vehicle.lane_id != vehicle.target_lane_id:
                    # execute lane change

                    vehicle.lane_change_progress += vehicle.lane_change_speed * dt

                    # interpolate X
                    side = 1 if vehicle.lane_id > 0 else -1
                    start_lane_idx = abs(vehicle.lane_id) - 1
                    target_lane_idx = abs(vehicle.target_lane_id) - 1

                    start_x = side * (start_lane_idx * lane_width + lane_width / 2.0)
                    target_x = side * (target_lane_idx * lane_width + lane_width / 2.0)

                    t = vehicle.lane_change_progress
                    # smooth step for position
                    smooth_t = t * t * (3 - 2 * t)
                    transform.local.position[0] = start_x + (target_x - start_x) * smooth_t

                    # derive rotation from movement.
                    # lateral velocity v_x = dx/dt
                    # dx/dt = (target_x - start_x) * d/dt(3t^2 - 2t^3)
                    # d/dt(3t^2 - 2t^3) = (6t - 6t^2) * lane_change_speed
                    v_x = (target_x - start_x) * 6.0 * t * (1.0 - t) * vehicle.lane_change_speed

                    # local lateral velocity (if facing -Z, world +X is local -X)
                    local_v_x = v_x * v_dir

                    # angle of velocity vector in local space
                    angle_rad = math.atan2(local_v_x, max(0.1, vehicle.current_speed))
                    angle_deg = math.degrees(angle_rad) * 0.5  # idk

                    tilt_rot = quaternion_from_axis_angle(vec3(0, 1, 0), angle_deg)
                    transform.local.rotation = quaternion_mul(vehicle.base_rotation, tilt_rot)

                    if vehicle.lane_change_progress >= 1.0:
                        vehicle.lane_id = vehicle.target_lane_id
                        vehicle.lane_change_progress = 0.0
                        transform.local.rotation = vehicle.base_rotation.copy()
                else:
                    # maintain lane

                    lane_idx = abs(vehicle.lane_id) - 1
                    side = 1 if vehicle.lane_id > 0 else -1
                    transform.local.position[0] = side * (lane_idx * lane_width + lane_width / 2.0)
                    transform.local.rotation = vehicle.base_rotation.copy()
