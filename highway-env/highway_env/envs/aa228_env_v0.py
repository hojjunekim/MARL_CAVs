"""
Roundabout Multi-Agent Environment for AA228/MARL projects.
Base road geometry is a simple single-lane roundabout.
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple, Dict, Any

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import CircularLane, LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
# You may need to import your custom vehicle types if they are not in the highway_env standard library

class RoundaboutEnv(AbstractEnv):
    """
    A single-lane roundabout environment.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                # "vehicles_count": 10,
                # "features": ["present", "x", "y", "vx", "vy", "cos_h", "sin_h"]
                },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600, # Increased size for roundabouts
            "centering_position": [2.05, 0.5],
            "simulation_frequency": 15,
            "duration": 40, # Longer duration for a continuous flow
            "policy_frequency": 5,
            "reward_speed_range": [0, 15],
            "COLLISION_REWARD": 200,
            "HIGH_SPEED_REWARD": 0.0,
            "INCENTIVE_REWARD": 0.5, # New reward for staying on the road/progress
            "scaling": 3,
        })
        return config

    def _reward(self, action: int) -> float:
        # Note: The base class uses a reward calculated for the first controlled vehicle.
        # This will be overridden or scaled in the MARL version.
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        Agent reward for the roundabout: high speed, collision avoidance, and progress.
        """
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        # 1. Collision Cost
        collision_cost = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed)
        
        # 2. High Speed Reward
        high_speed_reward = self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)

        # 3. Incentive Reward (for moving forward / staying in inner lane)
        # Simplified: reward proportional to speed if not crashed.
        incentive_reward = self.config["INCENTIVE_REWARD"] * scaled_speed
        
        # Compute overall reward
        reward = collision_cost + high_speed_reward + incentive_reward
        return reward

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # The base AbstractEnv handles the call to _agent_reward and _reward,
        # but since we inherit from AbstractEnv, the standard step should handle single-agent.
        # We assume this base class is primarily used for defining the road and rewards.
        
        # The MARL wrapper will handle the multi-agent obs/reward structure.
        obs, reward, done, info = super().step(action)
        # done = terminated or truncated

        # Add multi-agent compliant info that the MARL environment expects
        # Although this is the base, adding expected info keys prevents errors in multi-agent logic
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info["regional_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["average_speed"] = np.mean([v.speed for v in self.road.vehicles])

        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or duration is reached."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or duration is reached."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=0) -> None:
        self._make_road()
        self._make_vehicles(num_CAV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """Make a single-lane roundabout road network."""
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 24  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, c], [c, c]]
        # line = [[c, s], [n, c]]
        for lane in [0]:
            net.add_lane(
                "se",
                "ex",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 - alpha),
                    np.deg2rad(alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ex",
                "ee",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(alpha),
                    np.deg2rad(-alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ee",
                "nx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-alpha),
                    np.deg2rad(-90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "nx",
                "ne",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 + alpha),
                    np.deg2rad(-90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "ne",
                "wx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-90 - alpha),
                    np.deg2rad(-180 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "wx",
                "we",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(-180 + alpha),
                    np.deg2rad(-180 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "we",
                "sx",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(180 - alpha),
                    np.deg2rad(90 + alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
            net.add_lane(
                "sx",
                "se",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90 + alpha),
                    np.deg2rad(90 - alpha),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        net.add_lane(
            "ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "ses",
            "se",
            SineLane(
                [2 + a, dev / 2],
                [2 + a, dev / 2 - delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sx",
            "sxs",
            SineLane(
                [-2 - a, -dev / 2 + delta_en],
                [-2 - a, dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c))
        )

        net.add_lane(
            "eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
        )
        net.add_lane(
            "ees",
            "ee",
            SineLane(
                [dev / 2, -2 - a],
                [dev / 2 - delta_st, -2 - a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "ex",
            "exs",
            SineLane(
                [-dev / 2 + delta_en, 2 + a],
                [dev / 2, 2 + a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c))
        )

        net.add_lane(
            "ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c))
        )
        net.add_lane(
            "nes",
            "ne",
            SineLane(
                [-2 - a, -dev / 2],
                [-2 - a, -dev / 2 + delta_st],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nx",
            "nxs",
            SineLane(
                [2 + a, dev / 2 - delta_en],
                [2 + a, -dev / 2],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c))
        )

        net.add_lane(
            "wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c))
        )
        net.add_lane(
            "wes",
            "we",
            SineLane(
                [-dev / 2, 2 + a],
                [-dev / 2 + delta_st, 2 + a],
                a,
                w,
                -np.pi / 2,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wx",
            "wxs",
            SineLane(
                [dev / 2 - delta_en, -2 - a],
                [-dev / 2, -2 - a],
                a,
                w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),
            ),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c))
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self, num_CAV: int) -> None:
        """Populate the roundabout with vehicles based on the defined road network."""
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        
        # --- 1. Define Valid Entry Lanes ---
        # These are the first segments of the access roads defined in _make_road
        # Example: "ser" -> "ses" is the initial straight road segment leading to the S-E entry.
        entry_lanes = [
            ("ser", "ses", 0),  # South-East Access Road
            ("eer", "ees", 0),  # East-Exit Access Road
            ("ner", "nes", 0),  # North-East Access Road
            ("wer", "wes", 0),  # West-Exit Access Road
        ]
        
        # --- 2. Multi-Agent (CAV) Spawning ---
        num_controlled = self.config.get("controlled_vehicles", num_CAV)
        print(f"Spawning {num_controlled} controlled vehicles (CAVs).")
        
        # Determine the spacing for initial vehicle placement
        initial_spacing = 15  # [m] spacing between spawned vehicles on the same access road
        
        for i in range(num_controlled):
            # Cycle through the available entry lanes
            lane_index = entry_lanes[i % len(entry_lanes)]
            lane = road.network.get_lane(lane_index)
            
            # Position vehicles along the entry lane for a clean start
            # Place vehicles at varying longitudinal positions (s=15m, 30m, 45m, etc.)
            position = lane.position(initial_spacing * (i // len(entry_lanes) + 1), 0)
            
            # Initial speed and type
            initial_speed = np.random.uniform(10, 15)
            ego_vehicle = self.action_type.vehicle_class(road, position, speed=initial_speed)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        # self.vehicle = self.controlled_vehicles[0]  # For compatibility with base class
        
        # --- 3. Other (HDV) Spawning ---
        # Spawn HDVs on different lanes or with different spacing
        num_HDV = 0
        for i in range(num_HDV):
            # Use a different lane for HDVs or just continue the cycle
            lane_index = entry_lanes[i % len(entry_lanes)]
            lane = road.network.get_lane(lane_index)
            
            # Position HDVs further back or interspersed
            # Use a larger longitudinal position (e.g., 25m * i)
            position = lane.position(25 * (i + 1), 0)
            
            road.vehicles.append(
                other_vehicles_type(road, position, speed=np.random.uniform(10, 15)))

class RoundaboutEnvMARL(RoundaboutEnv):
    """
    Multi-Agent version of the Roundabout Environment.
    Inherits all road and reward logic from RoundaboutEnv but enforces MultiAgent I/O.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 8 # Example: control 8 vehicles
        })
        return config

# --- Environment Registration ---

register(
    id='aa228-v0', # Single-Agent version (if needed later)
    entry_point='highway_env.envs:RoundaboutEnv',
)

register(
    id='aa228-multi-agent-v0', # Multi-Agent version for your MAPPO script
    entry_point='highway_env.envs:RoundaboutEnvMARL',
)