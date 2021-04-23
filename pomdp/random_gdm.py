import math
import pickle
from abc import abstractmethod

import numpy as np
import pandas as pd
from gdm.common import NormalGasDistributionMapper, ObstacleMap, \
    Observation
from gdm.utils.benchmarking import ExecutionTimer
from gdm.utils.metrics import getKLD
from .pomdp import *


##==============================================================================
class AUC(AutoGDM):

    def __init__(self, gdm, obstacle_map, start_position, d=1):
        assert isinstance(obstacle_map, ObstacleMap)

        AutoGDM.__init__(self, name="advance_until_collision", gdm=gdm)
        self._d = d
        self._last_position = start_position
        self._om = obstacle_map

    def _addObservation(self, observation):
        return self

    def _plot(self, save):
        return self

    def _getBestMovementAction(self, current_pose):
        current_position = current_pose.position
        current_angle = current_pose.orientation
        step_size = self._d

        ## CHECK FOR COLISSION -> Rotate random
        moved = math.sqrt(
            (self._last_position[0] - current_position[0]) ** 2 + (self._last_position[1] - current_position[1]) ** 2)
        if (moved <= 0.9 * step_size):
            has_obstacle = True
            while has_obstacle:
                random_angle = np.random.uniform(-np.pi, np.pi)
                new_move_vector = (step_size * math.cos(random_angle), step_size * math.sin(random_angle))
                has_obstacle = self._om.hasObstacleForVector(current_position, new_move_vector)
        else:
            new_move_vector = (step_size * math.cos(current_angle), step_size * math.sin(current_angle))

        ## UPDATE AND RETURN
        self._last_position = current_position
        df = pd.DataFrame()
        return new_move_vector, df


class BrownianDiscrete(AutoGDM):

    def __init__(self, gdm, obstacle_map, actions, min_movement_dist):
        assert isinstance(obstacle_map, ObstacleMap)

        AutoGDM.__init__(self, name="discrete_brownian", gdm=gdm)
        self._min_movement_dist = min_movement_dist
        self._om = obstacle_map
        self._actions = actions
        return

    def _addObservation(self, observation):
        return self

    def _plot(self, save):
        return self

    def _getBestMovementAction(self, current_pose):

        path_length = 0
        while path_length < self._min_movement_dist:
            random_selection = int(np.random.uniform(0, len(self._actions)))
            if random_selection >= len(self._actions):
                random_selection = len(self._actions) - 1
            move_vector = self._actions[random_selection]

            path = self._om.getFeasiblePathForVector(current_pose.position, move_vector)
            path_length = np.sqrt(
                (current_pose.position[0] - path[-1][0]) ** 2 + (current_pose.position[1] - path[-1][1]) ** 2)

        df = pd.DataFrame()
        return move_vector, df


class BrownianDiscreteNoReturn(AutoGDM):

    def __init__(self, gdm, obstacle_map, actions, min_movement_dist):
        assert isinstance(obstacle_map, ObstacleMap)

        AutoGDM.__init__(self, name="discrete_brownian_no_return", gdm=gdm)
        self._min_movement_dist = min_movement_dist
        self._om = obstacle_map
        self._actions = actions
        self._previous_action = (0, 0)
        return

    def _addObservation(self, observation):
        return self

    def _plot(self, save):
        return self

    def _getBestMovementAction(self, current_pose):

        path_length = 0
        is_repeated_action = True
        while path_length < self._min_movement_dist or is_repeated_action:
            random_selection = int(np.random.uniform(0, len(self._actions)))
            if random_selection >= len(self._actions):
                random_selection = len(self._actions) - 1
            move_vector = self._actions[random_selection]

            path = self._om.getFeasiblePathForVector(current_pose.position, move_vector)
            path_length = np.sqrt(
                (current_pose.position[0] - path[-1][0]) ** 2 + (current_pose.position[1] - path[-1][1]) ** 2)

            if (np.allclose(np.array(move_vector) + np.array(self._previous_action), np.zeros(2), atol=1e-01)):
                is_repeated_action = True
            else:
                is_repeated_action = False

        self._previous_action = move_vector
        df = pd.DataFrame()
        return move_vector, df
