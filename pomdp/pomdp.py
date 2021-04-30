import pickle
from abc import abstractmethod

import numpy as np
import pandas as pd
from gdm.common import NormalGasDistributionMapper, ObstacleMap, \
    Observation
from gdm.utils.metrics import getKLD


##==============================================================================

class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation
        return

def getAngleBetween(v1, v2):
    assert type(v1) is type(v2) is tuple
    if (np.allclose(np.array(v1), np.zeros(2), 0.01) or np.allclose(np.array(v2), np.zeros(2), 0.01)):
        return 0

    angle = vectorToAngle(v2) - vectorToAngle(v1)
    while (angle > 2 * np.pi):
        angle -= 2 * np.pi
    while (angle < -2 * np.pi):
        angle += 2 * np.pi
    return angle

def vectorToAngle(vector):
    vector = np.array(vector)
    angle = np.arctan2(vector[1], vector[0])
    return angle

def angleToVector(angle):
    angle = np.array(angle)
    return (np.cos(angle), np.sin(angle))

def sigmoid(x, max):
    t = np.array(x) / np.array(max)
    s = (np.e ** t - np.e ** (-t)) / (np.e ** t + np.e ** (-t))
    return s


##==============================================================================
class AutoGDM:

    def __init__(self, name, gdm):
        assert type(name) is str
        assert isinstance(gdm, NormalGasDistributionMapper)

        self.name = name
        self._gdm = gdm

        return

    ## METHODS -----------------------------------------------------------------

    def getBestMovementAction(self, current_pose):
        action, df = self._getBestMovementAction(current_pose)
        assert type(action) is tuple
        assert type(df) is pd.DataFrame
        return action, df

    def addObservation(self, observation):
        assert (type(observation) is Observation) or (type(observation) is list and type(observation[:] is Observation))
        self._gdm.addObservation(observation)
        self._addObservation(observation)
        return self

    def getNormalDistributionMap(self):
        return self._gdm.toNormalDistributionMap()

    def plot(self, save):
        self._gdm.toNormalDistributionMap().plot(save=save, mean_max=0.5, variance_max=3)
        self._gdm.getWindEstimate().plot(save=save + "_wind")
        self._plot(save)
        return self

    ## Abstract ----------------------------------------------------------------

    @abstractmethod
    def _getBestMovementAction(self, current_pose):
        pass

    @abstractmethod
    def _addObservation(self, observation):
        pass

    @abstractmethod
    def _plot(self, save):
        pass


##==============================================================================
class IGDM(AutoGDM):
    class Weights:
        def __init__(self):
            ## FUNCTION SPECIFIC
            self.movement_turn = 1 / np.pi  # cost/rad
            self.movement_advance = 1  # cost/m
            self.gas_saturation = 0.25

            ## POMPD
            self.kld = 1
            self.movement = 1
            self.gas = 8
            self.gamma = 0.5

            self.name = "default"

            return

        def __str__(self):
            s = ""
            s += "KLD: " + str(self.kld) + "\t"
            s += "MOV: " + str(self.movement) + "\t"
            s += "GAS: " + str(self.gas) + "\t"
            s += "MOV_T: " + str(self.movement_turn) + "\t"
            s += "MOV_A: " + str(self.movement_advance) + "\t"
            s += "GASLIM: " + str(self.gas_saturation) + "\t"
            s += "GAMMA: " + str(self.gamma) + "\t"
            return s

    ## CONSTRUCTORS ------------------------------------------------------------

    def __init__(self, obstacle_map, gdm, coarse_gdm, movement_actions, weights, look_ahead_steps=3, verbose=False):

        ## CHECK ARGUMENTS
        assert isinstance(obstacle_map, ObstacleMap)
        assert isinstance(coarse_gdm, NormalGasDistributionMapper)
        assert type(movement_actions) is list and type(movement_actions[0]) is tuple
        assert type(weights) is self.Weights

        ## Init base
        AutoGDM.__init__(self, name="igdm", gdm=gdm)

        ## Prevent anything from getting into windows/doors - this might lead to bugs
        self._map = pickle.loads(pickle.dumps(obstacle_map))
        obstacle_map._data[0,:] = 1
        obstacle_map._data[-1, :] = 1
        obstacle_map._data[:,0] = 1
        obstacle_map._data[:, -1] = 1

        ## Member variables
        self._cgdm = coarse_gdm
        self._actions = movement_actions
        self._weights = weights
        self.name = "igdm"
        self._look_ahead_steps = look_ahead_steps
        self.__verbose = verbose

        return

    ## BASE CLASS -----------------------------------------------------------------

    def _getBestMovementAction(self, current_pose):
        look_ahead_steps = self._look_ahead_steps
        gammas = np.array([self._weights.gamma ** i for i in range(look_ahead_steps)])
        self._cgdm.estimate()

        assert self.__isPoseValid(current_pose)
        assert len(gammas) == look_ahead_steps
        assert gammas[0] == 1
        assert look_ahead_steps >= 1

        ## GET ACTION COMBINATIONS
        action_combinations, paths = \
            self.__computeActionCombinations(look_ahead_steps, current_pose.position)

        ## COMPUTE REWARDS
        df_rewards = []
        rewards = []
        for index in range(len(action_combinations)):

            ## COMPUTE INDIVIDUAL STEP REWARDS
            action_combination = action_combinations[index]
            path_segments = paths[index]
            raw_step_rewards_kld = self.__computeKLDReward(path_segments)
            raw_step_rewards_move_advance, raw_step_rewards_move_turn = self.__computeMovementReward(
                action_sequence=action_combination, pose=current_pose)
            raw_step_rewards_gas = self.__computeGasObservationReward(path_segments=path_segments)
            assert look_ahead_steps == len(action_combination) == len(path_segments)
            assert look_ahead_steps == len(raw_step_rewards_kld)
            assert look_ahead_steps == len(raw_step_rewards_move_advance)
            assert look_ahead_steps == len(raw_step_rewards_move_turn)
            assert look_ahead_steps == len(raw_step_rewards_gas)

            ## COMPUTE WEIGHTED REWARD
            ## If the weights are 0, use default values so that the total formula
            ## does not break
            reward_kld = self._weights.kld * (gammas * raw_step_rewards_kld).sum()
            reward_move_advance = self._weights.movement_advance * (gammas * raw_step_rewards_move_advance).sum()
            reward_move_turn = self._weights.movement_turn * (gammas * raw_step_rewards_move_turn).sum()
            reward_move = self._weights.movement * (reward_move_advance + reward_move_turn)
            reward_gas = self._weights.gas * (gammas * raw_step_rewards_gas).sum()
            assert reward_move > 0
            assert reward_gas >= 0

            if self._weights.kld > 0 and self._weights.movement > 0 and self._weights.gas > 0:
                reward = (reward_kld * (1 + reward_gas)) / reward_move
            elif self._weights.kld > 0 and self._weights.movement <= 0 and self._weights.gas <= 0:
                reward = reward_kld
            elif self._weights.kld <= 0 and self._weights.movement > 0 and self._weights.gas <= 0:
                reward = 1/reward_move
            elif self._weights.kld <= 0 and self._weights.movement <= 0 and self._weights.gas > 0:
                reward = reward_gas
            else:
                reward = 0
                assert False


            ## REPORT
            ## First complete reward, then, in separate columns, each step
            df_reward = pd.DataFrame(index=[index],
                                     data={"sequence": index,
                                           "kld": reward_kld,
                                           "move": reward_move,
                                           "gas": reward_gas,
                                           "reward": reward,
                                           "actions": str(action_combination)})
            df_concat = [df_reward]
            for step in range(0, look_ahead_steps):
                raw_step_reward_kld = raw_step_rewards_kld[step]
                raw_step_reward_move_advance = raw_step_rewards_move_advance[step]
                raw_step_reward_move_turn = raw_step_rewards_move_turn[step]
                raw_step_reward_gas = raw_step_rewards_gas[step]

                step_label_kld = str(step + 1) + "_kld"
                step_label_move_advance = str(step + 1) + "_adv"
                step_label_move_turn = str(step + 1) + "_trn"
                step_label_gas = str(step + 1) + "_gas"

                df_step = pd.DataFrame(index=[index],
                                       data={step_label_kld: raw_step_reward_kld,
                                             step_label_move_advance: raw_step_reward_move_advance,
                                             step_label_move_turn: raw_step_reward_move_turn,
                                             step_label_gas: raw_step_reward_gas})

                df_concat += [df_step]

            df_reward = pd.concat(df_concat, axis=1)
            df_rewards += [df_reward]
            rewards += [reward]

        ## GET ACTION WITH BEST REWARD
        best_reward = max(rewards)
        best_index = rewards.index(best_reward)
        best_action_combination = action_combinations[best_index]
        best_first_action = best_action_combination[0]
        assert type(best_first_action) is tuple

        ## Report
        df_reward = pd.concat(df_rewards, axis=0, ignore_index=True)
        if self.__verbose:
            print("Best combo: " + str(best_index) + " -> " + str(best_action_combination))
            print("Best movement action: " + str(best_first_action))

            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(df_reward)

        return best_first_action, df_reward

    def _addObservation(self, observation):
        self._cgdm.addObservation(observation)
        return self

    def _plot(self, save):
        self._cgdm.toNormalDistributionMap().plot(save=save + "_cg")
        self._cgdm.getWindEstimate().plot(save=save + "_cw")
        return self

    ## PRIVATE -----------------------------------------------------------------

    def __computeActionCombinations(self, n, current_position, previous_action=(0, 0)):

        ## CHECK ARGUMENTS
        assert n >= 0
        assert self.__isPositionValid(current_position)
        assert type(previous_action) == tuple and len(previous_action) == 2

        ## VARIABLES
        action_combinations = []  # list of list of actions
        path_combinations = []  # list of list of paths (list of poses)

        ## CHECK RECURSIVITY HALT CONDITIONS
        if n < 1:
            return [[]], [[]]

        ## COMPUTE VECTOR COMBINATIONS AND PATH
        ## Recursively:
        ## - For all possible available actions
        ##   - If the action does not "undo" the last (past) action
        ##     - Compute effect of action (limited by obstacles and such)
        ##     - If the effect is non-zero (i.e, there is actual movement)
        ##       - Compute the branch of sub actions for current action (recursion)
        ##       - Attach current action to beguinning of branch
        for action in self._actions:

            effect_consecutive_action = np.array(previous_action) + np.array(action)
            if not np.allclose(effect_consecutive_action, np.zeros(2), atol=1e-01):

                extra_path = self._map.getFeasiblePathForVector(current_position, action)
                next_position = extra_path[-1]
                feasible_action = (
                    round(next_position[0] - current_position[0], 4), round(next_position[1] - current_position[1], 4))

                feasible_action_distance = np.linalg.norm(feasible_action)
                initial_action_distance = np.linalg.norm(action)
                if feasible_action_distance > initial_action_distance * 0.1:
                    sub_actions, sub_paths = self.__computeActionCombinations(n - 1, next_position, action)
                    action_combinations += [[feasible_action] + a for a in sub_actions]
                    path_combinations += [[extra_path] + p for p in sub_paths]

        ## RETURN
        assert len(action_combinations) == len(path_combinations)
        assert type(action_combinations) is list and len(action_combinations) > 0
        assert type(action_combinations[0]) is list and len(action_combinations[0]) > 0
        assert type(action_combinations[0][0]) is tuple
        assert type(path_combinations) is list and len(path_combinations) > 0
        assert type(path_combinations[0]) is list and len(path_combinations[0]) > 0
        assert type(path_combinations[0][0]) is list and len(path_combinations[0][0]) > 0
        assert type(path_combinations[0][0][0]) is tuple
        return action_combinations, path_combinations

    def __computeKLDReward(self, path_segments):
        kld_gains = []
        gdm_exp = pickle.loads(pickle.dumps(self._cgdm)) #Clone coarse map
        for path_segment in path_segments:
            kld_gain, gdm_exp = self.__computeExpectedRelativeEntropyGain(gdm_exp, path_segment)
            kld_gains += [kld_gain]

        assert len(kld_gains) == len(path_segments) == self._look_ahead_steps
        return kld_gains

    def __computeExpectedRelativeEntropyGain(self, gdm_ref, path):
        assert isinstance(gdm_ref, NormalGasDistributionMapper)

        kld_gain = 0
        gdm_exp = pickle.loads(pickle.dumps(gdm_ref))
        assert np.all(np.equal(gdm_ref.getGasUncertainty().toMatrix(), gdm_exp.getGasUncertainty().toMatrix()))

        if len(path) > 2:
            extra_obs = gdm_ref.getPosition(path)
            gdm_exp.addObservation(extra_obs)
            P = gdm_exp.toNormalDistributionMap()
            Q = gdm_ref.toNormalDistributionMap()
            kld_gain = getKLD(P, Q)

            # Sanity check: I'm getting a weird bug. I want to check that uncertainty is always decreased
            """
            assert Q.shape == P.shape
            assert len(gdm_ref._observations) <= len(gdm_exp._observations)
            if not np.all(np.greater_equal(Q.getVariance().toMatrix(),P.getVariance().toMatrix())):
                print("Path:")
                print(path)
                print("Observations:")
                for o in extra_obs:
                    print(o)
                print(" -- ")
                print("Q:\n" + str(Q.getVariance().toMatrix()) + "\n\nP:\n" + str(P.getVariance().toMatrix()) + "\n\nQ>=P:\n" + str(np.greater_equal(Q.getVariance().toMatrix(),P.getVariance().toMatrix())))
                gdm_ref.getGasEstimate().plot(save='/home/andy/ref_g.png')
                gdm_ref.getWindEstimate().plot(interpol=1, vmax=0.6, save='/home/andy/ref_w.png')
                gdm_ref.getGasUncertainty().plot(save='/home/andy/ref_u.png')
                gdm_exp.getGasEstimate().plot(save='/home/andy/exp_g.png')
                gdm_exp.getWindEstimate().plot(interpol=1, vmax=0.6, save='/home/andy/exp_w.png')
                gdm_exp.getGasUncertainty().plot(save='/home/andy/exp_u.png')
                assert False
            assert P.getVariance().min() <= Q.getVariance().min(), "P_var_min: " + str(P.getVariance().min()) + " Q_var_min: " + str(Q.getVariance().min()) + " kld_gain: " + str(kld_gain)
            assert P.getVariance().max() <= Q.getVariance().max(), "P_var_max: " + str(P.getVariance().max()) + " Q_var_max: " + str(Q.getVariance().max()) + " kld_gain: " + str(kld_gain)
            """

        assert id(gdm_ref) != id(gdm_exp)
        assert kld_gain >= -0.1, "Got: " + str(kld_gain) #Numeric noise migh lead to negative values in the range of -0.000001. Let's keep those too
        assert type(kld_gain) is not type(gdm_ref)
        assert type(gdm_exp) is type(gdm_ref)

        return kld_gain, gdm_exp

    def __computeMovementReward(self, action_sequence, pose):
        assert type(action_sequence) is list
        assert len(action_sequence) > 0

        ## LINEAR MOVEMENT
        distances = [np.linalg.norm(np.array(vector)) for vector in action_sequence]

        ## ROTATION
        orientation = pose.orientation
        assert (-np.pi - 0.1) <= orientation <= (np.pi + 0.1)
        last_orientation = orientation
        angles = []
        for action in action_sequence:
            angle = getAngleBetween(angleToVector(last_orientation), action)
            last_orientation += angle
            angles += [abs(angle)]

        assert len(distances) == len(angles)
        return distances, angles

    def __computeGasObservationReward(self, path_segments):
        assert type(path_segments) is list
        assert type(path_segments[0]) is list
        assert type(path_segments[0][0]) is tuple

        gas_estiamte = self._cgdm.getGasEstimate()
        rewards = []
        for path_segment in path_segments:
            gas_estimates = gas_estiamte.getPosition(path_segment)
            average_expected_gas = sum(gas_estimates) / len(path_segment)
            reward = sigmoid(average_expected_gas, self._weights.gas_saturation)
            rewards += [reward]

        assert len(path_segments) == len(rewards)
        return rewards

    def __isPositionValid(self, position):
        return self._map.isPositionValid(position)

    def __isPoseValid(self, pose):
        return (type(pose) is Pose) and (self.__isPositionValid(pose.position))
