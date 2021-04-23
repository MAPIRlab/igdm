import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import datetime
import sys
import pandas as pd

from gdm.common import DiscreteScalarMap, Observation
from gdm.utils.benchmarking import ExecutionTimer
from gdm.gmrf import GMRF_Gas_Wind_Efficient
from igdm.pomdp.report import comparePDFWithGroundtruth, computePathLength
from igdm.pomdp.pomdp import IGDM, Pose, vectorToAngle
from igdm.pomdp.random_gdm import AUC, BrownianDiscrete, BrownianDiscreteNoReturn
from gdm.common import EnvironmentGroundTruth, EnvironmentRunningGroundTruth


class Test:
    class TestConfig:
        def __init__(self):
            self.scenario_id = 1
            self.start = "A"
            self.report_folder = "/home/andy/tmp/IGDM_reports"
            self.strategy = "igdm"
            self.max_steps = 20
            self.step_size = 1
            self.sensor_noise = 0
            self.report_every_steps = 5
            self.plot_every_steps = 5
            self.log_to_hdd = False
            self.plot_path = True
            self.plot_gdm = True
            self.weights = IGDM.Weights()
            self.verbose = True
            self.test_name = "",
            self.plot_display = False
            self.look_ahead_steps = 3
            self.coarse_resolution = 1.0
            self.iteration = 0 # If I want to run the same experiment several times
            self.past_path = []

            return

        def __str__(self):
            main_report = \
                "scenario_id " + str(self.scenario_id) + "\n" + \
                "start " + str(self.start) + "\n" + \
                "strategy " + str(self.strategy) + "\n" + \
                "step_size " + str(self.step_size) + "\n" + \
                "sensor_noise " + str(self.sensor_noise) + "\n" + \
                "report_folder " + str(self.report_folder) + "\n" + \
                "max_steps " + str(self.max_steps) + "\n" + \
                "coarse_resolution " + str(self.coarse_resolution) + "\n" + \
                "report_every_steps " + str(self.report_every_steps) + "\n" + \
                "plot_every_steps " + str(self.plot_every_steps) + "\n" + \
                "report_folder " + str(self.report_folder) + "\n" + \
                "H " + str(self.look_ahead_steps) + "\n"

            igdm_report = ""
            if self.strategy == "igdm":
                igdm_report = \
                    "weight.kld " + str(self.weights.kld) + "\n" + \
                    "weight.movement " + str(self.weights.movement) + "\n" + \
                    "weight.gas " + str(self.weights.gas) + "\n" + \
                    "weight.gas_saturation " + str(self.weights.gas_saturation) + "\n" + \
                    "weight.gamma " + str(self.weights.gamma) + "\n"

            return main_report + igdm_report

    def __init__(self, test_config):
        print(test_config)

        self.test_config = test_config
        self.environment = self._getEnvironment()
        self.start_position = self._getStartPosition()
        self.long_experiment_name = self._getLongExperimentName()
        self.auto_gdm = self._getAutoGdm()

        self._run()
        return

    def _getEnvironment(self):
        if (self.test_config.scenario_id == 1):
            from gdm.common.environments import corridor_1_running as env_data
            environment = env_data.corridor_1
        elif (self.test_config.scenario_id == 2):
            from gdm.common.environments import corridor_2_running as env_data
            environment = env_data.corridor_2
        elif (self.test_config.scenario_id == 3):
            from gdm.common.environments import mapir_4b_running as env_data
            environment = env_data.mapir_4b
        elif (self.test_config.scenario_id == 4):
            from gdm.common.environments import corridor_4_running as env_data
            environment = env_data.corridor_4
        elif (self.test_config.scenario_id == 5):
            from gdm.common.environments import corridor_5_running as env_data
            environment = env_data.corridor_5
        elif (self.test_config.scenario_id == 10):
            environment = getSquareScenario(3)
        elif (self.test_config.scenario_id == 15):
            environment = getSquareScenario(10)
        elif (self.test_config.scenario_id == 16):
            environment = getSquareScenario(20)
        elif (self.test_config.scenario_id == 17):
            environment = getSquareScenario(31)
        elif (self.test_config.scenario_id == 18):
            environment = getSquareScenario(5)
        elif (self.test_config.scenario_id == 19):
            environment = getSquareScenario(18)
        else:
            assert False
        return environment

    def _getStartPosition(self):
        if type(self.test_config.start) == tuple:
            start_position = self.test_config.start

        elif (self.test_config.scenario_id == 1 or self.test_config.scenario_id == 2 or self.test_config.scenario_id == 4 or self.test_config.scenario_id == 5):
            if self.test_config.start == "A":
                start_position = (2.5, 1.5)  # A
            elif self.test_config.start == "B":
                start_position = (7.5, 1.5)  # B
            elif self.test_config.start == "C":
                start_position = (12.5, 1.5)  # C
            elif self.test_config.start == "D":
                start_position = (2.5, 4)  # D
            elif self.test_config.start == "E":
                start_position = (7.5, 4)  # E
            elif self.test_config.start == "F":
                start_position = (12.5, 4)  # F
            else:
                assert False
        elif (self.test_config.scenario_id == 3):
            if self.test_config.start == "A":
                start_position = (2.5, 2)  # A
            elif self.test_config.start == "B":
                start_position = (2.5, 7)  # B
            elif self.test_config.start == "C":
                start_position = (5, 9)  # C
            elif self.test_config.start == "D":
                start_position = (6, 7)  # D
            elif self.test_config.start == "E":
                start_position = (9.5, 6)  # E
            elif self.test_config.start == "F":
                start_position = (7.5, 2)  # F
            else:
                assert False
        elif (self.test_config.scenario_id == 14):
            if self.test_config.start == "A":
                start_position = (1.5, 1.5)  # A
            elif self.test_config.start == "B":
                start_position = (1.1, 1.1)  # B
            else:
                assert False

        elif (self.test_config.scenario_id == 15):
            if self.test_config.start == "A":
                start_position = (5, 5)  # A
            elif self.test_config.start == "B":
                start_position = (1.1, 1.1)  # B
            else:
                assert False

        elif (self.test_config.scenario_id == 16):
            if self.test_config.start == "A":
                start_position = (10, 10)  # A
            elif self.test_config.start == "B":
                start_position = (1.1, 1.1)  # B
            else:
                assert False

        elif (self.test_config.scenario_id == 17):
            if self.test_config.start == "A":
                start_position = (15, 15)  # A
            elif self.test_config.start == "B":
                start_position = (1.1, 1.1)  # B
            else:
                assert False
        else:
            assert False
        return start_position

    def _getAutoGdm(self):
        d = self.test_config.step_size
        actions = [(d, 0), (-d, 0), (0, d), (0, -d)]

        if self.test_config.strategy == "igdm":
            obstacle_map = self.environment.obstacles

            if type(self.test_config.weights) == IGDM.Weights:
                igdm_weights = self.test_config.weights
            else:
                print("Using default IGDM weights")
                igdm_weights = IGDM.Weights()

            gdm = GMRF_Gas_Wind_Efficient(obstacle_map, resolution=obstacle_map.resolution)
            coarse_gdm = GMRF_Gas_Wind_Efficient(obstacle_map, resolution=self.test_config.coarse_resolution)

            igdm = IGDM(obstacle_map=obstacle_map,
                        gdm=gdm,
                        coarse_gdm=coarse_gdm,
                        movement_actions=actions,
                        weights=igdm_weights,
                        look_ahead_steps=self.test_config.look_ahead_steps,
                        verbose=self.test_config.verbose)
            auto_gdm = igdm

        elif self.test_config.strategy == "auc":
            gdm = GMRF_Gas_Wind_Efficient(fine_om)
            auto_gdm = AUC(gdm=gdm, obstacle_map=obstacle_map, d=d, start_position=start_position)

        elif self.test_config.strategy == "brownian":
            gdm = GMRF_Gas_Wind_Efficient(fine_om)
            auto_gdm = BrownianDiscrete(gdm=gdm, obstacle_map=obstacle_map, actions=actions, min_movement_dist=0.05)

        elif self.test_config.strategy == "brownian_no_return":
            gdm = GMRF_Gas_Wind_Efficient(fine_om)
            auto_gdm = BrownianDiscreteNoReturn(gdm=gdm, obstacle_map=obstacle_map, actions=actions,
                                                min_movement_dist=0.05)
        else:
            assert False, self.test_config.strategy
        return auto_gdm

    def _getLongExperimentName(self):
        main_name = \
            self.test_config.strategy + \
            "_S" + str(self.test_config.scenario_id) +  str(self.test_config.start) + \
            "_SS" + str(self.test_config.step_size) + \
            "_N" + str(self.test_config.sensor_noise) + \
            "_I" + str(self.test_config.iteration) + \
            "_H" + str(self.test_config.look_ahead_steps) + \
            "_CR" + str(self.test_config.coarse_resolution)

        gdm_name = ""
        if self.test_config.strategy == "igdm":
            gdm_name = "_" + self.test_config.weights.name

        return main_name + gdm_name

    def _run(self):

        ## Initial conditions
        path = test_config.past_path
        path_length = 0
        current_position = self.start_position
        path += [current_position]
        current_pose = Pose(current_position, vectorToAngle((0, 1)))
        path_map = DiscreteScalarMap(dimensions=2,
                                     size=self.environment.obstacles.size,
                                     resolution=self.environment.obstacles.resolution)
        path_map.setPosition(path, 1)
        self.auto_gdm.addObservation(self.environment.getObservation(path))

        ## Logging
        if self.test_config.report_folder[-1] != '/':
            self.test_config.report_folder += '/'
        stamp = str(datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%s'))
        run_name = self.long_experiment_name + "_" + stamp
        folder = self.test_config.report_folder + run_name
        os.mkdir(folder)
        report_base_name = folder + '/report_' + run_name
        if (self.test_config.log_to_hdd):
            print("Logging to HDD. You will se no more printed text here")
            sys.stdout = open(report_base_name + '.txt', 'w')
        df_performance = pd.DataFrame()
        df_action = pd.DataFrame()


        ## PARAM DF
        df_params = pd.DataFrame(data={"strategy": [self.test_config.strategy],
                                       "scenario": [self.test_config.scenario_id],
                                       "start": [self.test_config.start],
                                       "stamp": [stamp],
                                       "step_size": [self.test_config.step_size],
                                       "sensor_noise": [self.test_config.sensor_noise],
                                       "coarse_resolution": [self.test_config.coarse_resolution],
                                       "igdm_H": [self.test_config.look_ahead_steps],
                                       "igdm_weights_kld": [self.test_config.weights.kld],
                                       "igdm_weights_movement": [self.test_config.weights.movement],
                                       "igdm_weights_movement_advance": [self.test_config.weights.movement_advance],
                                       "igdm_weights_movement_turn": [self.test_config.weights.movement_turn],
                                       "igdm_weights_gas": [self.test_config.weights.gas],
                                       "igdm_weights_gas_saturation": [self.test_config.weights.gas_saturation],
                                       "igdm_weights_gamma": [self.test_config.weights.gamma],
                                       "iteration": [self.test_config.iteration]})

        ## LOOP
        t = ExecutionTimer(self.long_experiment_name)
        step = 0
        while step <= self.test_config.max_steps:
            if (self.test_config.verbose): print("------------- Step: " + str(step) + " -------------")

            ## REPORT
            if (step) % self.test_config.report_every_steps == 0:
                df_performance_line = comparePDFWithGroundtruth(pdf=self.auto_gdm.getNormalDistributionMap(),
                                                                ground_truth=self.environment.gas,
                                                                step=step,
                                                                path_length=path_length,
                                                                run_time=t.getElapsed())

                if self.test_config.verbose:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width',
                                           1000):
                        print(df_performance_line)

                ## APPEND & RETURN
                df_extended_performance_line = pd.concat([df_params, df_performance_line], axis=1)
                df_performance = pd.concat([df_performance, df_extended_performance_line], axis=0, ignore_index=True)
                df_performance.to_csv(report_base_name + ".csv")
                df_action.to_csv(report_base_name + "_action.csv")

            ## PLOT
            if (step) % self.test_config.plot_every_steps == 0:
                if self.test_config.plot_path:
                    path_map.plot(save=folder + "/" + str(step) + "_path")
                if self.test_config.plot_gdm:
                    self.auto_gdm.plot(save=folder + "/" + str(step))

            ## COMPUTE NEXT BEST MOVEMENT GIVEN LAST OBSERVATIONS
            move, df_action_new = self.auto_gdm.getBestMovementAction(current_pose)
            df_action_new["step"] = step
            df_action = pd.concat([df_action, df_action_new], axis=0, ignore_index=True)

            ## UPDATE ROBOT POSITION AND SAMPLES
            extra_path = self.environment.obstacles.getFeasiblePathForVector(current_pose.position, move)
            extra_obs = self.environment.getObservation(extra_path, time=step / 4)[::8]
            if self.test_config.sensor_noise > 0:
                assert type(extra_obs) is list
                assert type(extra_obs[0]) is Observation
                for obs in extra_obs:
                    obs.gas = abs(np.random.normal(obs.gas, sensor_noise))
                    obs.wind = np.random.normal(obs.wind, sensor_noise)

            self.auto_gdm.addObservation(extra_obs)
            path += extra_path
            current_pose.position = path[-1]
            current_pose.orientation = vectorToAngle(move)
            assert self.environment.obstacles.getPosition(current_pose.position) < 0.5
            if (self.test_config.verbose): print(current_pose.position)

            ## UPDATE PATH
            path_length = computePathLength(path)
            if (self.test_config.verbose): print("Total path lenght:\t" + str(path_length))
            old = path_map.getPosition(extra_path)
            path_map.setPosition(extra_path, [1 + o for o in old])

            step += 1
            if (self.test_config.verbose): print("\n\n")
        return df_performance


##==============================================================================

def getSquareScenario(N):
    square = EnvironmentGroundTruth(size=(N, N), resolution=0.1)
    m = np.zeros(square.shape, dtype=np.byte)
    m[0, :] = 1
    m[:, 0] = 1
    m[-1, :] = 1
    m[:, -1] = 1
    square.obstacles.loadMatrix(m)
    assert square.obstacles.getPosition((0, 0)) == 1
    e = EnvironmentRunningGroundTruth(size=(N, N), resolution=0.1)
    e.environments = [square]
    e.obstacles = square.obstacles
    return e


##==============================================================================


if __name__ == "__main__":

    test_config = Test.TestConfig()
    if len(sys.argv) <= 1:
        test_config.scenario_id = 1
        test_config.start = (11, 4.85)
        test_config.verbose = True
        test_config.look_ahead_steps = 2
        test_config.max_steps = 5
        test_config.step_size = 1
        test_config.sensor_noise = 0
        test_config.report_every_steps = 5
        test_config.plot_every_steps = 5
        test_config.coarse_resolution = 0.1

        from gdm.common.environments import corridor_1_running
        test_config.past_path = []
        test_config.past_path += corridor_1_running.corridor_1.getStraightPathBetweenPositions((10.8, 2.5), (14, 2.5))
        test_config.past_path += corridor_1_running.corridor_1.getStraightPathBetweenPositions((14, 2.5), (14, 0.2))
        test_config.past_path += corridor_1_running.corridor_1.getStraightPathBetweenPositions((14, 0.2), (10.8, 0.2))
        test_config.past_path += corridor_1_running.corridor_1.getStraightPathBetweenPositions((10.8, 0.2),(10.8, 4.85))

        test_config.weights = IGDM.Weights()
        test_config.weights.kld *= test_config.coarse_resolution


    else:
        test_config.report_folder = str(sys.argv[1])
        test_config.scenario_id = int(sys.argv[2])
        test_config.start = str(sys.argv[3])
        test_config.strategy = str(sys.argv[4])
        test_config.step_size = float(sys.argv[5])
        test_config.sensor_noise = float(sys.argv[6])

        test_config.weights = IGDM.Weights()
        if float(sys.argv[7]) >= 0:
            test_config.weights.kld = float(sys.argv[7])
        if float(sys.argv[8]) >= 0:
            test_config.weights.movement = float(sys.argv[8])
        if float(sys.argv[9]) >= 0:
            test_config.weights.gas = float(sys.argv[9])
        if float(sys.argv[10]) >= 0:
            test_config.weights.gas_saturation = float(sys.argv[10])
        if float(sys.argv[11]) >= 0:
            test_config.weights.gamma = float(sys.argv[11])

        test_config.coarse_resolution = float(sys.argv[12])
        test_config.max_steps = int(sys.argv[13])
        test_config.report_every_steps = int(sys.argv[14])
        test_config.plot_every_steps = int(sys.argv[15])
        test_config.log_to_hdd = int(sys.argv[16])
        test_config.iteration = int(sys.argv[17])
        test_config.look_ahead_steps = int(sys.argv[18])

        if test_config.log_to_hdd == 0:
            test_config.log_to_hdd = False
        else:
            test_config.log_to_hdd = True
        test_config.verbose = False
        test_config.plot_display = False

    test_config.weights.name = \
        "kld_" + str("{:.3f}".format(test_config.weights.kld)) + \
        "_m_" + str("{:.3f}".format(test_config.weights.movement)) + \
        "_g_" + str("{:.3f}".format(test_config.weights.gas)) + \
        "_gs_" + str("{:.3f}".format(test_config.weights.gas_saturation)) + \
        "_G_" + str("{:.3f}".format(test_config.weights.gamma))

    Test(test_config)
