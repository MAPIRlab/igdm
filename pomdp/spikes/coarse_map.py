from gdm.utils.benchmarking import ExecutionTimer
import numpy as np
from gdm.common.environments.corridor_1_small import corridor_1_small, corridor_1_small_coarse
from gdm.gmrf import GMRF_Gas_Wind_Efficient, GMRF_Gas_Efficient, GMRF_Gas
environment = corridor_1_small
from gdm.utils.metrics import getKLD
import copy








##==============================================================================
##  FINE REF (Probably incorrect)
##==============================================================================

## TEST SETUP VARIABLES
environment = corridor_1_small
om_fine     = environment.obstacles
om_coarse   = corridor_1_small_coarse.toObstacleMap()
ccm_fine    = om_fine.toCellConnectivityMap()
ccm_coarse  = corridor_1_small_coarse



## GMD
gdm_fine = GMRF_Gas(ccm_fine)
gdm_p    = GMRF_Gas(ccm_coarse)
gdm_ref  = gdm_fine



## TEST PATH
path_1 = om_fine.getStraightPathBetweenPositions((2.5, 2.5), (2.5, 3.5))
path_2 = om_fine.getStraightPathBetweenPositions((2.5, 3.5), (3.5, 3.5))
path_3 = om_fine.getStraightPathBetweenPositions((3.5, 3.5), (2.5, 3.5))



## REAL DATA
observation = environment.getObservation(path_1)
gdm_fine.addObservation(observation)
gdm_p.addObservation(observation)
P = gdm_p.toNormalDistributionMap()
#P.plot(mean_max=25, variance_max=10)



## EXPECTATION INFORMATION GAIN
expected_obs = gdm_ref.getPosition(path_2)
gdm_q = copy.deepcopy(gdm_p)
gdm_q.addObservation(expected_obs)
Q = gdm_q.toNormalDistributionMap()
#Q.plot(mean_max=25, variance_max=10)
print(getKLD(P,Q))



## REAL DATA
observation = environment.getObservation(path_2)
gdm_fine.addObservation(observation)
gdm_p.addObservation(observation)
P = gdm_p.toNormalDistributionMap()
#P.plot(mean_max=25, variance_max=10)


## EXPECTATION INFORMATION GAIN
expected_obs = gdm_ref.getPosition(path_3)
gdm_q = copy.deepcopy(gdm_p)
gdm_q.addObservation(expected_obs)
Q = gdm_q.toNormalDistributionMap()
#Q.plot(mean_max=25, variance_max=10)
print(getKLD(P,Q))










##==============================================================================
##  COARSE REF (Probably correct)
##==============================================================================

## TEST SETUP VARIABLES
environment = corridor_1_small
om_fine     = environment.obstacles
om_coarse   = corridor_1_small_coarse.toObstacleMap()
ccm_fine    = om_fine.toCellConnectivityMap()
ccm_coarse  = corridor_1_small_coarse



## GMD
gdm_fine = GMRF_Gas(ccm_fine)
gdm_p    = GMRF_Gas(ccm_coarse)
gdm_ref  = gdm_p



## TEST PATH
path_1 = om_fine.getStraightPathBetweenPositions((2.5, 2.5), (2.5, 3.5))
path_2 = om_fine.getStraightPathBetweenPositions((2.5, 3.5), (3.5, 3.5))
path_3 = om_fine.getStraightPathBetweenPositions((3.5, 3.5), (2.5, 3.5))



## REAL DATA
observation = environment.getObservation(path_1)
gdm_fine.addObservation(observation)
gdm_p.addObservation(observation)
P = gdm_p.toNormalDistributionMap()
#P.plot(mean_max=25, variance_max=10)


## EXPECTATION INFORMATION GAIN
expected_obs = gdm_ref.getPosition(path_2)
gdm_q = copy.deepcopy(gdm_p)
gdm_q.addObservation(expected_obs)
Q = gdm_q.toNormalDistributionMap()
#Q.plot(mean_max=25, variance_max=10)
print(getKLD(P,Q))



## REAL DATA
observation = environment.getObservation(path_2)
gdm_fine.addObservation(observation)
gdm_p.addObservation(observation)
P = gdm_p.toNormalDistributionMap()
#P.plot(mean_max=25, variance_max=10)



## EXPECTATION INFORMATION GAIN
expected_obs = gdm_ref.getPosition(path_3)
gdm_q = copy.deepcopy(gdm_p)
gdm_q.addObservation(expected_obs)
Q = gdm_q.toNormalDistributionMap()
#Q.plot(mean_max=25, variance_max=10)
print(getKLD(P,Q))

