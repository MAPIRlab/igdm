if(__name__ is "__main__"):
	from gdm.gmrf import GMRF_Gas_Wind_Efficient
	from gdm.common.debug_setups.debug_setup_4 import obstacle_map, observations

	ggw = GMRF_Gas_Wind_Efficient(obstacle_map)
	ggw.addObservation(observations)
	ggw.estimate()
	#ggw.getGasEstimate().plot()
	#ggw.getWindEstimate().plot(vmax=1, interpol=1)
	ggw.getGasUncertainty().plot()



positions = [(2.5, 1.5),
             (2.5, 2.0),
             (2.5, 2.5),
             (1.5, 2.0),
             (1.0, 0.5),
             (2.5, 4.6),
             (0.8, 4.6)]

from gdm.utils.metrics import getKLD
from gdm.common import Observation
from gdm.common import DiscreteScalarMap


kld_map = DiscreteScalarMap.fromMatrix(100*obstacle_map.toMatrix(), 0.1)
kld_data = []
for p in positions:
	gas = ggw.getGasEstimate().getPosition(p)
	wind = ggw.getWindEstimate().getPosition(p)
	new_obs = Observation(position=p, gas=gas, wind=wind, data_type='gas+wind')
	new_ggw = GMRF_Gas_Wind_Efficient(obstacle_map)
	new_ggw.addObservation(observations+[new_obs])
	kld = getKLD(ggw.toNormalDistributionMap(), new_ggw.toNormalDistributionMap())
	kld_map.setPosition(p, kld)
	kld_data += [kld]
kld_map.plot()
print(kld_data)
new_ggw.getGasEstimate().plot()
new_ggw.getWindEstimate().plot(vmax=1, interpol=1)
new_ggw.getGasUncertainty().plot()



## Assume center
kld_map = DiscreteScalarMap.fromMatrix(100*obstacle_map.toMatrix(), 0.1)
kld_data = []
gass = [1,1,1,0.4,0,1,0]
winds = [(0,1),(0,1),(0,1),(0,0),(0,0),(0,1),(0,0)]
for i in range(0, len(positions)):
	p = positions[i]
	gas = gass[i]
	wind = winds[i]
	new_obs = Observation(position=p, gas=gas, wind=wind, data_type='gas+wind')
	new_ggw = GMRF_Gas_Wind_Efficient(obstacle_map)
	new_ggw.addObservation(observations)
	new_ggw.addObservation(new_obs)
	kld = getKLD(ggw.toNormalDistributionMap(), new_ggw.toNormalDistributionMap())
	kld_map.setPosition(p, kld)
	kld_data += [kld]
kld_map.plot()
print(kld_data)
new_ggw.getGasEstimate().plot()
new_ggw.getWindEstimate().plot(vmax=1, interpol=1)
new_ggw.getGasUncertainty().plot()



## Assume left
kld_map = DiscreteScalarMap.fromMatrix(100*obstacle_map.toMatrix(), 0.1)
kld_data = []
gass = [1,1,1,0.4,0,1,0]
winds = [(0,1),(0,1),(0,0),(1,0),(0,0),(0,0),(0,1)]
for i in range(0, len(positions)):
	p = positions[i]
	gas = gass[i]
	wind = winds[i]
	new_obs = Observation(position=p, gas=gas, wind=wind, data_type='gas+wind')
	new_ggw = GMRF_Gas_Wind_Efficient(obstacle_map)
	new_ggw.addObservation(observations+[new_obs])
	kld = getKLD(ggw.toNormalDistributionMap(), new_ggw.toNormalDistributionMap())
	kld_map.setPosition(p, kld)
	kld_data += [kld]
kld_map.plot()
print(kld_data)
new_ggw.getGasEstimate().plot()
new_ggw.getWindEstimate().plot(vmax=1, interpol=1)
new_ggw.getGasUncertainty().plot()




################################################################################
##  PATHS
################################################################################

path_c1 = obstacle_map.getFeasiblePathBetweenPositions((2.5, 0.5), (2.5, 1.5))
path_c2 = path_c1 + obstacle_map.getFeasiblePathBetweenPositions((2.5, 1.5), (2.5, 2.0))
path_c3 = path_c2 + obstacle_map.getFeasiblePathBetweenPositions((2.5, 2.0), (2.5, 2.5))
path_c4 = path_c3 + obstacle_map.getFeasiblePathBetweenPositions((2.5, 2.5), (2.5, 4.6))
path_l1 = path_c2 + obstacle_map.getFeasiblePathBetweenPositions((2.5, 2.0), (1.5, 2.0))
path_l1 = path_c2 + obstacle_map.getFeasiblePathBetweenPositions((1.5, 2.0), (1.0, 2.0)) + obstacle_map.getFeasiblePathBetweenPositions((1.0, 2.0), (1.0, 4.6))

