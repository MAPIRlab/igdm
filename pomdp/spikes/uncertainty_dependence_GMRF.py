import sys
sys.path.insert(0, "../../../..")


if(__name__ is "__main__"):

	from gdm.gmrf import GMRF_Gas_Wind_Efficient
	from gdm.common.environments.corridor_1 import corridor_1 as cor_1

	#cor_1.plot()

	path_1 = cor_1.obstacles.getFeasiblePathBetweenPositions((2.5, 2.5),(2.5,4))
	path_2 = cor_1.obstacles.getFeasiblePathBetweenPositions((2.5, 4),(5,4))

	path = path_1 + path_2
	obs = cor_1.getObservation(path)
	ggw = GMRF_Gas_Wind_Efficient(cor_1.obstacles)
	ggw.addObservation(obs).toNormalDistributionMap().plot()
	u1 = ggw.getGasUncertainty().toMatrix()

	from gdm.common import Observation
	position = (12,4)
	gas = ggw.getGasEstimate().getPosition(position)
	wind = ggw.getWindEstimate().getPosition(position)
	new_obs = Observation(position=position, gas=gas, wind=wind)
	ggw.addObservation(new_obs).toNormalDistributionMap().plot()
	u2 = ggw.getGasUncertainty().toMatrix()

	import numpy as np
	from gdm.utils.report import plotScalarField
	plotScalarField(u1-u2)
	plotScalarField(np.abs(u1-u2))
