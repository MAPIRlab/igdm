import sys
sys.path.insert(0, "..")
from gdm.gmrf import GMRF_Gas_Wind, GMRF_Gas_Wind_Parallel
from gdm.common import Observation
import billiard
from gdm.utils.metrics import getKLD
import numpy as np




##========================================================================


def estimateInformationGain(environment,
                            past_path,
                            extra_path,
                            current_gdm_prediction):
	predicted_new_obs = []
	for p in extra_path:
		g = current_gdm_prediction.gas.getPosition(p)
		w = current_gdm_prediction.wind.getPosition(p)
		predicted_new_obs += [Observation(position=p, gas=g, wind=w, data_type='gas+wind')]

	past_obs = environment.getObservation(past_path)
	all_obs = past_obs + predicted_new_obs

	Q = GMRF_Gas_Wind(environment.obstacles)
	Q.addObservation(all_obs).predict().computeUncertainty()

	kld = getKLD(current_gdm_prediction.getGasProbabilityDistribution(), Q.getGasProbabilityDistribution())
	return kld



def compactEstimateInformationGain(data):
	environment = data[0]
	past_path   = data[1]
	extra_path  = data[2]
	current_gdm_prediction = data[3]

	return  estimateInformationGain(environment,past_path,extra_path,current_gdm_prediction)



##========================================================================

def computeRealInformationGain(environment,
                            past_path,
                            extra_path,
                            current_gdm_prediction):

	total_path = past_path + extra_path
	all_obs = environment.getObservation(total_path)
	Q = GMRF_Gas_Wind(environment.obstacles)
	Q.addObservation(all_obs).predict().computeUncertainty()
	kld = getKLD(current_gdm_prediction.getGasProbabilityDistribution(), Q.getGasProbabilityDistribution())
	return kld



def compactComputeInformationGain(data):
	environment = data[0]
	past_path   = data[1]
	extra_path  = data[2]
	current_gdm_prediction = data[3]

	return computeRealInformationGain(environment,past_path,extra_path,current_gdm_prediction)


##========================================================================
def autonomous(environment, initial_path, hop=0.5):

	path = initial_path
	obstacle_map = environment.obstacles
	P   = GMRF_Gas_Wind_Parallel(obstacle_map)

	i = 0
	while True:
		i+=1
		obs = environment.getObservation(path)
		print("Computing true estimate")
		P.reset().addObservation(obs).predict().computeUncertainty()
		#P.gas.plot()
		P.gas_uncertainty.plot()

		#print("Path:")
		#print("[", end=" ")
		#for p in path:
		#	print(p, end ="")
		#	print(",", end="")
		#print("]")

		#print("Observations")
		#for o in obs:
		#	print(str(o.position) + "\t" + str(o.gas))
		#print("")

		print("------------------------------\n\n")


		current_position = path[-1]
		x = current_position[0]
		y = current_position[1]

		extra_path_t = obstacle_map.getStraightPathBetweenPositions(current_position, (x, y + hop))
		extra_path_b = obstacle_map.getStraightPathBetweenPositions(current_position, (x, y - hop))
		extra_path_l = obstacle_map.getStraightPathBetweenPositions(current_position, (x - hop, y))
		extra_path_r = obstacle_map.getStraightPathBetweenPositions(current_position, (x + hop, y))
		"""
		kld_t = estimateInformationGain(environment, path, extra_path_t, P)
		kld_b = estimateInformationGain(environment, path, extra_path_b, P)
		kld_l = estimateInformationGain(environment, path, extra_path_l, P)
		kld_r = estimateInformationGain(environment, path, extra_path_r, P)

		KLD_estimation = np.array((kld_t, kld_b, kld_l, kld_r))
		"""


		with billiard.Pool(4) as pool:
			print("Computing hypothetical information gain")
			data_t = (environment, path, extra_path_t, P)
			data_b = (environment, path, extra_path_b, P)
			data_l = (environment, path, extra_path_l, P)
			data_r = (environment, path, extra_path_r, P)
			klds = pool.map(compactEstimateInformationGain,[data_t, data_b, data_l, data_r])
			KLD_estimation = np.array(klds)
			print("KLD_estimates:\t" + str(KLD_estimation))

			#print("Computing real information gain")
			#klds_real = pool.map(compactComputeInformationGain,[data_t, data_b, data_l, data_r])
			#KLD_real = np.array(klds_real).astype(int)
			#print("KLD_real:\t" + str(KLD_real))


		KLD_best = KLD_estimation.max()

		if (KLD_best == KLD_estimation[0]):
			path += extra_path_t
			print("Move up")
		elif (KLD_best == KLD_estimation[1]):
			path += extra_path_b
			print("Move down")
		elif (KLD_best == KLD_estimation[2]):
			path += extra_path_l
			print("Move left")
		elif (KLD_best == KLD_estimation[3]):
			path += extra_path_r
			print("Move right")
		else:
			print("ERROR!!")



if __name__ == '__main__':
	from gdm.common.environments.corridor_1_small import corridor_1_small
	init_path = [ (2.5, 2.5) , (2.5, 2.450000000002) , (2.5, 2.400000000004) , (2.5, 2.350000000006)]
	autonomous(corridor_1_small, init_path)

