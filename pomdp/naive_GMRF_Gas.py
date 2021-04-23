import sys
sys.path.insert(0, "..")
from gdm.gmrf import GMRF_Gas
from gdm.common.observation import Observation
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
		predicted_new_obs += [Observation(position=p, gas=g, data_type='gas')]

	past_obs = environment.getObservation(past_path)
	all_obs = past_obs + predicted_new_obs

	Q = GMRF_Gas(environment.obstacles)
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
	Q = GMRF_Gas(environment.obstacles)
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
	P   = GMRF_Gas(obstacle_map)

	i = 0
	while True:
		i+=1
		obs = environment.getObservation(path)
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
			data_t = (environment, path, extra_path_t, P)
			data_b = (environment, path, extra_path_b, P)
			data_l = (environment, path, extra_path_l, P)
			data_r = (environment, path, extra_path_r, P)
			klds = pool.map(compactEstimateInformationGain,[data_t, data_b, data_l, data_r])
			KLD_estimation = np.array(klds)
			print("KLD_estimates:\t" + str(KLD_estimation))

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
	init_path = [ (2.5, 2.5) , (2.5, 2.450000000002) , (2.5, 2.400000000004) , (2.5, 2.350000000006) , (2.5, 2.300000000008) , (2.5, 2.25000000001) , (2.5, 2.200000000012) , (2.5, 2.150000000014) , (2.5, 2.100000000016) , (2.5, 2.050000000018) , (2.5, 2.00000000002) , (2.5, 1.950000000022) , (2.5, 1.9000000000239998) , (2.5, 1.850000000026) , (2.5, 1.8000000000279999) , (2.5, 1.75000000003) , (2.5, 1.700000000032) , (2.5, 1.6500000000339998) , (2.5, 1.600000000036) , (2.5, 1.5500000000379999) , (2.5, 1.50000000004) , (2.5, 1.450000000042) , (2.5, 1.4000000000439998) , (2.5, 1.350000000046) , (2.5, 1.3000000000479999) , (2.5, 1.2500000000499998) , (2.5, 1.200000000052) , (2.5, 1.1500000000539998) , (2.5, 1.1000000000559997) , (2.5, 1.0500000000579999) , (2.5, 1.0000000000599998) , (2.5, 0.9500000000619999) , (2.5, 0.9000000000639998) , (2.5, 0.8500000000659997) , (2.5, 0.8000000000679999) , (2.5, 0.7500000000699998) , (2.5, 0.7000000000719999) , (2.5, 0.6500000000739998) , (2.5, 0.6000000000759997) , (2.5, 0.5500000000779999) , (2.5, 0.5000000000799998) , (2.5, 0.4500000000819999) , (2.5, 0.40000000008399983) , (2.5, 0.35000000008599974) , (2.5, 0.30000000008799965) , (2.5, 0.25000000008999956) , (2.5, 0.20000000009199992) , (2.5, 0.15000000009399983) , (2.5, 0.10000000009599974) , (2.5, 0.05000000009799965) , (2.5, 0.0) , (2.5, 0.0) , (2.54999999999, 0.0) , (2.59999999998, 0.0) , (2.64999999997, 0.0) , (2.69999999996, 0.0) , (2.74999999995, 0.0) , (2.79999999994, 0.0) , (2.84999999993, 0.0) , (2.89999999992, 0.0) , (2.94999999991, 0.0) , (3.0, 0)]
	autonomous(corridor_1_small, init_path)

