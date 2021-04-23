import math
import math

import gdm.utils.metrics
import pandas as pd
from gdm.common import NormalPDF, LatticeScalar

"""
def comparePDFWithGroundtruth(pdf, ground_truth):
	assert isinstance(pdf, NormalPDF)
	assert isinstance(ground_truth, LatticeScalar)

	mean = pdf.getMean()
	var = pdf.getVariance()

	error = gdm.utils.metrics.getError(mean, ground_truth)
	dist = gdm.utils.metrics.getDistance(mean, ground_truth)
	rmse = gdm.utils.metrics.getRMSE(mean, ground_truth)
	md = gdm.utils.metrics.getMahalanobisDistance(pdf, ground_truth)
	nlml = gdm.utils.metrics.getNLML(pdf, ground_truth)

	max_gas = mean.toMatrix().max()
	min_gas = mean.toMatrix().min()
	avg_gas = mean.toMatrix().sum() / (mean.shape[0] * mean.shape[1])

	max_var = var.toMatrix().max()
	min_var = var.toMatrix().min()
	avg_var = var.toMatrix().sum() / (var.shape[0] * var.shape[1])

	print("Error:\t" + str(error) + "\t" +
	      "Dist:\t" + str(dist) + "\t" +
	      "RSME:\t" + str(rmse) + "\t" +
	      "MDis:\t" + str(md) + "\t" +
	      "NLML:\t" + str(nlml) + "\t" +
	      "max_gas:\t" + str(max_gas) + "\t" +
	      "min_gas:\t" + str(min_gas) + "\t" +
	      "avg_gas:\t" + str(avg_gas) + "\t" +
	      "max_var:\t" + str(max_var) + "\t" +
	      "min_var:\t" + str(min_var) + "\t" +
	      "avg_var:\t" + str(avg_var) + "\t")

	return error, dist, rmse, md, nlml
"""


def comparePDFWithGroundtruth(pdf, ground_truth, step, path_length, run_time, verbose=True):
    assert isinstance(pdf, NormalPDF)
    assert isinstance(ground_truth, LatticeScalar)

    ## COMPUTE METRICS
    mean = pdf.getMean()
    var = pdf.getVariance()
    error = gdm.utils.metrics.getError(mean, ground_truth)
    dist = gdm.utils.metrics.getDistance(mean, ground_truth)
    rmse = gdm.utils.metrics.getRMSE(mean, ground_truth)
    mdist = gdm.utils.metrics.getMahalanobisDistance(pdf, ground_truth)
    nlml = gdm.utils.metrics.getNLML(pdf, ground_truth)
    max_gas = mean.toMatrix().max()
    min_gas = mean.toMatrix().min()
    avg_gas = mean.toMatrix().sum() / (mean.shape[0] * mean.shape[1])
    max_var = var.toMatrix().max()
    min_var = var.toMatrix().min()
    avg_var = var.toMatrix().sum() / (var.shape[0] * var.shape[1])

    ## REPORT BACK
    df_line = pd.DataFrame(index=[0],
                           data={"step": step,
                                 "path_length": path_length,
                                 "run_time": run_time,
                                 "error": error,
                                 "dist": dist,
                                 "rmse": rmse,
                                 "mdist": mdist,
                                 "nlml": nlml,
                                 "max_gas": max_gas,
                                 "min_gas": min_gas,
                                 "avg_gas": avg_gas,
                                 "max_var": max_var,
                                 "min_var": min_var,
                                 "avg_var": avg_var
                                 })
    return df_line


def computePathLength(path):
    assert len(path) > 1
    l = 0
    for i in range(0, len(path) - 1):
        dl = math.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)
        l += dl
    return l
