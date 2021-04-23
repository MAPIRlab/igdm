import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def myround(x, base=5):
    return base * round(x / base)


def convertReportToPandas(file, scenario, explorer, start, run):
    path_length = "path_length\t0.0"

    ## DELETE METRICS FILE IF IT EXISTS
    metrics_file_aux = file + "_metrics_aux.csv"
    if os.path.exists(metrics_file_aux):
        os.remove(metrics_file_aux)

    ## GREP CONTENT OF REPORT AND STORE IN METRICS FILE
    with open(metrics_file_aux, 'w') as metrics:
        with open(file) as report:
            for line in report:
                if "Total path lenght:" in line:
                    path_length = line.replace("Total path lenght:\t", "")
                    # path_length = myround(float(path_length))
                    path_length = float(path_length)
                    path_length = "path_length\t" + str(path_length)
                if "Error:" in line:
                    metrics.write(line[:-1] + path_length + "\n")

    ## DELETE METRICS FILE IF IT EXISTS
    metrics_file = file + "_metrics.csv"
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    ## DELETE REDUNDANT EMPTY LINES
    with open(metrics_file_aux) as infile, open(metrics_file, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output

    ## LOAD DATA: MANY GARBAGE COLUMNS
    df = pandas.read_csv(metrics_file, sep="\t", header=None)
    del_i = [i for i in np.arange(0, df.shape[1], 2)]

    ## LAST COLUMN IS FULL OF NAN, DELETE
    # i = del_i.pop()
    # df.drop([i],axis=1, inplace=True)

    ## FOR EACH COLUMN FULL OF TEXT:
    ## Use that column as title for the next column
    ## Delete column
    ## Rename next column
    while len(del_i) != 0:
        i = del_i.pop()
        name = df[i][0].replace(':', '').lower()
        df.drop([i], axis=1, inplace=True)
        df.rename({i + 1: name}, axis='columns', inplace=True)

    ## Fill in meta-data
    df["iteration"] = [5 * i for i in range(0, df.shape[0])]
    df["scenario"] = scenario
    df["explorer"] = explorer
    df["start"] = start
    df["run"] = run
    return df


def loadFolder(folder):
    dfs = []
    for sub_folder in os.listdir(folder):
        if os.path.isdir(folder + "/" + sub_folder):
            print("Parsing " + str(sub_folder))
            run = sub_folder[-1]
            scenario = str(sub_folder).split('_')[0]
            start = str(sub_folder).split('_')[1]
            explorer = str(sub_folder).split('_')[2]
            df = convertReportToPandas(folder + "/" + sub_folder + "/report.txt", scenario=scenario, explorer=explorer,
                                       start=start, run=run)
            dfs += [df]

    big_df = pandas.concat(dfs)
    print("Storing aggregated data CSV file")
    big_df.to_csv(folder + "/raw_data.csv")
    print("Computing statistics...")
    statistics = computeStatisticsForDF(big_df)
    print("Storing processed CSV file")
    statistics.to_csv(folder + "/statistics.csv")

    return statistics


def printStats(stats):
    sns.lineplot(x="path_length", y="nlml", hue="explorer", data=stats)
    plt.show()
    sns.lineplot(x="path_length", y="rsme", hue="explorer", data=stats)
    plt.show()


# sns.lineplot(x="path_length", y="mdis", hue="explorer", data=stats)
# plt.show()
# sns.lineplot(x="iteration", y="nlml", data=stats)
# plt.show()

## =======================================================================


def convertTestResultsToDF(test_folder):
    """
    :param test_folder: folder containing subfolders for each experiment
    :return: pandas DF
    """

    dfs = []
    for experiment_folder in os.listdir(test_folder):
        if os.path.isdir(test_folder + "/" + experiment_folder):
            for file_name in os.listdir(test_folder + "/" + experiment_folder):
                file = test_folder + "/" + experiment_folder + "/" + file_name
                if os.path.isfile(file):
                    if (file[-4:] == '.csv' and file[-11:] != '_action.csv'):
                        print("Reading " + file_name)
                        if os.path.getsize(file) > 0:
                            df = pd.read_csv(file, index_col=0)

                            ## QUICK FIXES
                            ## - Rename columns
                            ## - Add missing columns
                            df.rename(columns={"stategy": "strategy", "B": "b"}, inplace=True)
                            if not 'iteration' in df:
                                df['iteration'] = 0

                            ## ADD LONG TEST NAME
                            test_name = file_name[:-4]
                            df['test_name'] = test_name

                            dfs += [df]
                        else:
                            print("Empty file!")
    print("Concatenate...")
    big_df = pd.concat(dfs, axis=0, ignore_index=True)
    print("Data loaded!")
    return big_df


def combineMultipleTestsIntoDF(parent_folder):
    dfs = []
    for test_folder in parent_folders:
        dfs += [convertTestResultsToDF(test_folder)]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def computeStatisticsFromDF(df, statistics=["path_length", "nlml", "rmse", "mdist"]):
    scenarios = df["scenario"].unique()
    strategies = df["strategy"].unique()
    starts = df["start"].unique()
    iterations = df["iteration"].unique()

    iteration = []
    min = []
    max = []
    avg = []
    med = []
    q2 = []
    q3 = []
    scenario = []
    strategy = []
    start = []
    statistic = []
    num_runs = []

    for c in scenarios:
        print("\tProcessing " + str(c))
        for s in strategies:
            print("\t\tProcessing " + str(s))
            for t in starts:
                print("\t\t\tProcessing " + str(t))
                for i in iterations:
                    for st in statistics:
                        data = df.loc[(df["strategy"] == s) & (df["start"] == t) & (df["iteration"] == i)]
                        stat = data[st]

                        min += [stat.min()]
                        max += [stat.max()]
                        avg += [stat.mean()]
                        med += [stat.median()]
                        q2 += [stat.quantile([0.25]).iat[0]]
                        q3 += [stat.quantile([0.75]).iat[0]]
                        iteration += [i]
                        scenario += [c]
                        start += [t]
                        strategy += [s]
                        statistic += [st]
                        num_runs += [len(stat)]

    assert len(strategy) == len(start) == len(iteration) == len(min) == len(max) == len(avg) == len(scenario)

    df_stats = pd.DataFrame(data={"strategy": strategy,
                                  "scenario": scenario,
                                  "start": start,
                                  "iteration": iteration,
                                  "statistic": statistic,
                                  "min": min,
                                  "q2": q2,
                                  "med": med,
                                  "q3": q3,
                                  "max": max,
                                  "avg": avg,
                                  "runs": num_runs})

    return df_stats


################################################################################
################################################################################


def printRMSEPATH():
    df = combineFolders(("/home/andy/tmp/igdm", "/home/andy/tmp/test_2"))
    df = df[df["step"] <= 1000]
    df = df.replace("igdm_default_N0.000_S1.000", "IGDM")
    df = df.replace("igdm", "IGDM")
    df = df.replace("auc", "Advance until collision")
    df = df.replace("brownian", "Brownian")
    df = df.replace("brownian_no_return", "Brownian w/o backtracking")

    sns.set_style("whitegrid", {'font.family': ['serif']})
    fig = plt.figure(num=None, figsize=(7.5, 4), dpi=100, facecolor='w', edgecolor='k')
    plt.ylim(0, 0.2)
    sns_plot = sns.lineplot(hue="strategy", x="step", y="rmse", data=df)  # , err_style="bars")
    sns_plot.get_figure().savefig("/home/andy/rmse.svg")

    return df


def printRMSENOISE():
    df = combineFolders(("/home/andy/tmp/igdm", "/home/andy/tmp/test_10"))
    df = df[df["step"] <= 1000]
    df = df.replace("igdm_default_N0.000_S1.000", "IGDM")
    df = df.replace("igdm", "IGDM")

    df = df[(df["step"] == 10) | (df["step"] == 100) | (df["step"] == 1000)]
    df = df.replace("IGDM", 0)
    df = df.replace("igdm_default_N0.050_S1.000", 0.05)
    df = df.replace("igdm_default_N0.100_S1.000", 0.10)
    df = df.replace("igdm_default_N0.150_S1.000", 0.15)
    df = df.replace("igdm_default_N0.200_S1.000", 0.20)
    df = df.replace("igdm_default_N0.250_S1.000", 0.25)
    df = df.replace("igdm_default_N0.300_S1.000", 0.30)
    df = df.replace("igdm_default_N0.350_S1.000", 0.35)
    df = df.replace("igdm_default_N0.400_S1.000", 0.40)
    df = df.replace("igdm_default_N0.450_S1.000", 0.45)
    df = df.replace("igdm_default_N0.500_S1.000", 0.50)
    df = df.replace("igdm_default_N1.000_S1.000", 1.00)
    df = df[(df["strategy"] == 0) | (df["strategy"] == 0.1) | (df["strategy"] == 0.2) | (df["strategy"] == 0.3) | (
            df["strategy"] == 0.4) | (df["strategy"] == 0.5)]
    df = df.sort_values(by=['strategy'])

    sns.set_style("whitegrid", {'font.family': ['serif']})
    fig = plt.figure(num=None, figsize=(7.5, 4), dpi=100, facecolor='w', edgecolor='k')
    plt.ylim(0, 0.5)
    sns_plot = sns.barplot(hue="strategy", x="step", y="rmse", data=df)  # , err_style="bars")
    sns_plot.get_figure().savefig("/home/andy/rmse.svg")

    return df


def printRMSESTEP():
    df = combineFolders(("/home/andy/tmp/picasso/test_02"))
    df = df[df["step"] <= 1000]
    df = df.replace("igdm_default_N0.000_S1.000", "IGDM")
    df = df.replace("igdm", "IGDM")

    # df = df[(df["step"]==10)|(df["step"]==20)|(df["step"]==100)]
    df = df.replace("IGDM", "1.00")
    df = df.replace("igdm_default_N0.000_S0.150", "0.15")
    df = df.replace("igdm_default_N0.000_S0.250", "0.25")
    df = df.replace("igdm_default_N0.000_S0.500", "0.50")

    df.loc[(df["strategy"] == "0.15"), "step"] *= 0.150
    df.loc[(df["strategy"] == "0.25"), "step"] *= 0.250
    df.loc[(df["strategy"] == "0.50"), "step"] *= 0.500

    # df = df[ (df["strategy"]==0) | (df["strategy"]==0.1) | (df["strategy"]==0.2) | (df["strategy"]==0.3) | (df["strategy"]==0.4) | (df["strategy"]==0.5) ]
    # df=df.sort_values(by=['strategy'])

    sns.set_style("whitegrid", {'font.family': ['serif']})
    fig = plt.figure(num=None, figsize=(7.5, 4), dpi=100, facecolor='w', edgecolor='k')
    plt.ylim(0, 0.3)
    sns_plot = sns.lineplot(hue="strategy", x="step", y="rmse", data=df)  # , err_style="bars")
    sns_plot.get_figure().savefig("/home/andy/rmse.svg")

    return df


def printTime():
    df = pd.DataFrame(
        {"Environment size (m²)": (10, 50, 100, 500, 1000), "Execution time (s)": (0.13, 1.5, 12.48, 259, 1865)})
    print(df)
    sns.set_style("whitegrid", {'font.family': ['serif']})
    fig = plt.figure(num=None, figsize=(7.5, 4), dpi=100, facecolor='w', edgecolor='k')
    sns_plot = sns.barplot(x="Environment size (m²)", y="Execution time (s)", data=df)  # , err_style="bars")
    sns_plot.set_yscale("log")
    sns_plot.get_figure().savefig("/home/andy/time.svg")


if __name__ == "__main__":

    import matplotlib as mpl
    sns.set(rc={'figure.figsize': (6, 3)})
    sns.set_style("whitegrid")
    mpl.rc('font', family='serif', serif='Times New Roman')

    """
    #1: coarse resolution
    df = convertTestResultsToDF("/home/andy/tmp/picasso/test_01/")
    df = df.loc[df['step'] < 210]
    df = df.loc[df['scenario'] == 3 ]
    sns.lineplot(x="step", y="rmse", hue='coarse_resolution', data=df, palette="colorblind")
    plt.show()
    """

    """
    #4: contribution of energy terms
    df = convertTestResultsToDF("/home/andy/tmp/picasso/test_04/")

    def df_strategy(df):
        if (df['igdm_weights_kld'] == 0 and df['igdm_weights_movement'] == 0 and df['igdm_weights_gas'] > 0):
            return 'Rg'
        elif (df['igdm_weights_kld'] == 0 and df['igdm_weights_movement'] > 0 and df['igdm_weights_gas'] == 0):
            return 'Rm'
        elif (df['igdm_weights_kld'] > 0 and df['igdm_weights_movement'] == 0 and df['igdm_weights_gas'] == 0):
            return 'RKLD'
        elif (df['igdm_weights_kld'] > 0 and df['igdm_weights_movement'] > 0 and df['igdm_weights_gas'] > 0):
            return 'igdm'
        else:
            return "igdm_invalid"


    df["strategy"] = df.apply(df_strategy, axis = 1)
    df = df.loc[df['strategy'] != "igdm_invalid"]
    sns.lineplot(x="step", y="rmse", hue="strategy", data=df)
    plt.show()
    """



    ## 5: Horizon size
    df = convertTestResultsToDF("/home/andy/tmp/picasso/test_05/")
    df = df.loc[df['step'] < 400]
    df = df.loc[df['scenario'] != 3]
    sns.lineplot(x="step", y="rmse", hue='igdm_H', data=df, palette="colorblind")
    plt.show()

    df = convertTestResultsToDF("/home/andy/tmp/picasso/test_05/")
    df = df.loc[df['step'] < 400]
    df = df.loc[df['scenario'] == 3]
    sns.lineplot(x="step", y="rmse", hue='igdm_H', data=df, palette="colorblind")
    plt.show()


    """
    from gdm.common import ObstacleMap

    s1h1 = ObstacleMap.fromPGM("/home/andy/tmp/path/s1h1.pgm")
    s1h2 = ObstacleMap.fromPGM("/home/andy/tmp/path/s1h2.pgm")
    s1h3 = ObstacleMap.fromPGM("/home/andy/tmp/path/s1h3.pgm")
    s1h4 = ObstacleMap.fromPGM("/home/andy/tmp/path/s1h4.pgm")
    s2h1 = ObstacleMap.fromPGM("/home/andy/tmp/path/s2h1.pgm")
    s2h2 = ObstacleMap.fromPGM("/home/andy/tmp/path/s2h2.pgm")
    s2h3 = ObstacleMap.fromPGM("/home/andy/tmp/path/s2h3.pgm")
    s2h4 = ObstacleMap.fromPGM("/home/andy/tmp/path/s2h4.pgm")

    s1h1.plot(vmax=(20 / 20), scale="lin", save="/home/andy/s1h1.png")
    s1h2.plot(vmax=(20 / 23), scale="lin", save="/home/andy/s1h2.png")
    s1h3.plot(vmax=(20 / 35), scale="lin", save="/home/andy/s1h3.png")
    s1h4.plot(vmax=(20 / 50), scale="lin", save="/home/andy/s1h4.png")
    s2h1.plot(vmax=(14 / 14), scale="lin", save="/home/andy/s2h1.png")
    s2h2.plot(vmax=(14 / 24), scale="lin", save="/home/andy/s2h2.png")
    s2h3.plot(vmax=(14 / 29), scale="lin", save="/home/andy/s2h3.png")
    s2h4.plot(vmax=(14 / 33), scale="lin", save="/home/andy/s2h4.png")
    """

print("DONE")
