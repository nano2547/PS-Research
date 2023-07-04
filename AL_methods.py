import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#one period is 70 timesteps

#adjacency list mapping out node:[neighbors]
adj_list = {1: [29, 65],
 2: [30, 66],
 3: [31, 67],
 5: [12, 26, 19, 62],
 6: [13, 27, 20, 63],
 7: [14, 28, 21, 64],
 8: [15, 22],
 9: [16, 23],
 10: [17, 24],
 12: [5, 26, 33, 19],
 13: [6, 27, 34,20],
 14: [7, 28, 35,21],
 15: [8, 29, 52],
 16: [9, 53, 30],
 17: [10, 54, 31],
 19: [26, 5, 33, 12],
 20: [27, 34, 6,13],
 21: [28, 35, 7,14],
 22: [8, 36, 40, 29],
 23: [9, 37, 41, 30],
 24: [10, 38, 42, 31],
 26: [12, 33, 5, 19],
 27: [13, 20, 34, 6],
 28: [14, 21, 35, 7],
 29: [1, 65, 15, 48, 22],
 30: [2, 66, 49, 16, 23],
 31: [3, 67, 50, 17, 24],
 33: [56, 19, 26, 12],
 34: [57, 13, 27, 20],
 35: [58, 14, 28, 21],
 36: [22],
 37: [23],
 38: [24],
 40: [36, 48,22],
 41: [37, 49, 23],
 42: [38, 50, 24],
 44: [52, 59],
 45: [53, 60],
 46: [54, 61],
 48: [40, 52, 29],
 49: [53, 30, 41],
 50: [31, 42, 54],
 52: [44, 59, 48, 15],
 53: [45, 60, 49, 16],
 54: [17, 50, 46, 61],
 56: [33, 74, 68],
 57: [34, 75, 69],
 58: [35, 76, 70],
 59: [71, 77, 44, 52],
 60: [45, 78, 72, 53],
 61: [46, 79, 73, 54],
 62: [5, 86, 80],
 63: [6, 87, 81],
 64: [7, 88, 82],
 65: [29, 1, 89, 83],
 66: [2, 30, 90, 84],
 67: [85, 91, 3, 31],
 68: [56, 74],
 69: [57, 75],
 70: [58, 76],
 71: [59, 77],
 72: [60, 78],
 73: [79, 61],
 74: [56, 68],
 75: [57, 69],
 76: [58, 70],
 77: [59, 71],
 78: [60, 72],
 79: [61, 73],
 80: [62, 86],
 81: [63, 87],
 82: [64, 88],
 83: [89, 65],
 84: [90, 66],
 85: [91, 67],
 86: [62, 80],
 87: [63, 81],
 88: [64, 82],
 89: [65, 83],
 90: [66, 84],
 91: [85, 67]}


# index_mapping = [(1, 'C_CTWE5_A'), (2, 'C_CTWE5_B'), (3, 'C_CTWE5_C'), (4, 'C_CTWE5_N'), (5, 'V_PTWE4_AN'), (6, 'V_ PTWE4_BN'), (7, 'V_ PTWE4_CN'), (8, 'C_CTWE1_A'), (9, 'C_CTWE1_B'), (10, 'C_CTWE1_C'), (11, 'C_CTWE1_N'), (12, 'V_PTWE1_AN'), (13, 'V_ PTWE1_BN'), (14, 'V_ PTWE1_CN'), (15, 'C_CTWE9_A'), (16, 'C_CTWE9_B'), (17, 'C_CTWE9_C'), (18, 'C_CTWE9_N'), (19, 'V_PTWE3_AN'), (20, 'V_ PTWE3_BN'), (21, 'V_ PTWE3_CN'), (22, 'C_CTWE3_A'), (23, 'C_ CTWE3_B'), (24, 'C_ CTWE3_C'), (25, 'C_ CTWE3_N'), (26, 'V_PTWE2_AN'), (27, 'V_ PTWE2_BN'), (28, 'V_ PTWE2_CN'), (29, 'C_CTWE7_A'), (30, 'C_CTWE7_B'), (31, 'C_CTWE7_C'), (32, 'C_CTWE7_N'), (33, 'V_PTWE5_AN'), (34, 'V_ PTWE5_BN'), (35, 'V_ PTWE5_CN'), (36, 'C_CTWE2_A'), (37, 'C_ CTWE2_B'), (38, 'C_ CTWE2_C'), (39, 'C_ CTWE2_N'), (40, 'C_CTWE4_A'), (41, 'C_ CTWE4_B'), (42, 'C_ CTWE4_C'), (43, 'C_ CTWE4_N'), (44, 'C_CTWE6_A'), (45, 'C_ CTWE6_B'), (46, 'C_ CTWE6_C'), (47, 'C_ CTWE6_N'), (48, 'C_CTWE10_A'), (49, 'C_ CTWE10_B'), (50, 'C_ CTWE10_C'), (51, 'C_ CTWE10_N'), (52, 'C_CTWE8_A'), (53, 'C_CTWE8_B'), (54, 'C_CTWE8_C'), (55, 'C_CTWE8_N'), (56, 'V_PT_DIST1_AN'), (57, 'V_PT_DIST1_BN'), (58, 'V_PT_DIST1_CN'), (59, 'C_ CT_ DIST1_A'), (60, 'C_ CT_ DIST1_B'), (61, ' C_ CT_ DIST1_C'), (62, 'V_PT_DIST2_AN'), (63, 'V_PT_DIST2_BN'), (64, 'V_PT_DIST2_CN'), (65, 'C_ CT_ DIST2_A'), (66, 'C_ CT_ DIST2_B'), (67, ' C_ CT_ DIST2_C'), (68, 'V_PT_FDR3_AN'), (69, 'V_PT_FDR3_BN'), (70, 'V_PT_FDR3_CN'), (71, 'C_CT_FDR3_A'), (72, 'C_CT_FDR3_B'), (73, 'C_CT_FDR3_C'), (74, 'V_PT_FDR1_AN'), (75, 'V_PT_FDR1_BN'), (76, 'V_PT_FDR1_CN'), (77, 'C_CT_FDR1_A'), (78, 'C_CT_FDR1_B'), (79, 'C_CT_FDR1_C'), (80, 'V_PT_FDR32_AN'), (81, 'V_PT_FDR32_BN'), (82, 'V_PT_FDR32_CN'), (83, 'C_CT_FDR32_A'), (84, 'C_CT_FDR32_B'), (85, 'C_CT_FDR32_C'), (86, 'V_PT_FDR12_AN'), (87, 'V_PT_FDR12_BN'), (88, 'V_PT_FDR12_CN'), (89, 'C_CT_FDR12_A'), (90, 'C_CT_FDR12_B'), (91, 'C_CT_FDR12_C')]

emptyr = [4,11,18,25,32,39,43,47,51,55] #nodes without connections to anything

order_attack = ['CT', 'Vphase', 'Cphase', 'GPS', 'GPS', 'CT', 'PT']

#dictionary containing attack points with format number:(start time, end time, type, name)
attack_dict = { 52:(1.10,2.10,'CT','C_CTWE8_A'),   53:(1.10,2.10,'CT','C_CTWE8_B'), 54:(1.10,2.10,'CT','C_CTWE8_C'), 55:(1.10, 2.10,'CT','C_CTWE8_N'),
#                56:(2.30,3.00,'PT','V_PT_DIST1_AN'), 57:(2.30,3.00,'PT','V_PT_DIST1_BN'), 58:(2.30,3.00,'PT','V_PT_DIST1_CN'),
               5:(5.00,5.90,'Vphase','V_PTWE4_AN'), 66:(6.60,7.80,'Cphase','C_CT_DIST2_B'), 68:(9.00,10.10,'GPS','V_PT_FDR3_AN'), 69:(9.00,10.10,'GPS','V_PT_FDR3_BN'), 70:(9.00,10.10,'GPS','V_PT_FDR3_CN'), 71:(9.00,10.10,'GPS','C_CT_FDR3_A'), 72:(9.00,10.10,'GPS','C_CT_FDR3_B'), 73:(9.00,10.10,'GPS','C_CT_FDR3_C'), 36:(10.50,11.70,'GPS','C_CTWE2_A'), 37:(10.50,11.70,'GPS','C_CTWE2_B'), 38:(10.50,11.70,'GPS','C_CTWE2_C'),39:(10.50,11.70,'GPS','C_CTWE2_N'), 77:(16.20, 17.00, 'CT', 'C_CT_FDR1_A'), 78:(16.20, 17.00, 'CT', 'C_CT_FDR1_B'), 79:(16.20, 17.00, 'CT', 'C_CT_FDR1_C'), 86:(17.40, 18.30, 'PT','V_PT_FDR12_AN'), 87:(17.40, 18.30, 'PT','V_PT_FDR12_BN'), 88:(17.40, 18.30, 'PT','V_PT_FDR12_CN')}


#dictionary containing the only attack without a compromised unit
attack_wo_comp_dict = {'Load_decrease':(10.20, 10.20, 'Bus_3DB211', 'all')}


#dictionary contatining faults labeled    type:(start time, end time, location, effects?)
fault_dict = {'Downed_conductor':[3.25,3.50,'Bus_DBU3_A','all'], 'LG':[8.50,8.57,'Bus_FDR11LOAD_A','none'], 'LL':[12.00,12.08,'Bus_3DB12','none']}


#dictionary containing normal events labeled    time:(type, power, location)
normal_dict = { 6.10:('L+','1500kw+10kvar', 'Bus_3DB211'), 8.00:('L+','4500kw+1000kvar','Bus_TRAIN1'), 10.20:('L-','1500kw+10kvar', 'Bus_3DB211'), 13.50:('L+','6500kw+2000kvar', 'Bus_FDR12LOAD'), 14.00:('L+','1500kw+10kvar','Bus_2DB112'), 14.40:('L-','4500kw+1000kvar', 'Bus_TRAIN1'), 18.80:('L-','6500kw+2000kvar','FDR12LOAD'), 19.50:('L-','1500kw+10kvar','Bus_2DB112'), 19.80:('L+','1500kw+10kvar','Bus_3DB211')}


things = list(adj_list.keys())
things.sort()

all_attack_times = set()
for k, values in attack_dict.items():
    all_attack_times.add((values[0], values[1]))

    
attacks = {}
for sensor, (start, end, attack_type, sensor_name) in attack_dict.items():
    if sensor in emptyr:
        continue
    neighbour_list = [sensor] + adj_list[sensor]
    try:
        attacks[start].append(neighbour_list)
    except:
        attacks[start] = [neighbour_list]

attack_start_times = sorted(attacks.keys())

attacks_shifted_index = {}
for att, sensor_list in attacks.items():
    l = []
    for nbh in sensor_list:
        n = []
        for sensor in nbh:
            n.append(things.index(sensor))
        l.append(n)
    attacks_shifted_index[att] = l


# all_attack_indices = {(4080, 8879),
#  (22800, 27119),
#  (30480, 36239),
#  (42000, 47279),
#  (49200, 54959),
#  (76560, 80399),
#  (82320, 86639)}

# for start, end in all_attack_times:
#     attack_indices = df_test.index[(df_test['Time'] >= start * 1e6) & (df_test['Time'] <= end * 1e6)] - 1201
#     all_attack_indices.add((attack_indices[0], attack_indices[-1]))

# all_attack_indice = sorted(list(all_attack_indices))

# attacks_in = {}
# co = 0
# for key, value in attacks.items():
#   attacks_in[sorted(all_attack_indice)[co][0]] = value
#   co+=1
    
hops = {}
types = 1.1
for key, value in attack_dict.items():
    if types == value[0]:
        if value[1] not in hops:
            hops[value[1]]= [key]
        else:
            hops[value[1]] += [key]
    else:
        types = value[0]
        hops[value[1]]= [key]
    
stu = []
for i in hops:
    stu += [i]

def preprocess(df, time=0.25e6):
    return df[df['Time'] > time].iloc[:, 1:]


def correlation_pair(weights, dataframe, var1_index, var2_index, window, diff=0, pct_change=False):

    corrs = dataframe.iloc[:, var1_index].rolling(window).corr(dataframe.iloc[:, var2_index]*weights)[window-1:]
    corrs_change = np.diff(corrs, n=diff)
    if pct_change:
        corrs_change = np.divide(corrs_change, corrs[:-1])
        
    return corrs_change


def correlation_pair_plot(weights, dataframe, var1_index, var2_index, window, freq=True,
                     attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    
    cols = dataframe.columns

    corrs = correlation_pair(weights, dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change)

    plt.title(f"Sample Correlation against Time, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
    plt.plot(corrs)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    plt.show()

    if freq:
        plt.title(f"Sample Correlation Frequency, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
        plt.hist(corrs, bins=500)
        plt.show()

    return corrs

def correlation_multi(weight, dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):
    #weight being a list containing weights that corespond to each of the var2_indices
    corrs = []
    for ind, var2_index in enumerate(var2_indices):
        corrs.append(correlation_pair(weight[ind],dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(corrs, axis=0)


def correlation_multi_plot(weight,dataframe, var1_index, var2_indices, window, freq=True,
                     attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    
    cols = dataframe.columns
    corrs = correlation_multi(weight, dataframe, var1_index, var2_indices, window, diff=diff, pct_change=pct_change)

    plt.title(f"Sample Correlation against Time, Window Length {window}, Variables: {cols[var1_index]} with " + \
              f"{[cols[var2_index] for var2_index in var2_indices]}")
    plt.plot(corrs)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    plt.show()

    if freq:
        plt.title(f"Sample Correlation Frequency, Window Length {window}, Variables: {cols[var1_index]} with " + \
              f"{[cols[var2_index] for var2_index in var2_indices]}")
        plt.hist(corrs, bins=500)
        plt.show()

    return corrs


def var_pair(dataframe, var1_index, var2_index, window, diff=0, pct_change=False):
    
    var_x = dataframe.iloc[:, var1_index].rolling(window).var()[window-1:]
    var_y = dataframe.iloc[:, var2_index].rolling(window).var()[window-1:]
    var_ratio = np.divide(var_x, var_y)
    var_ratio_change = np.diff(var_ratio, n=diff)
    
    if pct_change:
        var_ratio_change = np.divide(var_ratio_change, var_ratio[:-diff])
        
    return var_ratio_change


def var_multi(dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):

    var = []
    for var2_index in var2_indices:
        var.append(var_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(var, axis=0)


def covariance_pair(weights, dataframe, var1_index, var2_index, window, diff=0, pct_change=False):
    
    covs = dataframe.iloc[:, var1_index].rolling(window).cov(dataframe.iloc[:, var2_index]*weight)[window-1:]
    covs_change = np.diff(covs, n=diff)
    if pct_change:
        covs_change = np.divide(covs_change, covs[:-1])
        
    return covs_change


def covariance_pair_plot(weights, dataframe, var1_index, var2_index, window, freq=True,
                     attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):

    covs = covariance_pair(weights, dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change)

    plt.title(f"Sample Covariance against Time, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
    plt.plot(covs)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    plt.show()

    if freq:
        plt.title(f"Sample Covariance Frequency, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
        plt.hist(covs, bins=500)
        plt.show()

    return covs


def covariance_multi(weight, dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):

    covs = []
    for ind, var2_index in enumerate(var2_indices):
        covs.append(covariance_pair(weight[ind], dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(covs, axis=0)


def covariance_multi_plot(weight, dataframe, var1_index, var2_indices, window, freq=True,
                     attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):

    covs = covariance_multi(weight, dataframe, var1_index, var2_indices, window, diff=diff, pct_change=pct_change)

    plt.title(f"Sample Covariance against Time, Window Length {window}, Variables: {cols[var1_index]} with " + \
              f"{[cols[var2_index] for var2_index in var2_indices]}")

    plt.plot(covs)


    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    plt.show()

    if freq:
        plt.title(f"Sample Covariance Frequency, Window Length {window}, Variables: {cols[var1_index]} with " + \
              f"{[cols[var2_index] for var2_index in var2_indices]}")
        plt.hist(covs, bins=500)
        plt.show()

    return covs



def stat_diff(weights, dataframe, var1_index, var2_indices, window, stat_fn=covariance_pair, fn="prod", diff=1, pct_change=False):

    stat = []
    for ind, var2_index in enumerate(var2_indices):
        stat.append(stat_fn(weights[ind], dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    if fn == "prod":
        return np.prod(np.abs(stat), axis=0)
    elif fn == "sum":
        return np.sum(np.abs(stat), axis=0)
    else:
        return fn(np.abs(stat), axis=0)

      
    
    
def get_labels(y, window):
    '''
    get cumlulative statistic for presence of anomaly
    if window has anomaly, label the whole window as anomalous
    y: array of presence of anomalies across time
    window: window size of statistic + roll sum window
    '''

    return (np.lib.stride_tricks.sliding_window_view(y, window) == 1).any(axis=0).squeeze()


# def steps(x):
#     return x.iloc[-1] - x.iloc[0]


def roll_sum(stat, window, axis=1):
    '''
    we calculate the rolling sum of the differences within a window

    Suppose we have something that is very different the rolling sum will change.

    stat: array of statistical property (eg. covariance) from training
    window: window for differences (different from window for calculating statistical property)
    axis: axis to perform difference on (time axis usually along axis 1)
    '''

    # calculate difference btw each time step
    diff = np.abs(np.diff(stat, n=1))

    # calculate rolling sum
    csum = np.cumsum(diff, dtype=float)
    xsum = csum[window:] - csum[:-window]
    return np.abs(xsum)


def get_empirical_diff_threshold(stat, window, fn, axis=1, quantile=0.95, sensor_names=[], cols=[]):
    '''
    we calculate the rolling sum of the differences within a window

    stat: array of statistical property (eg. covariance) from training
    window: window for differences (different from window for calculating statistical property)
    quantile: confidence level
    axis: axis to perform difference on
        (time axis usually along axis 1 for 2D array. for ID array, axis=-1)
    sensor_names: list of sensor names. if empty, don't plot
    '''

    # calculate difference btw each time step
    if fn is not None:
        test_stat = fn(stat, window, axis=axis)
    else:
        test_stat = stat

    # plot the histogram of the frequency of rolling sum (or whatever function)
    # --> can be commented out
    if len(sensor_names) > 0:
        fig, ax = plt.subplots(len(sensor_names), 1)
        #test_statdf = pd.DataFrame(test_stat)
        #test_statdf.plot.kde()
        if len(sensor_names) == 1:
            ax.hist(test_stat, bins=500)
            ax.set_title(f"Rolling Sum of Difference Frequency of {cols[sensor_names[0]]} {sensor_names[0]}")
        else:
            for i, n in enumerate(sensor_names):
                ax[0].hist(test_stat, bins=500)
                ax[0].set_title(f"Rolling Sum of Difference Frequency of {s}")
        plt.show()
    #########################################################

    # return the threshold
    # may need to make this 2-sided (for negative and positive)
    return np.quantile(test_stat, quantile)


def get_threshold(weight, data, stat_fn, fn, cols, window=350, w=1, adj_list=adj_list, quantile=1, axis=1, stat_list=None, plot=False):
    
    thresholds = []
    
    have_stat_list = True

    if stat_list is None:
        stat_list = []
        have_stat_list = False

    for i, (var1_index, var2_indices) in enumerate(adj_list.items()):

        var1_index -= 1
        var2_indices = np.array(var2_indices) - 1
        
        if have_stat_list:
            statistic = stat_list[i]
        else:
            statistic = stat_fn(weight[var1_index+1], data, var1_index, var2_indices, window)
            
        if plot:
            sensor_names = [var1_index]
        else:
            sensor_names = []
        
        thresholds.append(
            get_empirical_diff_threshold(stat=statistic, window=w, fn=fn, axis=-1, quantile=quantile,
                                         sensor_names=sensor_names, cols=cols))
        
        if not have_stat_list:
            stat_list.append(statistic)

        print("******************************************************************************")

    return thresholds, stat_list


def get_threshold_stat_only(weights, data, stat_fn, cols, window=350, w=1, adj_list=adj_list, quantile=1, axis=1, stat_list=None, plot=False, collate_fn="prod"):
    '''
    use for updated statistic: product of differences rather than differences of product
    '''
    
    thresholds = []
    
    have_stat_list = True

    if stat_list is None:
        stat_list = []
        have_stat_list = False

    for i, (var1_index, var2_indices) in enumerate(adj_list.items()):

        var1_index -= 1
        var2_indices = np.array(var2_indices) - 1
        
        if have_stat_list:
            statistic = stat_list[i]
        else:
            statistic = stat_diff(weights[var1_index+1], data, var1_index, var2_indices, window, stat_fn, fn=collate_fn)
            
        if plot:
            sensor_names = [var1_index]
        else:
            sensor_names = []
        
        thresholds.append(
            get_empirical_diff_threshold(stat=statistic, window=w, fn=None, axis=-1, quantile=quantile,
                                         sensor_names=sensor_names, cols=cols))
        
        if not have_stat_list:
            stat_list.append(statistic)

        print("******************************************************************************")

    return thresholds, stat_list

# For Test Data

def detect(stat, window, threshold, multi=False):
    '''
    we use the critical regions (ie. thresholds) obtained from the previous function
    to detect local violations (anomaly detection on the local level)
    stat: array of statistical property from test data
    '''
    violation = (roll_sum(stat, window) > threshold)

    if not multi:
        violation = (violation > 0).any(axis=0).squeeze()

    return violation


def get_metrics(y_true, y_pred):
    '''
    get detection metrics
    expand this to obtain EDD and ARL
    '''

    labels = get_labels(y_true)

    # can get more metrics here
    precision, recall, f1, support = precision_recall_fscore_support(labels, y_pred)

    return precision, recall, f1


def evaluate(y_true, stat, window, threshold, multi=False):

    violation = detect(stat, window, threshold, multi=multi)
    metrics = get_metrics(y_true, violation)

    return metrics


# def get_scores(y_true, y_pred):

#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)

#     return accuracy, precision, recall, f1


# def average_run_length(predictions, anomaly_label=1):
#     run_lengths = []
#     current_run_length = 0

#     for index, prediction in enumerate(predictions):
#         if prediction == anomaly_label[index]:
#             if current_run_length > 0:
#                 run_lengths.append(current_run_length)
#                 current_run_length = 0
#         else:
#             current_run_length += 1

#     if len(run_lengths) == 0:
#         return None

#     return np.mean(run_lengths)



def persistency_check(attack_window, persistency):
    # check for persistency
    if len(attack_window) < persistency:
        return None
    if persistency == 1:
        return attack_window[0]
    windows = np.lib.stride_tricks.sliding_window_view(attack_window, persistency)
    # check if timesteps are consecutive
#     print(windows.shape)
#     print(np.diff(windows, axis=0).shape)
    diff_windows = (np.diff(windows, axis=1) == 1).all(axis=1)#.squeeze()
#     print(diff_windows.shape)
#     attacks = np.argmax(diff_windows)

    # indices of persistent detection
    attacks_ = diff_windows.nonzero()[0]
#     print(attacks)

    if len(attacks_) == 0:
        return None

    detection = windows[attacks_[0]]
    return detection[-1]


# def localisation_delay(start_of_attack, end_of_attack, attack_pred, window, w, persistency):

#     start = start_of_attack - window - w + 1
#     end = end_of_attack + window + w - 1

#     # get all indices that are after the start of attack
#     idx = np.logical_and((attack_pred >= start), (attack_pred <= end))
#     attack_window = attack_pred[idx]

#     # check for persistency
#     return persistency_check(attack_window, persistency)


def detection_delay(detection, attack_times, window, w, persistency=1):
    detection_idx = []
    attack_pred = detection.nonzero()[0]
#     if not localise:
#         detection = violations.any(axis=1)
#     else:
#         detection = attack_times.squeeze()
    for start, end in attack_times:
#         print(start, end)
        start_w = start - window - w + 1
        end_w = end + window + w - 1
        # get all indices that are after the start of attack
        indices = np.logical_and((attack_pred >= start_w), (attack_pred <= end_w))
#         print(idx.sum())
#         print(attack_pred.shape())
        attack_window = attack_pred[indices]
#         print(attack_window)

        # check for persistency
        idx = persistency_check(attack_window, persistency)
        if idx is not None:
            idx -= start_w
        detection_idx.append(idx)
    return detection_idx


def localisation_delay(detection, attack_times, window, w, persistency=1):
    detection_idx = []
    attack_pred = detection.nonzero()
#     if not localise:
#         detection = violations.any(axis=1)
#     else:
#         detection = attack_times.squeeze()
    for i, (start, end) in enumerate(attack_times):
        localisation_idx = []
#         print(start, end)
        start_w = start - window - w + 1
        end_w = end + window + w - 1
        # get all indices that are after the start of attack
        indices = np.logical_and((attack_pred[1] >= start_w), (attack_pred[1] <= end_w))
#         print(idx.sum())
#         print(attack_pred.shape())
        attack_window = attack_pred[1][indices]
        attack_sensors = attack_pred[0][indices]

        sensor_list = attacks_shifted_index[attack_start_times[i]]

        for sensor_nbd in sensor_list:

            index = np.inf

            for sensor in sensor_nbd:
                sensor_indices = (attack_sensors == sensor)
                sensor_attack_window = np.sort(attack_window[sensor_indices])

                # check for persistency
                idx = persistency_check(sensor_attack_window, persistency)    
                if idx is not None:
                    idx -= start_w
                    index = min(idx, index)

    #         sensor_nbd_indices = np.isin(attack_sensors, sensor_nbd)
    #         sensor_nbd_attack_window = attack_window[sensor_nbd_indices]
    #         # sorted based on sensor first, then indices

    #         # check for persistency
    #         idx = persistency_check(sensor_nbd_attack_window, persistency)
    #         print(idx)
    #         if idx is not None:
    #             idx -= start_w
            if index == np.inf:
                index = None
            localisation_idx.append(index)
        detection_idx.append(localisation_idx)
    return detection_idx


def persistency_check_fpr(attack_window, persistency):
    # check for persistency
    if len(attack_window) < persistency:
        return 0
    if persistency == 1:
        return len(attack_window)
    windows = np.lib.stride_tricks.sliding_window_view(attack_window, persistency)
    # check if timesteps are consecutive
#     print(windows.shape)
#     print(np.diff(windows, axis=0).shape)
#     print((np.diff(windows, axis=0) == 1).all(axis=1).shape)
    return (np.diff(windows, axis=1) == 1).all(axis=1).sum()
#     print(diff_windows.shape)
#     attacks = np.argmax(diff_windows)

    # indices of persistent detection
#     attacks = diff_windows.nonzero()[0]
#     print(attacks)

#     if len(attacks) == 0:
#         return None

#     detection = windows[attacks[0]]
#     return detection[-1]


def FPR(violations, attack_times, window, w, total_normal, persistency):
    
    total_normal -= persistency 

    attack_pred = (violations)

    shift = window + w - 1

    # go from back to prevent index issue when removing
    attack_times = sorted(list(attack_times), reverse=True)

    for start, end in attack_times:
        start_w = start - shift
        end_w = end + shift
        attack_indices = np.arange(start_w, end_w+1)
#         print("attack_pred:", attack_pred.shape)
#         print("attack_indices:", attack_indices.shape)
        attack_pred = np.delete(attack_pred, attack_indices)

    attack_num = persistency_check_fpr(attack_pred.nonzero()[0], persistency)

    return attack_num / total_normal


def false_localization(attack_list ,hop):
    FL = 0
    for attacked_sensor in attack_list:
        if attacked_sensor in hop:
            continue
        else:
            FL += 1
    FLR = FL/len(hop)
    return FLR
