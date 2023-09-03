import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections

# one period is 70 timesteps
# True with other methods eg. RCD, eigenvalues
remove_neutral = False

# choose between A B C topology
topology = "A"

if topology == "A":
    # adjacency list mapping out node:[neighbors]
    adj_list = {112: [72, 61, 90],
 113: [73, 62, 91],
 114: [74, 63, 92],
 115: [75, 64, 93],
 119: [76, 57, 97],
 120: [77, 58, 98],
 121: [78, 59, 99],
 122: [79, 60, 100],
 105: [65, 50],
 106: [66, 51],
 107: [67, 52],
 108: [68, 53],
 76: [119, 57, 50, 97],
 77: [120, 58, 51, 98],
 78: [121, 59, 52, 99],
 79: [122, 60, 53, 100],
 72: [112, 65, 61, 90],
 73: [113, 66, 62, 91],
 74: [114, 67, 63, 92],
 75: [115, 68, 64, 93],
 65: [105, 72, 50],
 66: [106, 73, 51],
 67: [107, 74, 52],
 68: [108, 75, 53],
 57: [119, 76, 61, 97],
 58: [120, 77, 62, 98],
 59: [121, 78, 63, 99],
 60: [122, 79, 64, 100],
 50: [105, 76, 65],
 51: [106, 77, 66],
 52: [107, 78, 67],
 53: [108, 79, 68],
 61: [112, 72, 57, 90],
 62: [113, 73, 58, 91],
 63: [114, 74, 59, 92],
 64: [115, 75, 60, 93],
 97: [119, 76, 57, 101],
 98: [120, 77, 58, 102],
 99: [121, 78, 59, 103],
 100: [122, 79, 60, 104],
 90: [112, 72, 61, 83],
 91: [113, 73, 62, 84],
 92: [114, 74, 63, 85],
 93: [115, 75, 64, 86],
 101: [97, 1, 8, 15, 22],
 102: [98, 2, 9, 16, 23],
 103: [99, 3, 10, 17, 24],
 104: [100, 4, 11, 18, 25],
 83: [90],
 84: [91],
 85: [92],
 86: [93],
 1: [101],
 2: [102],
 3: [103],
 4: [104],
 8: [101, 29, 36],
 9: [102, 30, 37],
 10: [103, 31, 38],
 11: [104, 32, 39],
 29: [8, 36],
 30: [9, 37],
 31: [10, 38],
 32: [11, 39],
 36: [8, 29],
 37: [9, 30],
 38: [10, 31],
 39: [11, 32],
 15: [101, 43],
 16: [102, 44],
 17: [103, 45],
 18: [104, 46],
 43: [15],
 44: [16],
 45: [17],
 46: [18],
 22: [101],
 23: [102],
 24: [103],
 25: [104],
# 116: [],
# 117: [],
# 118: [],
# 123: [],
# 124: [],
# 125: [],
# 109: [],
# 110: [],
# 111: [],
 80: [69, 54, 94, 87],
 81: [70, 55, 95, 89],
 82: [71, 56, 96, 88],
 69: [80, 54, 94, 87],
 70: [81, 55, 95, 89],
 71: [82, 56, 96, 88],
 54: [80, 69, 94, 87],
 55: [81, 70, 95, 89],
 56: [82, 71, 96, 88],
 94: [80, 69, 54, 5, 12, 19, 26],
 95: [81, 70, 55, 6, 13, 20, 27],
 96: [82, 71, 56, 7, 14, 21, 28],
 87: [80, 69, 54],
 89: [81, 70, 55],
 88: [82, 71, 56],
 5: [94],
 6: [95],
 7: [96],
 12: [94, 33],
 13: [95, 34],
 14: [96, 35],
 33: [12, 40],
 34: [13, 41],
 35: [14, 42],
 40: [33],
 41: [34],
 42: [35],
 19: [94, 47],
 20: [95, 48],
 21: [96, 49],
 47: [19],
 48: [20],
 49: [21],
 26: [94],
 27: [95],
 28: [96]}
elif topology == "B":
    adj_list = {112: [72, 61, 90],
 113: [73, 62, 91],
 114: [74, 63, 92],
 115: [75, 64, 93],
 119: [76, 57, 97],
 120: [77, 58, 98],
 121: [78, 59, 99],
 122: [79, 60, 100],
 105: [65, 50],
 106: [66, 51],
 107: [67, 52],
 108: [68, 53],
 76: [119, 57, 50, 97],
 77: [120, 58, 51, 98],
 78: [121, 59, 52, 99],
 79: [122, 60, 53, 100],
 72: [112, 65, 61, 90],
 73: [113, 66, 62, 91],
 74: [114, 67, 63, 92],
 75: [115, 68, 64, 93],
 65: [105, 72, 50],
 66: [106, 73, 51],
 67: [107, 74, 52],
 68: [108, 75, 53],
 57: [119, 76, 61, 97],
 58: [120, 77, 62, 98],
 59: [121, 78, 63, 99],
 60: [122, 79, 64, 100],
 50: [105, 76, 65],
 51: [106, 77, 66],
 52: [107, 78, 67],
 53: [108, 79, 68],
 61: [112, 72, 57, 90],
 62: [113, 73, 58, 91],
 63: [114, 74, 59, 92],
 64: [115, 75, 60, 93],
 97: [119, 76, 57, 101],
 98: [120, 77, 58, 102],
 99: [121, 78, 59, 103],
 100: [122, 79, 60, 104],
 90: [112, 72, 61, 83],
 91: [113, 73, 62, 84],
 92: [114, 74, 63, 85],
 93: [115, 75, 64, 86],
 101: [97, 1, 8, 15, 22],
 102: [98, 2, 9, 16, 23],
 103: [99, 3, 10, 17, 24],
 104: [100, 4, 11, 18, 25],
 83: [90],
 84: [91],
 85: [92],
 86: [93],
 1: [101, 36],
 2: [102, 37],
 3: [103, 38],
 4: [104, 39],
 8: [101],
 9: [102],
 10: [103],
 11: [104],
 29: [],
 30: [],
 31: [],
 32: [],
 36: [1],
 37: [2],
 38: [3],
 39: [4],
 15: [101, 43],
 16: [102, 44],
 17: [103, 45],
 18: [104, 46],
 43: [15],
 44: [16],
 45: [17],
 46: [18],
 22: [101],
 23: [102],
 24: [103],
 25: [104],
 116: [],
 117: [],
 118: [],
 123: [],
 124: [],
 125: [],
 109: [],
 110: [],
 111: [],
 80: [69, 54, 94, 87],
 81: [70, 55, 95, 88],
 82: [71, 56, 96, 89],
 69: [80, 54, 94, 87],
 70: [81, 55, 95, 88],
 71: [82, 56, 96, 89],
 54: [80, 69, 94, 87],
 55: [81, 70, 95, 88],
 56: [82, 71, 96, 89],
 94: [80, 69, 54, 5, 12, 19, 26],
 95: [81, 70, 55, 6, 13, 20, 27],
 96: [82, 71, 56, 7, 14, 21, 28],
 87: [80, 69, 54],
 88: [81, 70, 55],
 89: [82, 71, 56],
 5: [94, 40],
 6: [95, 41],
 7: [96, 42],
 12: [94, 33],
 13: [95, 34],
 14: [96, 35],
 33: [12],
 34: [13],
 35: [14],
 40: [5],
 41: [6],
 42: [7],
 19: [94, 47],
 20: [95, 48],
 21: [96, 49],
 47: [19],
 48: [20],
 49: [21],
 26: [94],
 27: [95],
 28: [96]}

elif topology == "C":
    {112: [72, 61, 90],
 113: [73, 62, 91],
 114: [74, 63, 92],
 115: [75, 64, 93],
 119: [76, 57, 97],
 120: [77, 58, 98],
 121: [78, 59, 99],
 122: [79, 60, 100],
 105: [65, 50],
 106: [66, 51],
 107: [67, 52],
 108: [68, 53],
 76: [119, 57, 50, 97],
 77: [120, 58, 51, 98],
 78: [121, 59, 52, 99],
 79: [122, 60, 53, 100],
 72: [112, 65, 61, 90],
 73: [113, 66, 62, 91],
 74: [114, 67, 63, 92],
 75: [115, 68, 64, 93],
 65: [105, 72, 50],
 66: [106, 73, 51],
 67: [107, 74, 52],
 68: [108, 75, 53],
 57: [119, 76, 61, 97],
 58: [120, 77, 62, 98],
 59: [121, 78, 63, 99],
 60: [122, 79, 64, 100],
 50: [105, 76, 65],
 51: [106, 77, 66],
 52: [107, 78, 67],
 53: [108, 79, 68],
 61: [112, 72, 57, 90],
 62: [113, 73, 58, 91],
 63: [114, 74, 59, 92],
 64: [115, 75, 60, 93],
 97: [119, 76, 57, 101],
 98: [120, 77, 58, 102],
 99: [121, 78, 59, 103],
 100: [122, 79, 60, 104],
 90: [112, 72, 61, 83],
 91: [113, 73, 62, 84],
 92: [114, 74, 63, 85],
 93: [115, 75, 64, 86],
 101: [97, 1, 8, 15, 22],
 102: [98, 2, 9, 16, 23],
 103: [99, 3, 10, 17, 24],
 104: [100, 4, 11, 18, 25],
 83: [90],
 84: [91],
 85: [92],
 86: [93],
 1: [101, 36, 18],
 2: [102, 37, 43],
 3: [103, 38, 44],
 4: [104, 39, 45],
 8: [101],
 9: [102],
 10: [103],
 11: [104],
 29: [],
 30: [],
 31: [],
 32: [],
 36: [1, 43],
 37: [2, 44],
 38: [3, 45],
 39: [4, 46],
 15: [],
 16: [],
 17: [],
 18: [],
 43: [1, 36],
 44: [2, 37],
 45: [3, 38],
 46: [4, 39],
 22: [101],
 23: [102],
 24: [103],
 25: [104],
 116: [],
 117: [],
 118: [],
 123: [],
 124: [],
 125: [],
 109: [],
 110: [],
 111: [],
 80: [69, 54, 94, 87],
 81: [70, 55, 95, 88],
 82: [71, 56, 96, 89],
 69: [80, 54, 94, 87],
 70: [81, 55, 95, 88],
 71: [82, 56, 96, 89],
 54: [80, 69, 94, 87],
 55: [81, 70, 95, 88],
 56: [82, 71, 96, 89],
 94: [80, 69, 54, 5, 12, 19, 26],
 95: [81, 70, 55, 6, 13, 20, 27],
 96: [82, 71, 56, 7, 14, 21, 28],
 87: [80, 69, 54],
 88: [81, 70, 55],
 89: [82, 71, 56],
 5: [94, 40],
 6: [95, 41],
 7: [96, 42],
 12: [94, 33],
 13: [95, 34],
 14: [96, 35],
 33: [12],
 34: [13],
 35: [14],
 40: [5, 47],
 41: [6, 48],
 42: [7, 49],
 19: [94],
 20: [95],
 21: [96],
 47: [33],
 48: [34],
 49: [35],
 26: [94],
 27: [95],
 28: [96]}


new_adj_list = {}
for sensors, neigh in adj_list.items():
    if sensors in [4,11,18,25,32,39,46,53,60,64,68,75,79,86,93,100,104,108,115,122]:
        continue
    else:
        new_nei = []
        for i in neigh:
            if i in [4,11,18,25,32,39,46,53,60,64,68,75,79,86,93,100,104,108,115,122]:
                continue
            else:
                new_nei += [i]
        new_adj_list[sensors] = new_nei


adj_list = new_adj_list      
            
adj_list = collections.OrderedDict(sorted(adj_list.items()))

# index_mapping = [(1, 'C_CTWE5_A'), (2, 'C_CTWE5_B'), (3, 'C_CTWE5_C'), (4, 'C_CTWE5_N'), (5, 'V_PTWE4_AN'), (6, 'V_ PTWE4_BN'), (7, 'V_ PTWE4_CN'), (8, 'C_CTWE1_A'), (9, 'C_CTWE1_B'), (10, 'C_CTWE1_C'), (11, 'C_CTWE1_N'), (12, 'V_PTWE1_AN'), (13, 'V_ PTWE1_BN'), (14, 'V_ PTWE1_CN'), (15, 'C_CTWE9_A'), (16, 'C_CTWE9_B'), (17, 'C_CTWE9_C'), (18, 'C_CTWE9_N'), (19, 'V_PTWE3_AN'), (20, 'V_ PTWE3_BN'), (21, 'V_ PTWE3_CN'), (22, 'C_CTWE3_A'), (23, 'C_ CTWE3_B'), (24, 'C_ CTWE3_C'), (25, 'C_ CTWE3_N'), (26, 'V_PTWE2_AN'), (27, 'V_ PTWE2_BN'), (28, 'V_ PTWE2_CN'), (29, 'C_CTWE7_A'), (30, 'C_CTWE7_B'), (31, 'C_CTWE7_C'), (32, 'C_CTWE7_N'), (33, 'V_PTWE5_AN'), (34, 'V_ PTWE5_BN'), (35, 'V_ PTWE5_CN'), (36, 'C_CTWE2_A'), (37, 'C_ CTWE2_B'), (38, 'C_ CTWE2_C'), (39, 'C_ CTWE2_N'), (40, 'C_CTWE4_A'), (41, 'C_ CTWE4_B'), (42, 'C_ CTWE4_C'), (43, 'C_ CTWE4_N'), (44, 'C_CTWE6_A'), (45, 'C_ CTWE6_B'), (46, 'C_ CTWE6_C'), (47, 'C_ CTWE6_N'), (48, 'C_CTWE10_A'), (49, 'C_ CTWE10_B'), (50, 'C_ CTWE10_C'), (51, 'C_ CTWE10_N'), (52, 'C_CTWE8_A'), (53, 'C_CTWE8_B'), (54, 'C_CTWE8_C'), (55, 'C_CTWE8_N'), (56, 'V_PT_DIST1_AN'), (57, 'V_PT_DIST1_BN'), (58, 'V_PT_DIST1_CN'), (59, 'C_ CT_ DIST1_A'), (60, 'C_ CT_ DIST1_B'), (61, ' C_ CT_ DIST1_C'), (62, 'V_PT_DIST2_AN'), (63, 'V_PT_DIST2_BN'), (64, 'V_PT_DIST2_CN'), (65, 'C_ CT_ DIST2_A'), (66, 'C_ CT_ DIST2_B'), (67, ' C_ CT_ DIST2_C'), (68, 'V_PT_FDR3_AN'), (69, 'V_PT_FDR3_BN'), (70, 'V_PT_FDR3_CN'), (71, 'C_CT_FDR3_A'), (72, 'C_CT_FDR3_B'), (73, 'C_CT_FDR3_C'), (74, 'V_PT_FDR1_AN'), (75, 'V_PT_FDR1_BN'), (76, 'V_PT_FDR1_CN'), (77, 'C_CT_FDR1_A'), (78, 'C_CT_FDR1_B'), (79, 'C_CT_FDR1_C'), (80, 'V_PT_FDR32_AN'), (81, 'V_PT_FDR32_BN'), (82, 'V_PT_FDR32_CN'), (83, 'C_CT_FDR32_A'), (84, 'C_CT_FDR32_B'), (85, 'C_CT_FDR32_C'), (86, 'V_PT_FDR12_AN'), (87, 'V_PT_FDR12_BN'), (88, 'V_PT_FDR12_CN'), (89, 'C_CT_FDR12_A'), (90, 'C_CT_FDR12_B'), (91, 'C_CT_FDR12_C')]

emptyr = [ 116, 117, 118, 123, 124, 125, 109, 110, 111]  # nodes without connections to anything

order_attack = ['CT', 'Cphase', 'PT', 'Vphase', 'GPS', 'CT', 'Cphase', 'GPS']

# dictionary containing attack points with format number:(start time, end time, type, name)
attack_dict = {61: (2.20, 3.50, 'CT', 'C_CTW10_A'),
               62: (2.20, 3.50, 'CT', 'C_CTW10_B'),
               63: (2.20, 3.50, 'CT', 'C_CTW10_C'),
               #64: (2.20, 3.50, 'CT', 'C_CTW10_N'),
               52: (3.8, 4.5, 'Cphase', 'C_CTW9_C'),
               94: (4.5, 5.9, 'PT', 'V_PTW4_A'),
               95: (4.5, 5.9, 'PT', 'V_PTW4_B'),
               96: (4.5, 5.9, 'PT', 'V_PTW4_C'),
               89: (6.2, 7.1, 'Vphase', 'V_PTW4_BN'),
               76: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               77: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               78: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               #79: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               80: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               81: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
               82: (8.5, 9.4, 'GPS', 'C_CTWE1&V_PTWE1'),
#                123: (10.5, 11.5, 'PT', 'V_PT_SB_A'),
#                124: (10.5, 11.5, 'PT', 'V_PT_SB_B'),
#                125: (10.5, 11.5, 'PT', 'V_PT_SB_C'),
               29: (11.7, 12.7, 'CT', 'C_CTFDR2R1'),
               30: (11.7, 12.7, 'CT', 'C_CTFDR2R1'),
               31: (11.7, 12.7, 'CT', 'C_CTFDR2R1'),
               #32: (11.7, 12.7, 'CT', 'C_CTFDR2R1'),
               38: (14.6, 15.8, 'Cphase', 'C_CTFDR2R2_C'),
               15: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               16: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               17: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               #18: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               19: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               20: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3'),
               21: (16.5, 17.5, 'GPS', 'C_CTFDR3&V_PTFDR3')
               }

# dictionary containing the only attack without a compromised unit
attack_wo_comp_dict = {'Load_decrease': (10.20, 10.20, 'Bus_3DB211', 'all')}

# dictionary containing faults labeled    type:(start time, end time, location, effects?)
fault_dict = {'Downed_conductor': [3.25, 3.50, 'Bus_DBU3_A', 'all'], 'LG': [8.50, 8.57, 'Bus_FDR11LOAD_A', 'none'],
              'LL': [12.00, 12.08, 'Bus_3DB12', 'none']}

# dictionary containing normal events labeled    time:(type, power, location)
normal_dict = {6.10: ('L+', '1500kw+10kvar', 'Bus_3DB211'), 8.00: ('L+', '4500kw+1000kvar', 'Bus_TRAIN1'),
               10.20: ('L-', '1500kw+10kvar', 'Bus_3DB211'), 13.50: ('L+', '6500kw+2000kvar', 'Bus_FDR12LOAD'),
               14.00: ('L+', '1500kw+10kvar', 'Bus_2DB112'), 14.40: ('L-', '4500kw+1000kvar', 'Bus_TRAIN1'),
               18.80: ('L-', '6500kw+2000kvar', 'FDR12LOAD'), 19.50: ('L-', '1500kw+10kvar', 'Bus_2DB112'),
               19.80: ('L+', '1500kw+10kvar', 'Bus_3DB211')}
all_attack_times = set()
for k, values in attack_dict.items():
    all_attack_times.add((values[0], values[1]))

'''removing all the neutral'''
if remove_neutral:

    # sensors_connected = []
    new_adj_list = dict()
    new_attack_dict = dict()
    i = 0
    
    sensor_indices = sorted(list(adj_list.keys()))
    for no in sensor_indices:
#     for no, name in index_mapping:
        # if name[:-2] != "_N":
        if no not in emptyr:
            i += 1
            # sensors_connected.append(name)
            new_adj_list[i] = adj_list[no]
            if no in attack_dict:
                new_attack_dict[i] = attack_dict[no]

    adj_list = new_adj_list
    attack_dict = new_attack_dict

things = list(adj_list.keys())
things.sort()


#a set containing all attack times in (start,end) form
all_attack_times = set()
for k, values in attack_dict.items():
    all_attack_times.add((values[0], values[1]))
    
#neighbour_list is the sensor with all its neighbor in 
attacks = {}
for sensor, (start, end, attack_type, sensor_name) in attack_dict.items():
    if not remove_neutral:
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
            try:
                n.append(things.index(sensor))
            except:
                pass
        l.append(n)
    attacks_shifted_index[att] = l


def preprocess(df, time=0.25e6, remove_neutral=False):
    '''preprocessing data by removing the first few timestep to allow system to stabilize'''
    #returns dataframe
    if remove_neutral:
        df = df.loc[:, ~df.columns.str.endswith('_N')]
    return df[df['Time'] > time].iloc[:, 1:]


def get_nbd(neighbourhoods, save_set=True):
    '''get the attacked_sensor and attacked_neighbors from neighborhood'''
    #returns lists of attacked_sensors and attacked_nbd
    attacked_sensors = []
    attacked_nbd = []
    for nbd in neighbourhoods:
        attacked_sensors.append(nbd[0])
        attacked_nbd += nbd
    if save_set:
        attacked_sensors, attacked_nbd = set(attacked_sensors), set(attacked_nbd)
    return attacked_sensors, attacked_nbd

def correlation_pair(dataframe, var1_index, var2_index, window, diff=0, pct_change=False):
    '''pairwise correlation'''
    #returns an array
    corrs = dataframe.iloc[:, var1_index].rolling(window).corr(dataframe.iloc[:, var2_index])[window - 1:]
    corrs_change = np.diff(corrs, n=diff)
    if pct_change:
        corrs_change = np.divide(corrs_change, corrs[:-1])

    return corrs_change


def correlation_pair_plot(dataframe, var1_index, var2_index, window, freq=True,
                          attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    '''plot pairwise correlation'''
    #returns an array
    cols = dataframe.columns

    corrs = correlation_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change)

    plt.title(
        f"Sample Correlation against Time, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
    plt.plot(corrs)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    plt.show()

    if freq:
        plt.title(
            f"Sample Correlation Frequency, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
        plt.hist(corrs, bins=500)
        plt.show()

    return corrs


def correlation_multi(dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):
    '''product of multiple pairwise correlation'''
    #returns an array
    corrs = []
    for var2_index in var2_indices:
        corrs.append(correlation_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(corrs, axis=0)


def correlation_multi_plot(dataframe, var1_index, var2_indices, window, freq=True,
                           attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    '''plots multiple pairwise correlation'''
    #returns an array
    cols = dataframe.columns
    corrs = correlation_multi(dataframe, var1_index, var2_indices, window, diff=diff, pct_change=pct_change)

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
    '''variance ratio, using the difference of the ratios'''
    #returns an array
    var_x = dataframe.iloc[:, var1_index].rolling(window).var()[window - 1:]
    var_y = dataframe.iloc[:, var2_index].rolling(window).var()[window - 1:]
    var_ratio = np.divide(var_y, var_x)
    var_ratio_change = np.diff(var_ratio, n=diff)

    if pct_change:
        var_ratio_change = np.divide(var_ratio_change, var_ratio[:-diff])

    return var_ratio_change


def var_multi(dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):
    '''var pair but with more indexes'''
    #returns an array
    var = []
    for var2_index in var2_indices:
        var.append(var_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(var, axis=0)


def covariance_pair(dataframe, var1_index, var2_index, window, diff=0, pct_change=False):
    '''pairwise covariance'''
    #returns an array
    covs = dataframe.iloc[:, var1_index].rolling(window).cov(dataframe.iloc[:, var2_index])[window - 1:]
    covs_change = np.diff(covs, n=diff)
    if pct_change:
        covs_change = np.divide(covs_change, covs[:-1])

    return covs_change


def covariance_pair_plot(dataframe, var1_index, var2_index, window, freq=True,
                         attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    '''plot pairwise covariance'''
    #returns an array
    covs = covariance_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change)

    plt.title(
        f"Sample Covariance against Time, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
    plt.plot(covs)

    for start, end in attack_indices:
        plt.axvline(start, color='red')
        plt.axvline(end, color='red')

    for start, end in other_attack_indices:
        plt.axvline(start, color='pink', alpha=0.75)
        plt.axvline(end, color='pink', alpha=0.75)

    plt.show()

    if freq:
        plt.title(
            f"Sample Covariance Frequency, Window Length {window}, Variables: {cols[var1_index], cols[var2_index]}")
        plt.hist(covs, bins=500)
        plt.show()

    return covs


def covariance_multi(dataframe, var1_index, var2_indices, window, diff=0, pct_change=False):
    '''product of multiple pairwise covariance'''
    #returns an array
    covs = []
    for var2_index in var2_indices:
        covs.append(covariance_pair(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    return np.prod(covs, axis=0)


def covariance_multi_plot(dataframe, var1_index, var2_indices, window, freq=True,
                          attack_indices=[], other_attack_indices=[], cols=[], diff=0, pct_change=False):
    '''plot covariance_multi'''
    #returns an array

    covs = covariance_multi(dataframe, var1_index, var2_indices, window, diff=diff, pct_change=pct_change)

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


def stat_diff(dataframe, var1_index, var2_indices, window, stat_fn=covariance_pair, fn="prod", diff=1,
              pct_change=False):
    stat = []
    for var2_index in var2_indices:
        stat.append(stat_fn(dataframe, var1_index, var2_index, window, diff=diff, pct_change=pct_change))

    if fn == "prod":
        return np.prod(np.abs(stat), axis=0)
    elif fn == "sum":
        return np.mean(np.abs(stat), axis=0)
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
        # test_statdf = pd.DataFrame(test_stat)
        # test_statdf.plot.kde()
        if len(sensor_names) == 1:
            try:
                ax.hist(test_stat, bins=500)
            except:
                print(test_stat)
                print(sensor_names)
            ax.set_title(f"Rolling Sum of Difference Frequency of {cols[sensor_names[0]]}: {sensor_names[0]}")
        else:
            for i, n in enumerate(sensor_names):
                ax[0].hist(test_stat, bins=500)
                ax[0].set_title(f"Rolling Sum of Difference Frequency of {s}")
        plt.show()
    #########################################################

    # return the threshold
    # may need to make this 2-sided (for negative and positive)
    return np.quantile(test_stat, quantile)


def get_threshold(data, stat_fn, fn, cols, window=350, w=1, adj_list=adj_list, quantile=1, axis=1, stat_list=None,
                  plot=False):
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
            statistic = stat_fn(data, var1_index, var2_indices, window)

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


def get_threshold_stat_only(data, stat_fn, cols, window=350, w=1, adj_list=adj_list, quantile=1, axis=1, stat_list=None,
                            plot=False, collate_fn="prod"):
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
            statistic = stat_diff(data, var1_index, var2_indices, window, stat_fn, fn=collate_fn)

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
    diff_windows = (np.diff(windows, axis=1) == 1).all(axis=1)  # .squeeze()
    #     print(diff_windows.shape)
    #     attacks = np.argmax(diff_windows)

    # indices of persistent detection
    attacks_ = diff_windows.nonzero()[0]
    #     print(attacks)

    if len(attacks_) == 0:
        return None

    detection = windows[attacks_[0]]
    return detection[-1]


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
                index = np.nan
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


#import time
#t1 = time.perf_counter()

### Your code goes here ###

#t2 = time.perf_counter()
#print('time taken to run:',t2-t1)

'''trigger our method to run on the time that
time it takes to run for each sliding window
time how long does it take for the localization for one window,
times for different attack and average it'''

def avg_time (time, window):
    '''averages the time of the algo per sliding window'''
    return time/window

def FPR(violations, attack_times, window, w, total_normal, persistency):
    total_normal -= persistency

    attack_pred = (violations)

    shift = window + w - 1

    # go from back to prevent index issue when removing
    attack_times = sorted(list(attack_times), reverse=True)

    for start, end in attack_times:
        start_w = start - shift
        end_w = end + shift
        attack_indices = np.arange(start_w, end_w + 1)
        #         print("attack_pred:", attack_pred.shape)
        #         print("attack_indices:", attack_indices.shape)
        attack_pred = np.delete(attack_pred, attack_indices)

    attack_num = persistency_check_fpr(attack_pred.nonzero()[0], persistency)

    return attack_num / total_normal
