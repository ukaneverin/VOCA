import numpy as np
import glob
from scipy.misc import imsave
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

til_path = '/lila/home/xiec/TIL_detection/til'

survival_file = open('survival.csv', 'w+')
survival_file.write('svs_id,cell_count,tile_count,test_time\n')



all_til = []
for log_file in glob.glob(til_path+'/*.log'):
    svs_name = log_file.split('/')[-1].split('.')[0]
    log = open(log_file, 'r')
    data = [line.strip().split(' ') for line in log][-1]
    cell_number = int(data[0])
    tile_number = int(data[1])
    test_time = float(data[2])
    if tile_number != 0:
        survival_file.write('%s,%s,%s,%s\n' % (svs_name, cell_number, tile_number, test_time))
        til_percent = cell_number*2*3.14*9.5*9.5/(tile_number*762*762)
        all_til.append(til_percent)
all_til.sort()
l = len(all_til)
medium = all_til[int(l/2)]
one_third = all_til[15]
two_third = all_til[55]

detect_results = {}
for log_file in glob.glob(til_path+'/*.log'):
    svs_name = log_file.split('/')[-1].split('.')[0]
    log = open(log_file, 'r')
    data = [line.strip().split(' ') for line in log][-1]
    cell_number = int(data[0])
    tile_number = int(data[1])
    if tile_number != 0:
        til_percent = cell_number/(tile_number*762*762)
        detect_results[svs_name] = [cell_number, tile_number, til_percent]


plt.hist(all_til, bins = 74)
#plt.xlim([0,0.5])
plt.savefig('til_hist.png')
plt.close()
#get id dictionary
id_dict = {}
id_file = open('id_dict.csv', 'r')
id_list = [line.strip().split(',') for line in id_file]
pathologist_til = []; our_til = []; all_patients = set()
for id_info in id_list:
    if id_info[5] == '1':
        svs_id = id_info[6][-10:-4]
        if svs_id in detect_results:
            all_patients.add(id_info[3])
            if id_info[4][-1] == '%':
                if id_info[4][0] != '<':
                    path_percent = int(id_info[4][:-1])/100.0
                else:
                    path_percent = 0.05
            else:
                path_percent = 0.1

            if path_percent < 0.1:
                path_til_group = 0
            elif path_percent >=0.1 and path_percent < 0.6:
                path_til_group = 1
            else:
                path_til_group = 2

            id_dict[id_info[3]] = [svs_id, path_til_group]

            pathologist_til.append(path_percent)
            our_til.append(float(detect_results[svs_id][-1]))

"""normalize to the same scale as pathologist"""
area_per_cell  = 1.0/max(our_til)

detect_results = {}
for log_file in glob.glob(til_path+'/*.log'):
    svs_name = log_file.split('/')[-1].split('.')[0]
    log = open(log_file, 'r')
    data = [line.strip().split(' ') for line in log][-1]
    cell_number = int(data[0])
    tile_number = int(data[1])
    test_time = float(data[2])
    if tile_number != 0:
        til_percent = cell_number*area_per_cell/(tile_number*762*762)
        if til_percent < medium:
            til_group_2 = 0
        else:
            til_group_2 = 1

        if til_percent < 0.1:
            til_group_3 = 0
        elif til_percent < 0.6 and til_percent >= 0.1:
            til_group_3 = 1
        else:
            til_group_3 = 2
        detect_results[svs_name] = [cell_number, tile_number, test_time, til_group_2, til_group_3, til_percent]


low_number = 0
mid_number = 0
high_number = 0
for p in id_dict:
    if int(id_dict[p][1] ) == 0:
        low_number += 1
    elif int(id_dict[p][1] ) == 1:
        mid_number+=1
    else:
        high_number+=1
print(low_number,mid_number,high_number)

our_til = np.asarray(our_til)
our_til = area_per_cell*our_til

plt.scatter(pathologist_til, our_til)
plt.xlabel('pathologist_til_estimation')
plt.ylabel('our_til_estimation')
plt.savefig('til_scatter.png')
plt.close()

print(len(all_patients))
#create csv file for survival plot
survival_plot_file = open('survival_plot.csv', 'w+')
survival_plot_file.write('DSSstatus,DSSperiodmonths,Osstatus,OSSperiodmonths,path_group,our_group_2,our_group_3,til_count,svs_id\n')
data_1 = open('survival_data_1.csv', 'r')
data_1_list = [line.strip().split(',') for line in data_1][1:]
for i in range(len(data_1_list)):
    if data_1_list[i][2] in all_patients:
        DSSstatus = (data_1_list[i][6])
        DSSperiodmonths = (data_1_list[i][8])
        Osstatus = data_1_list[i][5]
        OSSperiodmonths = data_1_list[i][7]
        path_group = (data_1_list[i][4])
        group_2 = (detect_results[id_dict[data_1_list[i][2]][0]][-3])
        group_3 = (detect_results[id_dict[data_1_list[i][2]][0]][-2])
        til_number = detect_results[id_dict[data_1_list[i][2]][0]][-1]
        survival_plot_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (DSSstatus, DSSperiodmonths,Osstatus, OSSperiodmonths, path_group, group_2, group_3, til_number,id_dict[data_1_list[i][2]][0]))

data_2 = open('survival_data_2.csv', 'r')
data_2_list = [line.strip().split(',') for line in data_2][1:]
for i in range(len(data_2_list)):
    if data_2_list[i][3] in all_patients:
        DSSstatus = (data_2_list[i][7])
        DSSperiodmonths = (data_2_list[i][5])
        Osstatus = data_2_list[i][6]
        OSSperiodmonths = data_2_list[i][4]
        path_group = id_dict[data_2_list[i][3]][1]
        group_2 = (detect_results[id_dict[data_2_list[i][3]][0]][-3])
        group_3 = (detect_results[id_dict[data_2_list[i][3]][0]][-2])
        til_number = detect_results[id_dict[data_2_list[i][3]][0]][-1]
        survival_plot_file.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (DSSstatus, DSSperiodmonths,Osstatus, OSSperiodmonths, path_group, group_2, group_3, til_number,id_dict[data_2_list[i][3]][0]))
survival_plot_file.close()
