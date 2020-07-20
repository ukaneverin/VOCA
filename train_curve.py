import numpy as np
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

data_directory = '/lila/data/fuchs/xiec/TIL_detection/'

for log_path in glob.glob('./train_logs/*.log'):
    log_name = log_path.split('/')[-1].split('.')[0] + log_path.split('/')[-1].split('.')[1]
    log_file = open(log_path, 'r')
    log_info = [line.strip() for line in log_file]
    if log_name.split('_')[0] == 'tune':
        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                loss_vector.append(float(info.split(' ')[-1]))
        exec("%s_train_regloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                loss_vector.append(float(info.split(' ')[-1]))
        exec("%s_val_regloss = loss_vector" % log_name)

    elif log_name.split('_')[0] == 'cls':
        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                loss_vector.append(float(info.split(' ')[2]))
        exec("%s_train_clsloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                acc_vector.append(float(info.split(' ')[-1]))
        exec("%s_train_acc = acc_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                loss_vector.append(float(info.split(' ')[2]))
        exec("%s_val_clsloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                acc_vector.append(float(info.split(' ')[-1]))
        exec("%s_val_acc = acc_vector" % log_name)
    else:
        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                loss_vector.append(float(info.split(' ')[2]))
        exec("%s_train_clsloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                loss_vector.append(float(info.split(' ')[4]))
        exec("%s_train_regloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'train':
                acc_vector.append(float(info.split(' ')[-1]))
        exec("%s_train_acc = acc_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                loss_vector.append(float(info.split(' ')[2]))
        exec("%s_val_clsloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                loss_vector.append(float(info.split(' ')[4]))
        exec("%s_val_regloss = loss_vector" % log_name)

        loss_vector = [];acc_vector = []
        for info in log_info:
            if info.split(' ')[0] == 'val':
                acc_vector.append(float(info.split(' ')[-1]))
        exec("%s_val_acc = acc_vector" % log_name)


colors = {'cls': 'r',
          'jointjoint': 'g',
          'tune': 'b',
          'jointweighted': 'b',
          'jointfocal2': 'r',
          'jointfocal5': (0.5,0.5,0.0),
          'subearly': (0.9,0.5,0.1),
          'sublate': (0.5,0.0,0.5)}

# 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'

linestyles = {'train': ':',
              'val': '-'}

for overlap_percentage in ['03', '05', '08']:
    for loss in ['clsloss', 'regloss', 'acc']:
        fig, ax = plt.subplots()
        handler_map = {}
        for option in ['cls', 'jointjoint', 'tune', 'jointweighted', 'jointfocal2', 'jointfocal5']:
            for phase in ['train', 'val']:

                #print(eval('%s_%s_%s_%s' % (option, overlap_percentage, phase, loss)))
                if '%s_%s_%s_%s' % (option, overlap_percentage, phase, loss) in globals():
                    ax.plot(eval('%s_%s_%s_%s' % (option, overlap_percentage, phase, loss)),
                            color = colors[option],
                            linestyle = linestyles[phase],
                            label = '%s_%s_%s_%s' % (option, overlap_percentage, phase, loss))


        legend = ax.legend(bbox_to_anchor=(1.04, 1), ncol=1)
        plt.xlabel('epochs')
        plt.ylabel('error/acc')
        plt.title('%s training curves for IoU threshold %s ' % (loss, overlap_percentage[0]+'.'+overlap_percentage[1]))
        fig.savefig(data_directory+'Training_Validation_curves_IoU_%s_%s.png' % (overlap_percentage, loss),
                    additional_artists = legend,
                    bbox_inches = 'tight')
        plt.close(fig)
