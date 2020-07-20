import numpy as np
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

data_directory = '/lila/data/fuchs/xiec/TIL_detection/'

for score_file in glob.glob(data_directory+'detection_score/*.npy'):
    file_name = score_file.split('/')[-1].split('.')[0] + score_file.split('/')[-1].split('.')[1]
    vector = np.load(score_file)
    exec("%s = vector" % file_name)

linestyle = {'jointbalanced': '-',
          'jointweighted': '--'}

colors = {'vgg': 'm',
        'vggsplit': 'r',
          'vggskip': 'g',
          'vggsub': 'b',
          'res': 'c',
          'dense': 'y'}

for option in ['jointbalanced', 'jointweighted']:


    fig, ax = plt.subplots()
    handler_map = {}
    for config in ['vgg', 'vggsplit', 'vggskip', 'vggsub', 'res', 'dense']:
        try:
            ax.plot(eval('test_recall_jointweighted_%s_180' % config),
                    eval('test_precision_jointweighted_%s_180' %  config),
                    color = colors[config],
                    label = config)
        except:
            pass

    legend = ax.legend(loc='upper right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('Precision-Recall Curves for %s' % option)
    fig.savefig(data_directory+'Precision_Recall_Curve_%s.png' % option)
    plt.close(fig)
