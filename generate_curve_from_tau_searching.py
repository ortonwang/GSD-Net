import glob

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm
from matplotlib import font_manager, rcParams
import matplotlib.ticker as ticker

import math
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 'stix', 'cm', 'dejavusans', 'dejavuserif'
noise_type = 'DE'
result_dir = './model_search_tau/'
dataset= 'shenzhen'
dir_exp_h_ori = glob.glob(result_dir + dataset+'search_tau_*'+noise_type+'/log')

def extract_tau_order(folder):
    # folder = os.path.basename(os.path.dirname(path))
    s = folder.split(dataset+"search_tau_")[1].split(noise_type+"/log")[0]
    if len(s) == 2:      # 01, 02, 03
        main = int(s)
        sub = 0
    elif len(s) == 3:    # 015, 025
        # main = int(s[:-1])
        main = int(s[1:])
        # print(s)
        sub = 1
    else:
        raise ValueError(f"Unknown tau format: {s}")
    return (main, sub)
#code for sort tau value, for more intuitive visualization and tau selection
dir_exp_h = sorted(dir_exp_h_ori,    key=extract_tau_order)
exp_list_ori = {}
for exp_h in dir_exp_h:
    exp_name = exp_h.replace(noise_type+'/log','').split('/')[-1].replace(str(dataset)+'search_tau_','')
    exp_data_h = event_accumulator.EventAccumulator(exp_h).Reload().Scalars("Loss/loss_use_and_ignore_part_inverse")
    exp_list_ori[exp_name]=exp_data_h
def read_value_to_list(scalares_h):
    acc_h  = []
    for s in scalares_h:
        a = s.value
        if math.isnan(a):
            acc_h.append(0)
        else:
            acc_h.append(s.value)
    return acc_h
def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed
exp_list = {}
for tau, events in exp_list_ori.items():
    exp_list[tau] = read_value_to_list(events)

smooth_value = 0.98
exp_list_smooth = {}
for tau, events in exp_list.items():
    exp_list_smooth[tau] = smooth_curve(events, weight=smooth_value)

# compress, for more intuitive visualization
def compress_y(y, cutoff=0.1, factor=0.5):
    y = np.asarray(y, dtype=float)
    return np.where(y <= cutoff, y, cutoff + (y - cutoff) * factor)
def decompress_y(yc, cutoff=0.1, factor=0.5):
    yc = np.asarray(yc, dtype=float)
    return np.where(yc <= cutoff, yc, cutoff + (yc - cutoff) / factor)

first_key = next(iter(exp_list_smooth))
lens = len(exp_list_smooth[first_key])
start, end = 0, lens
x = range(start, end)

cutoff,factor,y_cutoff,y_factor = 200,1,0.3,0.05

exp_list_compress = {}
for tau, events in exp_list_smooth.items():
    exp_list_compress[tau] = compress_y(events, cutoff=y_cutoff, factor=y_factor)


colors =  [
    "#CFC33B",  # 黄
    "#79C347",  # 绿
    "#1f77b4",  # 蓝
    "black",    # 黑
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#ff7f0e",  # 橙
    "#8c564b",  # 棕
    "#17becf",  # 青
    "#2ca02c",  # 深绿
    "#bcbd22",  # 橄榄黄
    "#aec7e8",  # 浅蓝（对照 / 次要）
    "#e377c2",  # 粉紫（区分度高）
    "#7f7f7f",  # 灰（baseline / oracle）
    "#1a55A3",  # 深蓝（稳重）
    "#c49c94"   # 浅棕（补色）
]


plt.figure(figsize=(6,5),dpi=600)
line_width_here = 1
a = 0

for tau, events in exp_list_compress.items():
    plt.plot(x[start:end], events[start:end], label=r'$\tau$ = '+str(f"{float(tau)/100:.2f}"), linewidth=line_width_here, color=colors[a])
    a+=1


plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{decompress_y(y, cutoff=y_cutoff, factor=y_factor):.5f}"))

ticks_raw = np.concatenate(([0, 0.70], np.arange(0.73, 0.91, 0.03)))
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xlim(right=end)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

plt.xlabel(r"Iterations",fontsize=14)
plt.ylabel(r"Loss Value",fontsize=14)
plt.title("Loss Curve for shenzhen Dataset",fontsize=14)
plt.legend(loc="lower left",handlelength=1.5)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig('curve_for_drop_out_rate_search.png',dpi=600)
plt.show()

print('fds')