#!/usr/bin/env python
# coding: utf-8

# ## Approximation error line plot

# ### Approximation Error General Kernels

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from collections import defaultdict as dd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


tasks = ['cod-rna', 'letter', 'eeg', 'ijcnn1', 'skin', 'magic']

def extract_data(kernel):
    rff = dd(list)
    orf = dd(list)
    sorf = dd(list)
    fp_rff = dd(list)
    fp_orf = dd(list)
    fp_sorf = dd(list)

    for t in tasks:

        with open(f"resources/hardware/{t}/hardware_err.pkl" ,'rb') as f: err = pickle.load(f)
        rff[t] = err['rff'][kernel]
        orf[t] = err['orf'][kernel]
        sorf[t] = err['sorf'][kernel]

        with open(f"resources/hardware/{t}/fp_err.pkl" ,'rb') as f: err = pickle.load(f)
        fp_rff[t] = err['rff'][kernel]
        fp_orf[t] = err['orf'][kernel]
        fp_sorf[t] = err['sorf'][kernel]

    norm = 0
    for t in tasks:
        max_task_err = max(np.mean(rff[t][1]), np.mean(orf[t][1]),  np.mean(sorf[t][1]))
        if max_task_err > norm: norm = max_task_err

    def mean(x, norm, tasks):
        return np.array([np.mean([np.mean(x[t][n] / norm) for t in tasks])for n in range(1, 6)])
    def std(x, norm, tasks):
        return np.array([np.mean([np.std(x[t][n] / norm) for t in tasks])for n in range(1, 6)])
    
    approx = [rff, orf, sorf]
    fp = [fp_rff, fp_orf, fp_sorf]

    mean_fp = np.mean([mean(t, norm, tasks) for t in fp], axis=0)
    std_fp = np.mean([std(t, norm, tasks) for t in fp], axis=0)
    mean_app = np.mean([mean(t, norm, tasks) for t in approx], axis=0)
    std_app = np.mean([std(t, norm, tasks) for t in approx], axis=0)
    
    return mean_app, std_app, mean_fp, std_fp

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(6, 4.5)

cycle = ["#1f77b4","#ff7f0e",]

rbf_mean, rbf_std, rbf_fp_mean, rbf_fp_std = extract_data("rbf")
arc_mean, arc_std, arc_fp_mean, arc_fp_std = extract_data("arccos0")

ax.plot(range(1, 6), rbf_mean,marker='^',lw=2, linestyle="dashed",color=cycle[0],)
ax.fill_between(range(1, 6), rbf_mean+rbf_std, rbf_mean-rbf_std, facecolor=cycle[0], alpha=0.2)
ax.plot(range(1, 6), rbf_fp_mean, marker='^',lw=2, linestyle="solid",color=cycle[0],)
ax.fill_between(range(1, 6), rbf_fp_mean+rbf_fp_std, rbf_fp_mean-rbf_fp_std, facecolor=cycle[0], alpha=0.2)


ax.plot(range(1, 6), arc_mean, lw=2,marker='^',linestyle="dashed",color=cycle[1],)
ax.fill_between(range(1, 6), arc_mean+arc_std, arc_mean-arc_std, facecolor=cycle[1], alpha=0.2)
ax.plot(range(1, 6), arc_fp_mean, lw=2,marker='^',linestyle="solid",color=cycle[1],)
ax.fill_between(range(1, 6), arc_fp_mean+arc_fp_std, arc_fp_mean-arc_fp_std, facecolor=cycle[1], alpha=0.2)

ax.set_xlim([1,5])
ax.set_xlabel("$\log_2(s/d)$", fontsize=12)
ax.set_ylabel("Normalized Average Approximation Error", fontsize=12)
rff_patch = mpatches.Patch(color=cycle[0], label="RBF")
orf_patch = mpatches.Patch(color=cycle[1], label="ArcCos0")
dash = Line2D([0], [0], label="HW", color="k", linestyle="dashed")
soli = Line2D([0], [0], label="FP", color="k", linestyle="solid")
lgd = plt.legend(handles=[ dash, soli],  loc="best", ncol=1, )
# plt.grid(axis='y')

# plt.show()
plt.savefig(f"resources/hardware/rbf_err.pdf",bbox_extra_artists=(lgd,),bbox_inches="tight",)
plt.close("all")


# ### Accuracy General Kernels


import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from collections import defaultdict as dd

tasks = ['cod-rna', 'letter', 'eeg', 'ijcnn1', 'skin', 'magic']
# tasks = ['letter', 'eeg', 'ijcnn1', 'skin']

def extract_data(kernel):
    rff = dd(list)
    orf = dd(list)
    sorf = dd(list)
    fp_rff = dd(list)
    fp_orf = dd(list)
    fp_sorf = dd(list)

    for t in tasks:

        with open(f"resources/hardware/{t}/hardware_acc.pkl" ,'rb') as f: acc = pickle.load(f)
        rff[t] = acc['rff'][kernel]
        orf[t] = acc['orf'][kernel]
        sorf[t] = acc['sorf'][kernel]
    
        with open(f"resources/hardware/{t}/fp_acc.pkl" ,'rb') as f: acc = pickle.load(f)
        fp_rff[t] = acc['rff'][kernel]
        fp_orf[t] = acc['orf'][kernel]
        fp_sorf[t] = acc['sorf'][kernel]


    mean_acc_fp = dict()
    std_acc_fp = dict()
    for t in tasks:
        mean_acc_fp[t] = np.mean([np.max([np.mean(x[n]) for n in range(1, 6)])for x in [fp_rff[t], fp_orf[t], fp_sorf[t]]])
        std_acc_fp[t] = np.std([np.max([np.mean(x[n]) for n in range(1, 6)])for x in [fp_rff[t], fp_orf[t], fp_sorf[t]]])

    mean_acc = dict()
    std_acc = dict()
    for t in tasks:
        mean_acc[t] = np.mean([np.max([np.mean(x[n]) for n in range(1, 6)])for x in [rff[t], orf[t], sorf[t]]])
        std_acc[t] = np.std([np.max([np.mean(x[n]) for n in range(1, 6)])for x in [rff[t], orf[t], sorf[t]]])
    return mean_acc,std_acc,mean_acc_fp,std_acc_fp

# plt.set_cmap("Pastel2")
fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(15, 6)

# set width of bar
barWidth = 0.2
 
# Set position of bar on X axis
br1 = [x - 3*barWidth/2 - 0.02 for x in range(6)]
br2 = [x - barWidth/2 - 0.02 for x in range(6)]
br3 = [x + barWidth/2 + 0.02 for x in range(6)]
br4 = [x + 3*barWidth/2 + 0.02 for x in range(6)]

mean_acc,std_acc,mean_acc_fp,std_acc_fp = extract_data("rbf")
mean_acc_ar,std_acc_ar,mean_acc_fp_ar,std_acc_fp_ar = extract_data('arccos0')

# Make the plot
bar = plt.bar(br1, mean_acc_fp.values(), width = barWidth,  yerr=std_acc_fp.values(), label ='RBF FP', capsize=7,align='center', edgecolor="black")
plt.bar(br2, mean_acc.values(), width = barWidth, color="#1f77b4", yerr=std_acc.values(),  label ='RBF HW', capsize=7,align='center', hatch='///', edgecolor="black")
bar1 = plt.bar(br3, mean_acc_fp_ar.values(), width = barWidth,  yerr=std_acc_fp_ar.values(), label ='ArcCos0 FP', capsize=7,align='center', edgecolor="black")
plt.bar(br4, mean_acc_ar.values(), width = barWidth, color="#ff7f0e", yerr=std_acc_ar.values(),  label ='ArcCos0 HW', capsize=7,align='center', hatch='///', edgecolor="black")

for i, rect in enumerate(bar):
    height = rect.get_height()
    delta = (list(mean_acc.values())[i] - list(mean_acc_fp.values())[i]) * 100
    hdelta = 0.008 if i != 4 else 0.005
    plt.text(rect.get_x() + barWidth, height + hdelta, f'$\Delta={delta:.3f}$%', ha='center', va='bottom')
for i, rect in enumerate(bar1):
    height = rect.get_height()
    delta = (list(mean_acc_ar.values())[i] - list(mean_acc_fp_ar.values())[i]) * 100
    hdelta = 0.008 if i != 4 else 0.03
    plt.text(rect.get_x() + barWidth, height + hdelta, f'$\Delta={delta:.3}$%', ha='center', va='bottom')
 
# Adding Xticks
plt.ylim([0.6, 1])
plt.xlabel('Dataset')
plt.ylabel('Downstream Accuracy')
plt.xticks([r for r in range(len(tasks))], tasks)
ax.xaxis.set_ticks_position('none') 
# plt.xlim(-0.6, 5.6)

plt.legend()
# plt.show()
plt.savefig(f"resources/hardware/acc.pdf",bbox_inches="tight",)
plt.close("all")


# ### Approximation Error FAVOR+

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
from matplotlib.lines import Line2D

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(6, 4.5)
cycle = [
        "#1f77b4",
        "#ff7f0e",
    ]
with open(f"resources/hardware/attention/hardware_err.pkl" ,'rb') as f: err = pickle.load(f)
with open(f"resources/hardware/attention/emulated_fp_err.pkl" ,'rb') as f: fp = pickle.load(f)


def mean(x):
    return np.array([np.mean([x['favor+'][n]]) for n in range(1,8)])

def std(x):
    return np.array([np.std([x['favor+'][n]]) for n in range(1,8)])



ax.plot(range(1, 8), mean(err),marker='^',lw=2, linestyle="dashed",color=cycle[0],)
ax.fill_between(range(1, 8), mean(err)+std(err), mean(err)-std(err), facecolor=cycle[0], alpha=0.2)
ax.plot(range(1, 8), mean(fp), marker='^',lw=2, linestyle="solid",color=cycle[0],)
ax.fill_between(range(1, 8), mean(fp)+std(fp), mean(fp)-std(fp), facecolor=cycle[0], alpha=0.2)


ax.set_xlabel("$s/d$", fontsize=12)
ax.set_ylabel("approximation error", fontsize=12)

dash = Line2D([0], [0], label="HW", color="k", linestyle="dashed")
soli = Line2D([0], [0], label="FP", color="k", linestyle="solid")
lgd = plt.legend(handles=[ dash, soli],  loc="best", ncol=1, )
# plt.grid(axis='y')

# plt.show()
plt.savefig(f"resources/hardware/favor_err.pdf",bbox_extra_artists=(lgd,),bbox_inches="tight",)
plt.close("all")


# ### Attention Matrices Visualization

# In[12]:


import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

with open(f"resources/hardware/attention/attn_mat.pickle" ,'rb') as f:
    mat = pickle.load(f)

def arg_padding(m):
    m_diag = np.diag(m)
    embedding = m_diag[-1]
    for i in range(len(m_diag)):
        if m_diag[i] == embedding: return i
    
def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def threshold_and_norm(m, p=0.03):
    for x in np.arange(0, 1, 0.01):
        b = np.unique(m>x, return_counts=True)
        if b[1][1] / b[1][0] < p: break
    m[m > x] = x
    return norm(m)

for i in range(0, 1):

    idx = arg_padding(mat[i]['a'])
    a = norm(mat[i]['a'][:idx, :idx])
    a = threshold_and_norm(a)
    a_hat = norm(mat[i]['a_hat'][:idx, :idx])
    a_hat = threshold_and_norm(a_hat)
    a_hat_fp = norm(mat[i]['a_hat_fp'][:idx, :idx])
    a_hat_fp = threshold_and_norm(a_hat_fp, p=0.04)

    def plot(m, title, j):
        fig, ax = plt.subplots()
        fig.set_size_inches(6.2, 6)

        ax.imshow(m, cmap="inferno", vmin=np.min(m), vmax=np.max(m))
        ax.set_title(title, y=-0.08, size=16)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"resources/hardware/attn{i}_{j}.pdf")
        plt.close("all")

    plot(a, '$Attn(Q,K,V)$', 0)
    plot(a_hat_fp, '$\widetilde{Attn}(Q,K,V)$', 1)
    plot(a_hat, '$\widetilde{Attn}_{HW}(Q,K,V)$', 2)        

