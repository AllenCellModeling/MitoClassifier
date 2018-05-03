import os, sys
sys.path.append("./src")

import matplotlib.pyplot as plt
from ThreeChannel import MitosisClassifierZProj
from model_analysis import plot_confusion_matrix
import numpy as np

n_of_it = 10
mito_runs = list()

for m_it in range(n_of_it):
    mito_runs.append( MitosisClassifierZProj('/root/projects/three_channel', m_it))
    mito_runs[m_it].run_me()
    print("itteration_{0} complete.".format(str(m_it).zfill(2)))


master_table = {k: {'true_labels': [], 'pred_labels': []} for k in mito_runs[0].mito_labels.keys()}
p_master_table = {k: {'pred_labels': [], 'pred_entropy': [], 'pred_uid': []} for k in mito_runs[0].pred_mito_labels.keys()}

for k in range(n_of_it):
    for lname in mito_runs[0].mito_labels.keys():
        print("lname: {0}".format(lname))
        master_table[lname]['true_labels'] += mito_runs[k].mito_labels[lname]['true_labels']
        master_table[lname]['pred_labels'] += mito_runs[k].mito_labels[lname]['pred_labels']

for k in range(n_of_it):
    for lname in mito_runs[0].pred_mito_labels.keys():
        p_master_table[lname]['pred_labels'] += mito_runs[k].pred_mito_labels[lname]['pred_labels']
        p_master_table[lname]['pred_entropy'] += mito_runs[k].pred_mito_labels[lname]['pred_entropy']
        p_master_table[lname]['pred_uid'] += mito_runs[k].pred_mito_labels[lname]['pred_uid']

fig, ax = plot_confusion_matrix(master_table['train']['true_labels'], master_table['train']['pred_labels'])
fig.savefig('CM_master_train.png', bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)

fig, ax = plot_confusion_matrix(master_table['test']['true_labels'], master_table['test']['pred_labels'])
fig.savefig('CM_master_test.png', bbox_extra_artists=(ax,), bbox_inches='tight')
plt.close(fig)

all_preds_labels = np.array(p_master_table['all']['pred_labels'])
all_preds_entropy = np.array(p_master_table['all']['pred_entropy'])
nz_preds_entropy = all_preds_entropy[all_preds_labels != 0]

n, bins, patches = plt.hist([all_preds_entropy, nz_preds_entropy], 25, density=True, alpha=0.75)
plt.savefig('Master_hist.png')
plt.close()

print("all Done.")