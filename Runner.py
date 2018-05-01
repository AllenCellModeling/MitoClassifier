import os, sys
sys.path.append("./src")
from ThreeChannel import MitosisClassifier
from model_analysis import plot_confusion_matrix

n_of_it = 1
mito_runs = list()

for m_it in range(n_of_it):
    mito_runs.append( MitosisClassifier('/root/projects/three_channel', m_it) )
    mito_runs[m_it].run_me()
    print("itteration_{0} complete.".format(str(m_it).zfill(2)))


master_table = {k: {'true_labels': [], 'pred_labels': []} for k in mito_runs[0].phases()}
for k in range(n_of_it):
    for lname in mito_runs[0].phases():
        master_table[lname]['true_labels'] += mito_runs[k]['true_labels']
        master_table[lname]['pred_labels'] += mito_runs[k]['pred_labels']

img = plot_confusion_matrix(master_table['train']['true_labels'], master_table['train']['pred_labels'], classes=mito_runs[0].class_names)
img.save(mito_runs[0].ofname('CM_master_test', 'png'))
print("all Done.")