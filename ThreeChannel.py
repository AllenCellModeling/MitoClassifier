import os, sys, gc, copy, itertools, json

sys.path.append("src")
from train_model import train_model
from model_analysis import model_analysis, torch_confusion_matrix, plot_confusion_matrix
from plot_images import torch_to_PIL_single_image, ims_labels_to_grid, ims_preds_to_grid, ims_labels_preds_to_grid

import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, utils, models
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from IPython.core.display import display

from pytorch_learning_tools.data_providers.DataframeDataProvider import DataframeDataProvider
from pytorch_learning_tools.data_providers.DataframeDataset import DatasetSingleRGBImageToTarget, \
    DatasetSingleRGBImageToTargetUniqueID
from pytorch_learning_tools.utils.dataframe_utils import filter_dataframe
from pytorch_learning_tools.utils.data_utils import classes_and_weights

import collections


class MitosisClassifier(object):
    """This is a script to automate the running of the mitotic program

    Attributes:


    """

    def __init__(self, base_path, iter_n=0, f_label='MitosisLabel'):
        self.GPU_ID = 1
        self.BATCH_SIZE = 64
        self.base_path = base_path
        self.iter_n = iter_n
        self.greg_data = '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/'
        self.csv_input = 'input_data_files/mito_annotations_all.csv'
        self.f_type = 'save_flat_proj_reg_path'
        self.f_label = f_label
        # human readable classes
        self.m_class_names = np.array(["not mitotic",
                                     "M1: prophase 1",
                                     "M2: prophase 2",
                                     "M3: pro metaphase 1",
                                     "M4: pro metaphase 2",
                                     "M5: metaphase",
                                     "M6: anaphase",
                                     "M7: telophase-cytokinesis"])
        self.dp = None

    def class_names(self):
        return self.m_class_names

    def read_and_filter_input(self, f_label='MitosisLabel'):
        filt = f_label + ' >= 0'

        # read file
        df_unfiltered = pd.read_csv(self.csv_input,
                                    dtype={'structureSegOutputFilename': str,
                                           'structureSegOutputFolder': str}
                                    )

        # filter for mito annotations
        df = df_unfiltered.query(filt)
        df = df.reset_index(drop=True)

        # add absolute path
        df[self.f_type] = '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/' + df[self.f_type]

        # filter for rows where images are actually present
        df = filter_dataframe(df, '', self.f_type)

        # add numeric labels
        le = LabelEncoder()
        df['targetNumeric'] = le.fit_transform(df[f_label]).astype(np.int64)

        # print label map
        label_map = dict(zip(le.classes_, [int(i) for i in le.transform(le.classes_)]))
        print(json.dumps(label_map, indent=2))

        # save a csv
        csv_out = 'input_data_files/mito_annotations_only_with_pngs_{0}.csv'.format(str(self.iter_n).zfill(2))
        df.to_csv(csv_out, index=False)
        return df

    def three_channel_transform(self):
        w = 224
        s = 2.5

        # this is the xyz geometry imbeded in the image
        gx = 96
        gy = 128
        gz = 64

        # xyz once the image has been scaled by s
        nx = int(round(s * gx))
        ny = int(round(s * gy))
        nz = int(round(s * gz))

        nwidth = int(round(170 * s))
        nheight = int(round(230 * s))

        # offsets for each pain within the image
        xo = 0
        yo = 90
        x2o = 40
        y2o = 50
        x3o = 40

        # start coordinates for each fram
        x1s = xo + 0
        y1s = yo + 0
        x2s = x2o + nz
        y2s = y2o + (nheight - w - 50)
        x3s = x3o + nz

        return transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(nwidth), transforms.CenterCrop((nheight, nwidth)),
             transforms.ToTensor(), transforms.Lambda(lambda x: torch.stack(
                [x[2, y1s:(w + y1s), x1s:(w + x1s)], x[2, y1s:(w + y1s), x2s:(w + x2s)],
                 x[2, y2s:(w + y2s), x3s:(w + x3s)]]))])

    def create_data_provider(self, df, testf=0.2):
        split_fracs = {'train': 1.0 - testf, 'test': testf}
        split_seed = 1

        dataset_kwargs = {split: {'target': 'targetNumeric', 'image': self.f_type} for split in split_fracs.keys()}
        dataloader_kwargs = {
            split: {'batch_size': self.BATCH_SIZE, 'shuffle': True, 'drop_last': True, 'num_workers': 4, 'pin_memory': True}
        for
            split in split_fracs.keys()}

        dataset_kwargs['train']['imageTransform'] = self.three_channel_transform()
        dataset_kwargs['test']['imageTransform']  = self.three_channel_transform()

        self.dp = DataframeDataProvider(df, datasetClass=DatasetSingleRGBImageToTarget,
                                   split_fracs=split_fracs,
                                   split_seed=split_seed,
                                   uniqueID='save_h5_reg_path',
                                   dataset_kwargs=dataset_kwargs,
                                   dataloader_kwargs=dataloader_kwargs)

    def check_images(self, model, dkey='test'):
        i, mb = next(enumerate(dp.dataloaders[dkey]))
        ims_labels_preds = [(im, label, pred) for i, (im, label, pred) in enumerate(
            zip(mb['image'], mb['target'].numpy(),
                model(Variable(mb['image']).cuda(self.GPU_ID)).data.cpu().max(1)[1].numpy())) if i < 16]
        img = ims_labels_preds_to_grid(ims_labels_preds, ncol=4)
        fname = "Inspect_{0}".format(dkey)
        img.save(self.ofname(fname, "png"))


    def ofname(self, b_name, f_ext):
        fname = "{0}_{1}.{2}".format(b_name, str(self.iter_n).zfill(2), f_ext)
        return os.path.join(self.base_path, fname)

    def generate_class_weights(self):
        classes, weights = classes_and_weights(self.dp, split='train', target_col='targetNumeric')
        weights = weights.cuda(self.GPU_ID)
        CWP = collections.namedtuple('CWP', ['cls','weights'])
        return CWP(classes, weights = weights)

    def phases(self):
        return self.dp.dataloaders.keys();


    def select_n_train_n_run_model(self):
        cwp = self.generate_class_weights()
        model_name = 'resnet18'
        model_class = getattr(models, model_name)
        model = model_class(pretrained=True)

        model.fc = nn.Linear(model.fc.in_features, len(cwp.cls), bias=True)
        model = model.cuda(self.GPU_ID)

        N_epochs = 10
        model = train_model(model, self.dp,
                            class_weights=cwp.weights,
                            class_names=self.class_names(),
                            N_epochs=N_epochs,
                            phases=('train', 'test'),
                            learning_rate=1e-4,
                            gpu_id=self.GPU_ID)

        torch.save(model.state_dict(), self.ofname('saved_model_10E', 'pt'))
        self.check_images('train')

        model.eval()

        mito_labels = {k: {'true_labels': [], 'pred_labels': []} for k in self.dp.dataloaders.keys()}

        for phase in self.dp.dataloaders.keys():
            for i, mb in tqdm_notebook(enumerate(self.dp.dataloaders[phase]), total=len(self.dp.dataloaders[phase]),
                                       postfix={'phase': phase}):
                x = mb['image']
                y = mb['target']

                y_hat_pred = model(Variable(x).cuda(GPU_ID))
                _, y_hat = y_hat_pred.max(1)

                mito_labels[phase]['true_labels'] += list(y.data.cpu().squeeze().numpy())
                mito_labels[phase]['pred_labels'] += list(y_hat.data.cpu().numpy())

        #capture training / test data report
        self.mito_labels = mito_labels



        # model_analysis(mito_labels['train']['true_labels'], mito_labels['train']['pred_labels'])

        img = plot_confusion_matrix(mito_labels['train']['true_labels'], mito_labels['train']['pred_labels'], classes=self.class_names())
        img.save(self.ofname('CF_training', 'png'))

        img = plot_confusion_matrix(mito_labels['test']['true_labels'], mito_labels['test']['pred_labels'], classes=self.class_names())
        img.save(self.ofname('CF_test', 'png'))

        print("done with training and test.")
        #Apply to unannotated data

        df_no_annots_unfiltered = None
        # read file
        df_no_annots_unfiltered = pd.read_csv(self.csv_input, #  'input_data_files/mito_annotations_all.csv',
                                              dtype={'structureSegOutputFilename': str,
                                                     'structureSegOutputFolder': str}
                                              )

        # filter for NaN mito annotations -- a NaN isn't equal to itself
        lfilt = self.f_label + ' != ' + self.f_label
        df_no_annots = df_no_annots_unfiltered.query(lfilt)
        df_no_annots = df_no_annots.reset_index(drop=True)

        # add absolute path  #save_flat_proj_reg_path
        df_no_annots[self.f_type] = '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/' + df_no_annots[self.f_type]

        # filter for rows where images are actually present
        df_no_annots = filter_dataframe(df_no_annots, '/root/aics/modeling/gregj/results/ipp/ipp_17_12_03/', self.f_type)
        df_no_annots['targetNumeric'] = -1
        df_no_annots['targetNumeric'] = df_no_annots['targetNumeric'].astype(np.int64)

        # save a csv
        csv_out = 'input_data_files/mito_annotations_missing_with_pngs_{0}.csv'.format(str(self.iter_n).zfill(2))
        df_no_annots.to_csv(csv_out, index=False)

        split_fracs = {'all': 1.0}
        split_seed = 1

        dataset_kwargs = {split: {'target': 'targetNumeric', 'image': self.f_type, 'uniqueID': 'save_h5_reg_path'} for split
                          in split_fracs.keys()}
        dataloader_kwargs = {
        split: {'batch_size': self.BATCH_SIZE, 'shuffle': False, 'drop_last': False, 'num_workers': 4, 'pin_memory': True}
        for split in split_fracs.keys()}

        dataset_kwargs['all']['imageTransform'] = self.three_channel_transform()

        dp_no_annots = DataframeDataProvider(df_no_annots, datasetClass=DatasetSingleRGBImageToTargetUniqueID,
                                             split_fracs=split_fracs,
                                             split_seed=split_seed,
                                             uniqueID='save_h5_reg_path',
                                             dataset_kwargs=dataset_kwargs,
                                             dataloader_kwargs=dataloader_kwargs)

        print("get predictions.")
        # Get predictions
        model.eval()

        p_mito_labels = {phase: {'pred_labels': [], 'pred_entropy': [], 'pred_uid': []} for phase in
                       dp_no_annots.dataloaders.keys()}

        for phase in dp_no_annots.dataloaders.keys():
            for i, mb in tqdm_notebook(enumerate(dp_no_annots.dataloaders[phase]),
                                       total=len(dp_no_annots.dataloaders[phase]), postfix={'phase': phase}):
                x = mb['image']
                y = mb['target']
                u = mb['uniqueID']

                y_hat_pred = model(Variable(x).cuda(self.GPU_ID))
                _, y_hat = y_hat_pred.max(1)

                probs = F.softmax(y_hat_pred, dim=1)
                entropy = -torch.sum(probs * torch.log(probs), dim=1)

                p_mito_labels[phase]['pred_labels'] += list(y_hat.data.cpu().numpy())
                p_mito_labels[phase]['pred_entropy'] += list(entropy.data.cpu().numpy())
                p_mito_labels[phase]['pred_uid'] += u

        self.pred_mito_labels = p_mito_labels


    def run_me(self):
        df = self.read_and_filter_input()
        self.create_data_provider(df)
        print("ready to run.")
        self.select_n_train_n_run_model()
        print("finished.")

