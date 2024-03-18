from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='/home/whc/Desktop/Noisy/code/AoANet-master/log/log_aoanet_u_20_kl_save_5_2/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='/home/whc/Desktop/Noisy/code/AoANet-master/log/log_aoanet_u_20_kl_save_5_2/infos_aoanet_u_20_kl_save_5_2-best.pkl',
                help='path to infos to evaluate')
# parser.add_argument('--split', type=str, default='test',
#                 help='path to infos to evaluate')
# parser.add_argument('--dump_images', type=int,  default=0,
#                 help='')
# parser.add_argument('--dump_json', type=int,  default=1,
#                 help='')
# parser.add_argument('--num_images', type=int,  default=-1,
#                 help='')
# parser.add_argument('--language_eval', type=int,  default=1,
#                 help='')
# parser.add_argument('--beam_size', type=int,  default=2,
#                 help='')
# parser.add_argument('--batch_size', type=int,  default=100,
#                 help='')
# parser.add_argument('--split', type=str,  default='test',
#                 help='')

opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()
# crit = utils.LabelSmoothing()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
