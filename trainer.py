#
# Developed by Farid Yagubbayli <faridyagubbayli@gmail.com> | <farid.yagubbayli@tum.de>
#

import argparse
import pytorch_lightning as pl

import models.local_model as model
from data.datamodule import DataModule
from utils import form_expr_name

pl.seed_everything(100)

# python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
parser = argparse.ArgumentParser(
    description='Run Model'
)

parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
parser.add_argument('-voxels', dest='pointcloud', action='store_false')
parser.set_defaults(pointcloud=False)
parser.add_argument('-pc_samples' , default=3000, type=int)
parser.add_argument('-dist','--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
parser.add_argument('-std_dev','--sample_sigmas',default=[0.15,0.015], nargs='+', type=float)
parser.add_argument('-batch_size' , default=30, type=int)
parser.add_argument('-res' , default=32, type=int)
parser.add_argument('-m','--model' , default='LocNet', type=str)
parser.add_argument('-o','--optimizer' , default='Adam', type=str)

try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

model_map = {
    'ShapeNet32Vox': model.ShapeNet32Vox,
    'ShapeNet128Vox': model.ShapeNet128Vox,
    'ShapeNetPoints': model.ShapeNetPoints,
    'SVR': model.SVR,
}

assert args.model in model_map

net = model_map[args.model]()
data_module = DataModule(args)

logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=form_expr_name(args))

trainer = pl.Trainer(max_epochs=1500, checkpoint_callback=False,
                     logger=logger, gpus=1, num_sanity_val_steps=0)
trainer.fit(net, data_module)
