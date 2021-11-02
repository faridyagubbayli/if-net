#
# Developed by Farid Yagubbayli <faridyagubbayli@gmail.com>
#

import pytorch_lightning as pl
import models.local_model as model
from data.datamodule import DataModule
from utils import *

pl.seed_everything(100)
args = prepare_and_parse_args()

model_map = {
    'ShapeNet32Vox': model.ShapeNet32Vox,
    'ShapeNet128Vox': model.ShapeNet128Vox,
    'ShapeNetPoints': model.ShapeNetPoints,
    'SVR': model.SVR,
}
net = model_map[args.model](args.optimizer)

data_module = DataModule(args)

logger = pl.loggers.TensorBoardLogger(save_dir='lightning_logs', name=form_expr_name(args))

trainer = pl.Trainer(max_epochs=1500, checkpoint_callback=False,
                     logger=logger, gpus=1, num_sanity_val_steps=0)
trainer.fit(net, data_module)
