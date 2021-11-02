import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F


class BaseModel(pl.LightningModule):

    def configure_optimizers(self):
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.run_step(batch, 'val')

    def run_step(self, batch, tag):
        loss = self.compute_loss(batch)
        self.log(f'loss/{tag}', loss)
        return loss

    def compute_loss(self, batch):
        device = self.device

        p = batch.get('grid_coords').to(device)
        occ = batch.get('occupancies').to(device)
        inputs = batch.get('inputs').to(device)


        # General points
        logits = self(p,inputs)
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.

        loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        return loss
