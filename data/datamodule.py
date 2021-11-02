
import pytorch_lightning as pl
import data.voxelized_dataset as voxelized_data


class DataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        args = self.args
        train_dataset = voxelized_data.VoxelizedDataset('train', voxelized_pointcloud=args.pointcloud,
                                                        pointcloud_samples=args.pc_samples, res=args.res,
                                                        sample_distribution=args.sample_distribution,
                                                        sample_sigmas=args.sample_sigmas, num_sample_points=50000,
                                                        batch_size=args.batch_size, num_workers=12)
        return train_dataset.get_loader()

    def val_dataloader(self):
        args = self.args
        val_dataset = voxelized_data.VoxelizedDataset('val', voxelized_pointcloud=args.pointcloud,
                                                      pointcloud_samples=args.pc_samples, res=args.res,
                                                      sample_distribution=args.sample_distribution,
                                                      sample_sigmas=args.sample_sigmas, num_sample_points=50000,
                                                      batch_size=args.batch_size, num_workers=12)
        return val_dataset.get_loader()
