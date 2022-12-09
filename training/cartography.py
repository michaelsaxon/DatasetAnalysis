import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from manage_settings import get_write_settings, lazymkdir

def define_samplewise_metric(key_nums_dict):
    out = {}
    for key in key_nums_dict.keys():
        out[key] = np.array(key_nums_dict[key])
    return out

# confidence
# C_i = 1/E \sum_epochs p_model(label | input)
def confidence_elementwise(targets, logits):
    return torch.gather(logits, -1, targets.unsqueeze(-1)).detach().cpu().numpy()

def correct_elementwise(targets, preds):
    return torch.eq(preds, targets).squeeze().detach().cpu().numpy()

# a model will contain self.confidences and self.correctnesses, two dicts of dicts
# integrate a new value into it from the returned elements
def accumulate_cartography_metric(keys, values, target_dict):
    # keys, values are ndarrays
    print(keys)
    print(values)
    print(target_dict)
    target_dict[keys] = values.squeeze()

class CartographyCallback(Callback):
    def __init__(self, output_base):
        super().__init__()
        self.output_base = output_base
        lazymkdir(output_base)

    def init_buffers(self, trainer):
        key_nums = {
            "train" : len(trainer.datamodule.train), 
            "val" : len(trainer.datamodule.valid), 
            "test" : len(trainer.datamodule.test)
        }
        self.confidences = define_samplewise_metric(key_nums)
        self.correctnesses = define_samplewise_metric(key_nums)

    def cartography_save(self, epoch, key):
        fname = f"{self.output_base}/$$$_{key}_{epoch}.npy"
        np.save(fname.replace("$$$", "conf"), self.confidences[key])
        np.save(fname.replace("$$$", "corr"), self.correctnesses[key])

    def batch_end_accumulate(self, batch, outputs, key):
        targets = batch['labels'].squeeze()
        logits = outputs['logits']
        preds = torch.max(logits, dim=-1).indices
        batch_idces = batch['idx'].cpu().numpy().squeeze()
        accumulate_cartography_metric(
            batch_idces,
            confidence_elementwise(targets, logits),
            self.confidences[key]
        )
        accumulate_cartography_metric(
            batch_idces,
            correct_elementwise(targets, preds),
            self.correctnesses[key]
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end_accumulate(batch, outputs, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end_accumulate(batch, outputs, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end_accumulate(batch, outputs, "test")

    def on_train_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "train")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "val")
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "test")

    def on_train_start(self, trainer, pl_module):
        self.init_buffers(trainer)
    
    def on_sanity_check_start(self, trainer, pl_module):
        self.init_buffers(trainer)