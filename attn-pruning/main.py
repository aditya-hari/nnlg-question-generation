from analysis import *

dm = DataModule()
dl = dm.val_dataloader()
#compute_heads_importance(model.to(DEVICE), dl)
h = mask_heads(model.to(DEVICE), dl)