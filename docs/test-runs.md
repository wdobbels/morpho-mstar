# Experiments

Some of the experiments to get to a viable, simple pipeline for predicting M/L from g-band images.

## Pure-CNN

### ResNet-50 V2 with 3 Dense layers on top

Unsuccessful without log-scaling (and final layer ReLU).

## CNN features with LGBM

### ResNet-50 V2 (fixed)

RMSE = 0.227, MAE = 0.154, R2 = 0.171
So, not very good, but probably best we can do with fixed (ResNet-50) CNN.