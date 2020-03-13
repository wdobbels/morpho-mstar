# Experiments

Some of the experiments to get to a viable, simple pipeline for predicting M/L from g-band images.

## Pure-CNN

### ResNet-50 V2 with 3 Dense layers on top

Unsuccessful without log-scaling (and final layer ReLU).
With log scaling: reports very low error after 2 epochs (MSE = 0.14 after epoch 2, MSE = 0.10 after epoch 3), but predictions seem to be very wrong (predicting between 20 and 55 instead of -0.75 to 1).

## CNN features with LGBM

### ResNet-50 V2 (fixed)

RMSE = 0.227, MAE = 0.154, R2 = 0.171
So, not very good, but probably best we can do with fixed (ResNet-50) CNN.