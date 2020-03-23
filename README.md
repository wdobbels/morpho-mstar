# Morphological stellar mass

A workshop for predicting the total stellar mass of a galaxy, based on a single-band image.

# Setup

## Image format

Use the log-scaled images (with automatic scaling value), and convert to png. A good pixel scale seems to be 128x128, for which the results are a few percentage better than for the 69x69 pixel sizes. It is possible to train first on 69x69 and then further tune (transfer learning) on 128x128.

## Additional metadata

Including the distance and g-band luminosity (from SED fit, so no longer pure single band estimate) results in a very noticable improvement (RMSE from 0.88 to 0.75, R2 from 0.48 to 0.63).

It is possible to also include the galaxy zoo data. This allows one to pretrain on morphology, which is more directly related to the image. However, this makes the setup somewhat more complicated. A different loss function has to be set up, the output needs to be properly normalized (questions follow a tree structure, where the conditional probabilities are subject to constraints), and only a subset of the training data has GZ2 data. However, keras (and other ML libraries) make it quite easy to set up multi-input, multi-output models.

## Output

Use M/L as the prediction target. With the g-band luminosity available, it is possible to convert to stellar mass (just for fun).

## Training set size

10000 images seems ok. Without additional metadata, this leads to an RMSE of 0.91. By going to 30000 images, the RMSE decreases to 0.88. However, for more complicated architectures, a larger training set would be more beneficial.

## Architecture

The custom architecture (from the paper) has 5 convolutional layers, and 2 fully connected layers. This works well (without the need for morphology), and trains fast.

It might be possible to use something like a ResNet34, pretrained on imagenet. First tune the top (fully connected) layers, then unfreeze the rest. However, this of course will train quite a bit slower. Imagenet is not that useful for this dataset, but it is better than nothing, and of course the architecture is proven to work well.

# Results summary

| npix | ntrain | add $D$ & $L_g$? | RMSE  | $R^2$ |
| ---- | ------ | ------------ | ----- | ----- |
| 69   | 10 000 | no           | 0.942 | 0.411 |
| 128  | 10 000 | no           | 0.915 | 0.445 |
| 128  | 30 000 | no           | 0.882 | 0.484 |
| 69   | 10 000 | yes          | 0.853 | 0.517 |
| 128  | 30 000 | yes          | 0.747 | 0.630 |
