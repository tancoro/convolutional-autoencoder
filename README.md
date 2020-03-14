# convolutional-autoencoder


## Useage

Use Under-Sampling. Cross Entropy Loss (70%), Binary Cross Entropy Loss (30%).

```
# Cross Entropy Loss(50%)
$ python train.py --imbalance US --ce_weights 0.7
```

Use Normal-Sampling. Cross Entropy Loss (100%), Binary Cross Entropy Loss (0%).

```
# Cross Entropy Loss(50%)
$ python train.py --imbalance US --ce_weights 1.0
```