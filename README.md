# convolutional-autoencoder


## Useage

Use Under-Sampling. Cross Entropy Loss (70%) and Binary Cross Entropy Loss (30%).

```
$ python train.py --imbalance US --ce_weights 0.7
```

Use Normal-Sampling. Cross Entropy Loss (100%) and Binary Cross Entropy Loss (0%).

```
$ python train.py --imbalance N --ce_weights 1.0
```
