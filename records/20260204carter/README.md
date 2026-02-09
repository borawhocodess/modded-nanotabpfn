# PR ([#2](https://github.com/borawhocodess/modded-nanotabpfn/pull/2))

Only one run's log is included here for reference. The median run is the one added to the general record table.

```
#  experiment_id                  hostname   total_time  in_mins  roc_auc   epoch  μ_epoch_t  datasets
-  -----------------------------  ---------  ----------  -------  --------  -----  ---------  --------
1  260205-225948-e452851f-carter  dlc2gpu03  542.34s     9.04m    0.807031  206    2.63s      13184
2  260205-225951-d60000eb-carter  dlc2gpu09  606.25s     10.10m   0.809095  206    2.94s      13184
3  260205-230001-2a48a187-carter  dlc2gpu11  659.57s     10.99m   0.807605  239    2.76s      15296
4  260205-230002-7e1ea7a1-carter  dlc2gpu12  562.82s     9.38m    0.808042  209    2.69s      13376
5  260205-230004-a4f2c2f5-carter  dlc2gpu01  798.23s     13.30m   0.807989  306    2.61s      19584

stats: mean: 633.84s (10.56m) std: 91.50s median: 606.25s (10.10m)
```

## Plots

![val vs datasets plot](./260206-152132-records-plot-x-datasets-y-val.png)
![val vs time plot](./260206-152129-records-plot-x-time-y-val.png)
