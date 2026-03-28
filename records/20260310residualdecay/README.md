# Residual Decay

![NanoTabPFN Diagram with Residual Decay](residualdecay.png)

Only one run's log is included here for reference. The median run is the one added to the general record table.

```
##  #   experiment_id                    hostname   total_time  in_mins  roc_auc   epoch  μ_epoch_t  datasets
--  --  -------------------------------  ---------  ----------  -------  --------  -----  ---------  --------
1   27  260311-154259-9fcfcbb7-new2-s11  dlc2gpu09  356.08s     5.93m    0.806860  126    2.83s      8064    
2   14  260311-154251-76f25a24-new2-s11  dlc2gpu17  368.55s     6.14m    0.807486  126    2.93s      8064    
3   9   260310-024152-6a6d610f-new2-s11  dlc2gpu02  369.78s     6.16m    0.807043  126    2.93s      8064    
4   3   260310-021709-025fde5b-new2-s11  dlc2gpu01  383.68s     6.39m    0.806982  139    2.76s      8896    
5   8   260310-022845-d0387e29-new2-s11  dlc2gpu02  385.28s     6.42m    0.807365  178    2.16s      11392   
6   19  260311-154253-6a5558bf-new2-s11  dlc2gpu16  385.95s     6.43m    0.807004  138    2.80s      8832    
7   18  260311-154253-15518150-new2-s11  dlc2gpu16  390.12s     6.50m    0.807858  139    2.81s      8896    
8   25  260311-154258-e53e99c6-new2-s11  dlc2gpu09  400.81s     6.68m    0.808903  142    2.82s      9088    
9   11  260311-154251-2945a8ff-new2-s11  dlc2gpu17  402.36s     6.71m    0.806978  142    2.83s      9088    
10  26  260311-154259-6d70db82-new2-s11  dlc2gpu08  404.51s     6.74m    0.806880  147    2.75s      9408    
11  1   260310-015102-ac58d6c4-new2-s11  dlc2gpu03  407.67s     6.79m    0.806877  142    2.87s      9088    
12  29  260311-154259-dd4a1df0-new2-s11  dlc2gpu08  410.11s     6.84m    0.807495  147    2.79s      9408    
13  17  260311-154252-b40ca92a-new2-s11  dlc2gpu17  416.91s     6.95m    0.806865  149    2.80s      9536    
14  6   260310-022037-eda6d44f-new2-s11  dlc2gpu01  420.91s     7.02m    0.809092  157    2.68s      10048   
15  16  260311-154251-b11a20a1-new2-s11  dlc2gpu07  431.08s     7.18m    0.807047  152    2.84s      9728    
16  20  260311-154253-a787618f-new2-s11  dlc2gpu16  454.12s     7.57m    0.808967  175    2.59s      11200   
17  13  260311-154251-51c419de-new2-s11  dlc2gpu07  462.45s     7.71m    0.806994  175    2.64s      11200   
18  5   260310-021839-e8ed47f9-new2-s11  dlc2gpu03  466.06s     7.77m    0.807612  170    2.74s      10880   
19  21  260311-154254-19816e9c-new2-s11  dlc2gpu07  467.71s     7.80m    0.808167  178    2.63s      11392   
20  23  260311-154258-5bc033ec-new2-s11  dlc2gpu09  469.70s     7.83m    0.807437  177    2.65s      11328   
21  4   260310-021741-67aedfe7-new2-s11  dlc2gpu04  471.34s     7.86m    0.808003  172    2.74s      11008   
22  12  260311-154251-371c392f-new2-s11  dlc2gpu07  471.39s     7.86m    0.809256  172    2.74s      11008   
23  22  260311-154258-1107b5f4-new2-s11  dlc2gpu09  473.04s     7.88m    0.807799  178    2.66s      11392   
24  15  260311-154251-895d6c16-new2-s11  dlc2gpu17  479.34s     7.99m    0.806951  185    2.59s      11840   
25  10  260310-024346-ffbadb38-new2-s11  dlc2gpu01  485.86s     8.10m    0.806891  181    2.68s      11584   
26  7   260310-022551-a084da86-new2-s11  dlc2gpu03  511.07s     8.52m    0.807301  189    2.70s      12096   
27  30  260311-161030-d0c28055-new2-s11  dlc2gpu16  528.15s     8.80m    0.807724  206    2.56s      13184   
28  28  260311-154259-c0035b59-new2-s11  dlc2gpu08  531.39s     8.86m    0.807219  209    2.54s      13376   
29  24  260311-154258-7e973e45-new2-s11  dlc2gpu08  538.02s     8.97m    0.807243  207    2.60s      13248   
30  31  260316-114825-045f6dfa-new2-s11  dlc2gpu12  539.38s     8.99m    0.807643  147    3.67s      9408    
31  2   260310-020202-9f21f9fc-new2-s11  dlc2gpu02  545.97s     9.10m    0.807476  206    2.65s      13184   

                                                    446.09s     7.43m    0.807530  164    2.74s      10482
                                                    55.72s      0.93m    0.000687  25     0.22s      1581
                                                    454.12s     7.57m    0.807365  170    2.74s      10880
```

> **Note:** A later re-run of the same technique produced a faster median of **6.06m** vs the 7.57m reported at record time. It produced the same dataset efficiency while epochs running faster on the cluster.

```
mean               6.14m    165    2.24s      10558     31.56m
std                0.72m    22     0.16s      1416      4.04m
median             6.06m    172    2.24s      11008     32.18m
-----------------  -------  -----  ---------  --------  -------  ---------------
##  #   hostname   in mins  epoch  μ epoch t  datasets  runtime  id-name
--  --  ---------  -------  -----  ---------  --------  -------  ---------------
1   22  dlc2gpu09  4.83m    126    2.30s      8064      24.00m   7e3bef16-rdecay
2   32  dlc2gpu07  5.00m    132    2.27s      8448      25.04m   27811d78-rdecay
3   24  dlc2gpu02  5.06m    126    2.41s      8064      23.60m   59d46a12-rdecay
4   2   dlc2gpu16  5.25m    140    2.25s      8960      28.46m   44e083b2-rdecay
5   15  dlc2gpu16  5.29m    139    2.28s      8896      27.49m   a657db92-rdecay
6   13  dlc2gpu16  5.29m    140    2.27s      8960      27.80m   2bb502b1-rdecay
7   11  dlc2gpu16  5.39m    147    2.20s      9408      29.05m   d5cf6cc0-rdecay
8   3   dlc2gpu16  5.40m    147    2.20s      9408      29.89m   56e102f5-rdecay
9   1   dlc2gpu16  5.46m    150    2.19s      9600      30.44m   221ca6e7-rdecay
10  39  dlc2gpu14  5.61m    147    2.29s      9408      27.16m   0251e5d1-rdecay
11  4   dlc2gpu16  5.63m    157    2.15s      10048     31.51m   ad14e459-rdecay
12  37  dlc2gpu15  5.63m    152    2.22s      9728      28.32m   8028aa52-rdecay
13  41  dlc2gpu06  5.65m    134    2.53s      8576      25.24m   beef2320-rdecay
14  16  dlc2gpu09  5.74m    161    2.14s      10304     29.80m   8d420941-rdecay
15  33  dlc2gpu07  5.84m    164    2.14s      10496     30.65m   d117c72a-rdecay
16  27  dlc2gpu11  5.91m    138    2.57s      8832      27.23m   a9db8828-rdecay
17  12  dlc2gpu16  5.93m    172    2.07s      11008     32.92m   bfcedbec-rdecay
18  23  dlc2gpu09  5.94m    171    2.08s      10944     31.71m   d038806c-rdecay
19  21  dlc2gpu09  6.00m    175    2.06s      11200     32.40m   6a487f98-rdecay
20  18  dlc2gpu09  6.01m    174    2.07s      11136     32.18m   297f85ec-rdecay
21  17  dlc2gpu09  6.06m    174    2.09s      11136     32.26m   d65267a2-rdecay
22  5   dlc2gpu16  6.09m    172    2.12s      11008     34.51m   c807d733-rdecay
23  14  dlc2gpu01  6.09m    175    2.09s      11200     31.93m   a02341bd-rdecay
24  34  dlc2gpu16  6.14m    175    2.11s      11200     32.59m   e8f5d88c-rdecay
25  20  dlc2gpu09  6.28m    187    2.02s      11968     34.38m   3d257711-rdecay
26  36  dlc2gpu14  6.31m    140    2.70s      8960      27.43m   f099d2b1-rdecay
27  40  dlc2gpu05  6.53m    153    2.56s      9792      29.32m   bc5da34c-rdecay
28  25  dlc2gpu09  6.55m    199    1.97s      12736     36.34m   a93f2746-rdecay
29  19  dlc2gpu02  6.62m    178    2.23s      11392     32.37m   4d6bb34f-rdecay
30  38  dlc2gpu01  6.65m    178    2.24s      11392     33.27m   f774ee33-rdecay
31  28  dlc2gpu11  6.67m    170    2.36s      10880     32.70m   06e1f10d-rdecay
32  8   dlc2gpu09  6.71m    178    2.26s      11392     35.32m   c1527be2-rdecay
33  7   dlc2gpu09  6.77m    176    2.31s      11264     35.21m   18085bf6-rdecay
34  9   dlc2gpu09  6.80m    174    2.35s      11136     34.68m   d4f0c952-rdecay
35  31  dlc2gpu11  6.82m    175    2.34s      11200     33.36m   46a5dfb5-rdecay
36  30  dlc2gpu11  6.87m    178    2.32s      11392     34.19m   3c943e6b-rdecay
37  6   dlc2gpu09  7.05m    189    2.24s      12096     37.51m   b5bb105c-rdecay
38  26  dlc2gpu11  7.09m    187    2.27s      11968     35.74m   3af6bc68-rdecay
39  35  dlc2gpu14  7.33m    178    2.47s      11392     33.91m   0e499684-rdecay
40  10  dlc2gpu16  7.75m    222    2.09s      14208     42.18m   50691d63-rdecay
41  29  dlc2gpu11  7.80m    214    2.19s      13696     40.07m   29b8a0b2-rdecay
```
