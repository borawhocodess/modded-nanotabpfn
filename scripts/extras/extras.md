# Extras — retime on the TabPFN-paper eval set

Time-to-jackpot (cross RF baseline `avg_roc_auc = 0.876071`), 5 reps, 1×L40S.

| variant | jackpot | median (min) | mean | sd | min–max | med epochs | speedup |
|---|---|---|---|---|---|---|---|
| baseline | 5/5 | 119.0 | 118.9 | 0.7 | 118.1–120.2 | 2254 | 1.0× |
| muon | 5/5 | 101.9 | 101.7 | 0.4 | 101.1–102.2 | 1385 | 1.17× |
| carter | 5/5 | 31.6 | 31.6 | 3.4 | 25.8–35.4 | 822 | 3.8× |
| batchedmuon | 5/5 | 22.3 | 22.5 | 3.1 | 17.8–26.2 | 866 | 5.3× |
| residualdecay | 5/5 | 15.5 | 15.9 | 1.3 | 14.2–18.2 | 602 | 7.7× |
| rmsthink | 5/5 | 13.2 | 13.1 | 0.7 | 12.0–14.2 | 528 | 9.0× |
| lawa1wd | 5/5 | 8.6 | 9.3 | 1.3 | 8.0–11.3 | 337 | 13.8× |
| featuregroup | 5/5 | 6.6 | 6.7 | 0.6 | 5.8–7.4 | 264 | 18.0× |
| autohuman | 5/5 | 2.1 | 2.0 | 0.2 | 1.7–2.3 | 139 | 56.7× |
