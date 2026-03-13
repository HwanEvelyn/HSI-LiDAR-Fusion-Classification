| Run | Setting | Best Epoch | Val OA | Val AA | Val Kappa | Test OA | Test AA | Test Kappa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 25 | 0.8781 | 0.9090 | 0.8671 | 0.8434 | 0.8726 | 0.8303 |
| hct_bgc_v1 | fusion_layers=1, gate=on | 13 | 0.9170 | 0.9320 | 0.9093 | 0.8596 | 0.8717 | 0.8480 |
| hct_bgc_v1_contrastive | fusion_layers=1, gate=on, contrastive(w=0.1, tau=0.1) | 8 | 0.9187 | 0.9301 | 0.9112 | 0.8412 | 0.8631 | 0.8281 |
| fusion2 | fusion_layers=2, gate=on | 11 | 0.9187 | 0.9345 | 0.9112 | 0.8480 | 0.8715 | 0.8356 |
| fusion3 | fusion_layers=3, gate=on | 12 | 0.9117 | 0.9245 | 0.9034 | 0.8283 | 0.8510 | 0.8145 |
| no_gate | fusion_layers=1, gate=off | 23 | 0.9170 | 0.9304 | 0.9092 | 0.8450 | 0.8713 | 0.8323 |
