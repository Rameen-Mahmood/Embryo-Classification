# Embryo-Classification
- The folder bench_mk_pred includes the baseline model reproduced from [Gomez et al., 2022](https://arxiv.org/pdf/2203.00531.pdf?fbclid=IwAR0JOnWY3dG3psIUlncAT8R1mYbpBZw3TpfSAODfA_wYmZrU-P3_qeJHluM).
- The folder ml4h_ivf includes our implementation of multi-classification (xception and transformer) and survival analysis with transformer.

- To run the multi-classification:
```
cd ml4h_ivf
python3 main.py
```
Positional and optional arguments are provided to adjust number of epochs, models, batch size, etc.
- To run the survival analysis:
```
cd ml4h_ivf
python3 analysis.py
```
