# Anomaly detection using multiscale signatures

We combine sliding window operations with the signature transform to create a set of insightful features of time series. Then, this set of features is injected in anomaly detection methods, namely, Local Outlier Factor (LOF) and One-Class SVM (OCSVM). For an introduction to the signature method in Machine Learning, see <https://arxiv.org/abs/2206.14674> (Lyons and McLeod, 2024).

# Link to publication

R. Mignot, V. Mange, K. Usevich, M. Clausel, J.-Y. Tourneret and F. Vincent, "Anomaly Detection Using Multiscale Signatures". In: Proceedings EUSIPCO 2024. <https://doi.org/10.23919/EUSIPCO63174.2024.10714963> (pdf [here](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0002757.pdf))

```
@inproceedings{mignot2024anomaly,
author={Mignot, Raphael and Mang{\' e}, Val{\' e}rian and Usevich, Konstantin and Clausel, Marianne and Tourneret, Jean–Yves and Vincent, Fran{\c c}ois},
title={Anomaly Detection Using Multiscale Signatures},
year={2024},
booktitle={Proceedings of the 32nd European Signal Processing Conference (EUSIPCO)}, 
pages={2757--2761},
location={Lyon, France},
doi={10.23919/EUSIPCO63174.2024.10714963}}
```

# Data

-   Anomalies in maritime data (ship trajectories): <https://avires.dimi.uniud.it/papers/trclust/> (we use `dataset2.zip`)
-   Unsupervised Anomaly Detection Benchmark: <https://doi.org/10.7910/DVN/OPQMVF> [2]
-   NB: this code can be applied to any dataset of time series.

[2] Goldstein, Markus, 2015, "Unsupervised Anomaly Detection Benchmark", Harvard Dataverse
