# Anomaly detection using multiscale signatures

We combine sliding window operations with the signature transform [1] to create a set of insightful features of time series. Then, this set of features is injected in anomaly detection methods, namely, Local Outlier Factor (LOF) and One-Class SVM (OCSVM).

[1] <https://arxiv.org/abs/1603.03788>

# Data

-   Anomalies in maritime data (ship trajectories): <https://avires.dimi.uniud.it/papers/trclust/> (`dataset2.zip`)
-   Unsupervised Anomaly Detection Benchmark: <https://doi.org/10.7910/DVN/OPQMVF> [2]
-   NB: this code can be applied to any dataset of time series.

[2] Goldstein, Markus, 2015, "Unsupervised Anomaly Detection Benchmark", Harvard Dataverse
