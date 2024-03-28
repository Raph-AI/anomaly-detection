# Anomaly detection using multiscale signatures

We combine sliding window operations with the signature transform [1] to create a set of insightful features of time series. Then, this set of features is injected in anomaly detection methods, namely, Local Outlier Factor (LOF) and One-Class SVM (OCSVM).

[1] <https://arxiv.org/abs/1603.03788>

# Data

-   Anomalies in maritime data (ship trajectories): <https://avires.dimi.uniud.it/papers/trclust/> (`dataset2.zip`)
-   NB: this code can be applied to any dataset of time series.
