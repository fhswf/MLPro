.. _target_oa_anomaly_detection:
Anomaly detection
=================

Anomaly detection involves identifying instances that are structurally or dimensionally similar to non-anomalous data but deviate significantly from the normal data distribution or pattern. In real-world problems, anomaly detection helps uncover unusual activities in banking and finance, abnormalities in medical test results, uncommon behavior or sensor readings of machines, defective products in manufacturing lines, or malicious activities in network traffic monitoring. Detecting and analyzing these instances or behaviors is crucial for taking immediate action, preventing future occurrences of undesirable events, and ensuring data quality. Moreover, anomaly detection plays a pivotal role in making unbiased and accurate decisions across various domains.

Anomaly detection techniques can be broadly classified into two categories based on the under- lying principles and methodologies. The two categories are Statistical anomaly detectors and Machine Learning anomaly detectors.

**Types of anomalies**
There are three main types of anomalies- Point anomalies, Contextual anomalies and Collective anomalies.

  (a) Point Anomalies : Type I anomalies or point anomalies are individual data instances that are significantly different from the rest of the dataset. Also known as global anomalies, these do not fit the normal distribution or pattern of the dataset.
  (b) Contextual Anomalies : Type II anomalies or contextual anomalies are data instances that are anomalies only in a particular context or subset of the dataset. Also known as conditional anomalies, these are not necessarily anomalies in the context of the whole dataset but anomalous within a specific context or condition.
  (c) Group Anomalies : Type III anomalies or group anomalies or collective anomalies are anomalous data instances when taken as a group or subset of the dataset. They may or may not be anomalies when considered individually. Also known as group anomalies, these occur when there is a deviation or unexpected relationship or behaviour among a group of data instances from the normal distribution of data.

**Classification of anomaly detectors**
Anomaly detection techniques can be broadly classified into two categories based on the under- lying principles and methodologies. The two categories are Statistical anomaly detectors and Machine Learning anomaly detectors.

  (a) Statistical Anomaly Detectors : Statistical anomaly detectors use statistical methods to find data point deviations from the normal distribution. Common algorithms for this category of anomaly detectors are Z-score, Kernel Density Estimate, and Gaussian Mixture Models (GMM).
  (b) Machine Learning Anomaly Detectors : The anomaly detectors which employ machine learning algorithms by training a model against labelled or unlabeled data to detect anomalies are categorized as machine learning anomaly detectors. In general, based on the type of data and amount of labelled used to train the ML model, ML anomaly detectors can be classified into supervised, semi-supervised and unsupervised, requiring all labelled, some labelled, and all unlabeled datasets respectively. Based on the methodologies used to detect anomalies, the three types can be further classified into cluster-based, information-theoretic, rule-based, deep-learning, etc. Based on whether the data is offline-available or online streaming and the type of learning methods employed, ML anomaly detectors can be categorized into three categories: offline learning, semi-online learning, and online learning.


**Learn more**

.. toctree::
   :maxdepth: 2
   :glob:

   30_anomaly_detection/*


**Cross reference**

- Selected open access papers
- Howtos
- API
