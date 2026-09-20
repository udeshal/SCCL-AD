# SCCL-AD : Scale and Correlation Aware Contrastive Learning for Anomaly Detection on Multivariate Time Series
SCCL-AD aims to address the high-dimensional and correlated data challenge and enhance the accuracy and performance of anomaly detection in time series data by explicitly incorporating inter-variable correlations into the representation learning objective and by capturing temporal patterns at multiple scales. The proposed approach, Scale and Correlation-Aware Contrastive Learning for Anomaly Detection (SCCL-AD), applied to multivariate time series and consists of three main components: a multi-scale layer to capture temporal patterns at different scales, a transformation layer with a fixed set of learnable transformations to generate diverse augmented views, and an encoder that maps both original and transformed representations into a shared embedding space. All three components are trained jointly on normal (non-anomalous) data using a correlation-aware contrastive learning objective.

<img width="910" height="508" alt="image" src="https://github.com/user-attachments/assets/82b86aed-701e-47c4-a48b-ba0adddea587" />

# Quickstart

Given a python environment (note: this project is fully tested under python 3.13), install the fllowing dependency:

```
!pip install git+https://github.com/ahstat/affiliation-metrics-py.git
```

Please run the following command to train and test the model:

```
python Launch_Exps.py --config-file $1 --dataset-name $2
```

Note: replace $# with available options

## How to Use
1. When using your own data, please put your data files under [DATA](SCCL-AD/DATA).

2. Create a config file which contains your hyper-parameters under [config_files](SCCL-AD/config_files).  

3. Add your data loader to the function ''load_data'' in the [loader/LoadData.py](SCCL-AD/loader/LoadData.py).
