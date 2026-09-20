from pathlib import Path
import json
import yaml
import pickle
import numpy as np
from datetime import timedelta
from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve

def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")


class Logger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def log(self, str):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(str + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

def format_time(avg_time):
    avg_time = timedelta(seconds=avg_time)
    total_seconds = int(avg_time.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{str(avg_time.microseconds)[:3]}"


def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')

    # precision, recall, thresholds = precision_recall_curve(target, score)
    # numerator = 2 * recall * precision
    # denom = recall + precision
    # f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    # f1 = np.max(f1_scores)
    return f1


def compute_affiliated_f1(target, score):
    """
    Affiliation-based F1 for time series anomaly detection
    (Huet, Navarro & Rossi, "Local Evaluation of Time Series Anomaly Detection", KDD 2022).

    Unlike point-wise F1, this measures how close predicted anomalous events are
    to true anomalous events (and vice versa), which is more informative than
    exact point overlap for time series with contiguous anomaly segments.

    Requires the official reference implementation:
        pip install affiliation-metrics-py
    (repo: https://github.com/ahstat/affiliation-metrics-py)

    target: 1D array-like of binary ground-truth labels (0=normal, 1=anomaly), ordered in time
    score:  1D array-like of anomaly scores (same thresholding convention as compute_pre_recall_f1)
    """
    try:
        from affiliation.generics import convert_vector_to_events
        from affiliation.metrics import pr_from_events
    except ImportError as e:
        raise ImportError(
            "compute_affiliated_f1 requires the 'affiliation-metrics-py' package. "
            "Install it with: pip install affiliation-metrics-py"
        ) from e

    target = np.asarray(target).astype(int)
    score = np.asarray(score)

    # Use the same percentile-based thresholding as compute_pre_recall_f1, so both
    # metrics are computed on the same predicted-anomaly set for a fair comparison.
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score), dtype=int)
    pred[score > threshold] = 1

    # Affiliation metrics are undefined if either side has no events at all.
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(target)
    Trange = (0, len(target))

    result = pr_from_events(events_pred, events_gt, Trange)
    precision = result['precision']
    recall = result['recall']

    if precision is None or recall is None or (precision + recall) == 0:
        return 0.0

    aff_f1 = 2 * precision * recall / (precision + recall)
    return aff_f1


def compute_point_adjusted_f1(target, score):
    """
    Point-adjusted F1 for time series anomaly detection
    (Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder for
    Seasonal KPIs in Web Applications", WWW 2018).

    Rule: for each ground-truth anomalous segment, if at least one point inside it
    is predicted as anomalous, every point in that segment is counted as a true
    positive (even if the rest of the segment was missed). Predictions outside
    ground-truth segments are scored normally (no adjustment).

    Note: this metric is known to be easy to inflate (see Kim et al., AAAI 2022) —
    it's included here alongside point-wise and affiliation F1 for comparison,
    not as a replacement for them.

    target: 1D array-like of binary ground-truth labels (0=normal, 1=anomaly), ordered in time
    score:  1D array-like of anomaly scores (same thresholding convention as compute_pre_recall_f1)
    """
    target = np.asarray(target).astype(int)
    score = np.asarray(score)

    # Same percentile-based thresholding as compute_pre_recall_f1 / compute_affiliated_f1,
    # so all three F1 variants are computed on the same predicted-anomaly set.
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(target), dtype=int)
    pred[score > threshold] = 1

    adjusted_pred = pred.copy()

    # Find contiguous ground-truth anomalous segments and adjust predictions within them.
    diff = np.diff(np.concatenate(([0], target, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]  # exclusive end

    for s, e in zip(starts, ends):
        if pred[s:e].any():
            adjusted_pred[s:e] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(target, adjusted_pred, average='binary')
    return f1


class EarlyStopper:

    def stop(self, epoch, val_loss, val_auc=None,  test_loss=None, test_auc=None, test_ap=None,test_f1=None,test_afff1=None,test_paf1=None,test_infer_time=None,test_peak_mem_gb=None, train_loss=None,score=None,target=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return  self.train_loss, self.val_loss,self.val_auc,self.test_loss,self.test_auc,self.test_ap,self.test_f1,self.test_afff1,self.test_paf1,self.test_infer_time,self.test_peak_mem_gb, self.best_epoch,self.score,self.target

class Patience(EarlyStopper):

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=10, use_train_loss=True):
        self.local_val_optimum = float("inf")
        self.use_train_loss = use_train_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss= None
        self.val_loss, self.val_auc, = None, None
        self.test_loss, self.test_auc,self.test_ap,self.test_f1,self.test_afff1,self.test_paf1 = None, None,None, None, None, None
        self.test_infer_time, self.test_peak_mem_gb = None, None
        self.score, self.target = None, None

    def stop(self, epoch, val_loss, val_auc=None, test_loss=None, test_auc=None, test_ap=None,test_f1=None,test_afff1=None,test_paf1=None,test_infer_time=None,test_peak_mem_gb=None,train_loss=None,score=None,target=None):
        if self.use_train_loss:
            if train_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = train_loss
                self.best_epoch = epoch
                self.train_loss= train_loss
                self.val_loss, self.val_auc= val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap,self.test_f1,self.test_afff1,self.test_paf1\
                    = test_loss, test_auc, test_ap,test_f1,test_afff1,test_paf1
                self.test_infer_time, self.test_peak_mem_gb = test_infer_time, test_peak_mem_gb
                self.score, self.target = score,target
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss= train_loss
                self.val_loss, self.val_auc = val_loss, val_auc
                self.test_loss, self.test_auc, self.test_ap,self.test_f1,self.test_afff1,self.test_paf1\
                    = test_loss, test_auc, test_ap,test_f1,test_afff1,test_paf1
                self.test_infer_time, self.test_peak_mem_gb = test_infer_time, test_peak_mem_gb
                self.score, self.target = score, target
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
