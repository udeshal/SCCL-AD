import pandas as pd
from scipy import io
import numpy as np
from .PreprocessSMAP import *
from .PreprocessMSL import *
from .PreProSMAP_MSL import *
from .preprocess_psm import *
from .preprocess_swat import *
from .preprocess_wadi import *

def train_test_split(inliers,outliers):
    num_split = len(inliers) // 2
    train_data = inliers[:num_split]
    train_label = np.zeros(num_split)
    test_data = np.concatenate([inliers[num_split:],outliers],0)

    test_label = np.zeros(test_data.shape[0])
    test_label[num_split:]=1
    return train_data, train_label, test_data, test_label

def Thyroid_train_test_split(path):
    data = io.loadmat(path+"thyroid/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    return train_data, train_label, test_data, test_label

def Arrhythmia_train_test_split(path):
    data = io.loadmat(path+"arrhythmia/arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    inliers = samples[labels == 0]  # 452 norm
    outliers = samples[labels == 1]  # 66 anom

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    return train_data, train_label, test_data, test_label



def KDD_train_test_split(path):
    samples, labels, continual_idx = KDD_preprocessing(path)
    inliers = samples[labels == 0]  # attack: 396743
    outliers = samples[labels == 1]  # norm: 97278
    idx_perm = np.random.permutation(inliers.shape[0])
    inliers = inliers[idx_perm]

    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
    train_data, test_data= norm_kdd_data(train_data, test_data, continual_idx)

    return train_data, train_label, test_data, test_label


def KDDRev_train_test_split(path):
    samples, labels, continual_idx = KDD_preprocessing(path)

    inliers = samples[labels == 1]  # norm: 97278
    outliers = samples[labels == 0]  # attack: 396743

    random_cut = np.random.permutation(len(outliers))[:24319]
    outliers = outliers[random_cut]  # attack:24319

    idx_perm = np.random.permutation(inliers.shape[0])
    inliers = inliers[idx_perm]

    train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
    train_data, test_data= norm_kdd_data(train_data, test_data, continual_idx)

    return train_data, train_label, test_data, test_label


def KDD_preprocessing(path):
    file_names = [path+"kddcup.data_10_percent.gz",path+"kddcup.names"]

    column_name = pd.read_csv(file_names[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
    column_name.loc[column_name.shape[0]] = ['status', ' symbolic.']
    data = pd.read_csv(file_names[0], header=None, names=column_name['f_names'].values)
    data_symbolic = column_name[column_name['f_types'].str.contains('symbolic.')]
    data_continuous = column_name[column_name['f_types'].str.contains('continuous.')]
    samples = pd.get_dummies(data.iloc[:, :-1], columns=data_symbolic['f_names'][:-1])

    sample_keys = samples.keys()
    continuous_idx = []
    for cont_idx in data_continuous['f_names']:
        continuous_idx.append(sample_keys.get_loc(cont_idx))

    labels = np.where(data['status'] == 'normal.', 1, 0)
    return np.array(samples), np.array(labels), continuous_idx


def norm_kdd_data(train_data, test_data, continuous_idx):
    symbolic_idx = np.delete(np.arange(train_data.shape[1]), continuous_idx)
    mu = np.mean(train_data[:, continuous_idx],0,keepdims=True)
    std = np.std(train_data[:, continuous_idx],0,keepdims=True)
    std[std == 0] = 1

    train_continual = (train_data[:, continuous_idx]-mu)/std
    train_normalized = np.concatenate([train_data[:, symbolic_idx], train_continual], 1)
    test_continual = (test_data[:, continuous_idx]-mu)/std
    test_normalized = np.concatenate([test_data[:, symbolic_idx], test_continual], 1)

    return train_normalized, test_normalized

def smd_train_test_split(path):
    print('Path : ', path)
    path = path + 'SMD-Mac3.csv'
    data = pd.read_csv(path, header=1)
    
    # Split features and labels
    samples = data.iloc[:, :-1].values  # All columns except the last
    labels = data.iloc[:, -1].values     # Last column only

    inliers = samples[labels == 0]  # 3679 norm
    outliers = samples[labels == 1]  # 93 anom

    print('smd FULL Inliers: ', inliers.shape)
    print('smd outliers: ', outliers.shape)

    train_data, train_label, test_data, test_label=train_test_split(inliers,outliers)
    print('smd Train Data Shape ', train_data.shape)
    print('smd Train Label Shape: ', train_label.shape)
    print('smd Test Data Shape: ', test_data.shape)
    print('smd Test Label Shape: ', test_label.shape)
    return train_data, train_label, test_data, test_label

def smap_train_test_split(base_dir):
    #base_dir = '/content/drive/MyDrive/Udesha/DATASET/SMAP/';
    train_data,test_data,test_label,train_label = preprocess_smap_msl_trim(base_dir)
    
    #train_dir = os.path.join(base_dir, 'train')
    #test_dir = os.path.join(base_dir, 'test')
    #label_csv = os.path.join(base_dir, 'labeled_anomalies.csv')

    # Step 1: Load and stack train/test
    #stacked_train, train_channels = load_and_stack_npy(train_dir)
    #stacked_test, test_channels = load_and_stack_npy(test_dir)

    # Step 2: Generate stacked anomaly label array
    #test_labels = generate_anomaly_labels(test_channels, test_dir, label_csv)
    #train_len = stacked_train.shape[0] # Renamed from len
    #train_labels = np.zeros(train_len, dtype=int) # Use train_len

    print('SMAP Train Data : ', train_data.shape)
    print('SMAP Test Data : ', test_data.shape)
    print('SMAP Test Labels : ', test_label.shape)
    print('SMAP Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label

def msl_train_test_split(base_dir):
    train_data,test_data,test_label,train_label = preprocess_msl_trim(base_dir)    
    # --- Update this path to your dataset root ---
    #smap_base_dir = '/content/drive/MyDrive/Udesha/DATASET/MSL/'
    #train_dir = os.path.join(base_dir, 'train')
    #test_dir = os.path.join(base_dir, 'test')
    #label_csv = os.path.join(base_dir, 'labeled_anomalies.csv')

    # Step 1: Load and stack train/test
    #stacked_train, train_channels = load_and_stack_npy(train_dir)
    #stacked_test, test_channels = load_and_stack_npy(test_dir)

    # Step 2: Generate stacked anomaly label array
    #test_labels = generate_anomaly_labels(test_channels, test_dir, label_csv)
    #train_len = stacked_train.shape[0] # Renamed from len
    #train_labels = np.zeros(train_len, dtype=int) # Use train_len

    print('MSL Train Data : ', train_data.shape)
    print('MSL Test Data : ', test_data.shape)
    print('MSL Test Labels : ', test_label.shape)
    print('MSL Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label


def psm_train_test_split(base_dir):
    train_data,train_label, test_data,test_label = psm_train_test_load(base_dir)

    print('PSM Train Data : ', train_data.shape)
    print('PSM Test Data : ', test_data.shape)
    print('PSM Test Labels : ', test_label.shape)
    print('PSM Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label

def swat_train_test_split(base_dir):
    train_data,train_label, test_data,test_label = swat_train_test_load(base_dir)

    print('SWAT Train Data : ', train_data.shape)
    print('SWAT Test Data : ', test_data.shape)
    print('SWAT Test Labels : ', test_label.shape)
    print('SWAT Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label

def wadi_train_test_split(base_dir):
    train_data,train_label, test_data,test_label = wadi_train_test_load(base_dir)

    print('Wadi Train Data : ', train_data.shape)
    print('Wadi Test Data : ', test_data.shape)
    print('Wadi Test Labels : ', test_label.shape)
    print('Wadi Train labels : ', train_label.shape)
    return train_data, train_label, test_data, test_label
