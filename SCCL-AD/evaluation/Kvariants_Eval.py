import os
import json
import torch
import random
import numpy as np
from loader.LoadData import load_data
from utils import Logger


class KVariantEval:

    def __init__(self, dataset, exp_path, model_configs):
        self.num_cls = dataset.num_cls
        self.data_name = dataset.data_name
        self.model_configs = model_configs
        self._NESTED_FOLDER = exp_path
        self._FOLD_BASE = '_CLS'
        self._RESULTS_FILENAME = 'results.json'
        self._ASSESSMENT_FILENAME = 'assessment_results.json'

    def process_results(self):

        TS_f1s = []
        TS_afff1s = []
        TS_paf1s = []
        TS_aps = []
        TS_aucs = []
        TS_train_times = []
        TS_infer_times = []
        TS_train_mems = []
        TS_test_mems = []

        results = {}

        for i in range(self.num_cls):
            try:
                config_filename = os.path.join(self._NESTED_FOLDER, str(i)+self._FOLD_BASE,
                                               self._RESULTS_FILENAME)
                with open(config_filename, 'r') as fp:
                    variant_scores = json.load(fp)
                    ts_f1 = np.array(variant_scores['TS_F1'])
                    ts_afff1 = np.array(variant_scores['TS_AffF1'])
                    ts_paf1 = np.array(variant_scores['TS_PAF1'])
                    ts_auc = np.array(variant_scores['TS_AUC'])
                    ts_ap = np.array(variant_scores['TS_AP'])
                    ts_train_time = np.array(variant_scores['TS_TrainTimePerEpoch'])
                    ts_infer_time = np.array(variant_scores['TS_InferTime'])
                    ts_train_mem = np.array(variant_scores['TS_TrainPeakMemGB'])
                    ts_test_mem = np.array(variant_scores['TS_TestPeakMemGB'])

                    TS_f1s.append(ts_f1)
                    TS_afff1s.append(ts_afff1)
                    TS_paf1s.append(ts_paf1)
                    TS_aucs.append(ts_auc)
                    TS_aps.append(ts_ap)
                    TS_train_times.append(ts_train_time)
                    TS_infer_times.append(ts_infer_time)
                    TS_train_mems.append(ts_train_mem)
                    TS_test_mems.append(ts_test_mem)

                results['avg_TS_f1_' + str(i)] = ts_f1.mean()
                results['std_TS_f1_' + str(i)] = ts_f1.std()
                results['avg_TS_afff1_' + str(i)] = ts_afff1.mean()
                results['std_TS_afff1_' + str(i)] = ts_afff1.std()
                results['avg_TS_paf1_' + str(i)] = ts_paf1.mean()
                results['std_TS_paf1_' + str(i)] = ts_paf1.std()
                results['avg_TS_ap_' + str(i)] = ts_ap.mean()
                results['std_TS_ap_' + str(i)] = ts_ap.std()
                results['avg_TS_auc_' + str(i)] = ts_auc.mean()
                results['std_TS_auc_' + str(i)] = ts_auc.std()
                results['avg_TS_train_time_per_epoch_' + str(i)] = ts_train_time.mean()
                results['avg_TS_infer_time_' + str(i)] = ts_infer_time.mean()
                results['avg_TS_train_peak_mem_gb_' + str(i)] = ts_train_mem.mean()
                results['avg_TS_test_peak_mem_gb_' + str(i)] = ts_test_mem.mean()
            except Exception as e:
                print(e)

        TS_f1s = np.array(TS_f1s)
        TS_afff1s = np.array(TS_afff1s)
        TS_paf1s = np.array(TS_paf1s)
        TS_aps = np.array(TS_aps)
        TS_aucs = np.array(TS_aucs)
        TS_train_times = np.array(TS_train_times)
        TS_infer_times = np.array(TS_infer_times)
        TS_train_mems = np.array(TS_train_mems)
        TS_test_mems = np.array(TS_test_mems)
        avg_TS_f1 = np.mean(TS_f1s, 0)
        avg_TS_afff1 = np.mean(TS_afff1s, 0)
        avg_TS_paf1 = np.mean(TS_paf1s, 0)
        avg_TS_ap = np.mean(TS_aps, 0)
        avg_TS_auc = np.mean(TS_aucs, 0)
        avg_TS_train_time = np.mean(TS_train_times, 0)
        avg_TS_infer_time = np.mean(TS_infer_times, 0)
        avg_TS_train_mem = np.mean(TS_train_mems, 0)
        avg_TS_test_mem = np.mean(TS_test_mems, 0)
        results['avg_TS_f1_all'] = avg_TS_f1.mean()
        results['std_TS_f1_all'] = avg_TS_f1.std()
        results['avg_TS_afff1_all'] = avg_TS_afff1.mean()
        results['std_TS_afff1_all'] = avg_TS_afff1.std()
        results['avg_TS_paf1_all'] = avg_TS_paf1.mean()
        results['std_TS_paf1_all'] = avg_TS_paf1.std()
        results['avg_TS_ap_all'] = avg_TS_ap.mean()
        results['std_TS_ap_all'] = avg_TS_ap.std()
        results['avg_TS_auc_all'] = avg_TS_auc.mean()
        results['std_TS_auc_all'] = avg_TS_auc.std()
        results['avg_TS_train_time_per_epoch_all'] = avg_TS_train_time.mean()
        results['avg_TS_infer_time_all'] = avg_TS_infer_time.mean()
        results['avg_TS_train_peak_mem_gb_all'] = avg_TS_train_mem.mean()
        results['avg_TS_test_peak_mem_gb_all'] = avg_TS_test_mem.mean()

        with open(os.path.join(self._NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(results, fp,indent=0)

    def risk_assessment(self, experiment_class):

        if not os.path.exists(self._NESTED_FOLDER):
            os.makedirs(self._NESTED_FOLDER)

        for cls in range(self.num_cls):

            folder = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE)
            if not os.path.exists(folder):
                os.makedirs(folder)

            json_results = os.path.join(folder, self._RESULTS_FILENAME)
            if not os.path.exists(json_results):

                self._risk_assessment_helper(cls, 'normal', experiment_class, folder)
            else:
                print(
                    f"File {json_results} already present! Shutting down to prevent loss of previous experiments")
                continue

        self.process_results()

    def _risk_assessment_helper(self, cls, cls_type, experiment_class, exp_path):

        best_config = self.model_configs[0]
        experiment = experiment_class(best_config, exp_path)

        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')
        # logger = None

        val_auc_list, test_auc_list,test_ap_list,test_f1_list,test_afff1_list,test_paf1_list = [], [],[], [], [], []
        train_time_list, infer_time_list, train_mem_list, test_mem_list = [], [], [], []
        num_repeat = best_config['num_repeat']
        saved_results = {}
        # Mitigate bad random initializations
        for i in range(num_repeat):
            torch.cuda.empty_cache()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(i + 41)
            random.seed(i + 41)
            torch.manual_seed(i + 41)
            torch.cuda.manual_seed(i + 41)
            torch.cuda.manual_seed_all(i + 41)
            dataset = load_data(self.data_name, cls, cls_type)
            val_auc, test_auc, test_ap,test_f1,test_afff1,test_paf1,train_time_per_epoch,test_infer_time,train_peak_mem_gb,test_peak_mem_gb,scores,labels = experiment.run_test(dataset,logger)
            print(f'Final training run {i + 1}: {val_auc}, {test_auc,test_ap, test_f1, test_afff1, test_paf1}, '
                  f'train_time/epoch={train_time_per_epoch:.4f}s, infer_time={test_infer_time:.4f}s, '
                  f'train_mem={train_peak_mem_gb:.3f}GB, test_mem={test_peak_mem_gb:.3f}GB')
            saved_results['scores_'+str(i)] = scores.tolist()
            saved_results['labels_' + str(i)] = labels.tolist()
            val_auc_list.append(val_auc)
            test_auc_list.append(test_auc)
            test_ap_list.append(test_ap)
            test_f1_list.append(test_f1)
            test_afff1_list.append(test_afff1)
            test_paf1_list.append(test_paf1)
            train_time_list.append(train_time_per_epoch)
            infer_time_list.append(test_infer_time)
            train_mem_list.append(train_peak_mem_gb)
            test_mem_list.append(test_peak_mem_gb)
        if best_config['save_scores']:
            save_path = os.path.join(self._NESTED_FOLDER, str(cls)+self._FOLD_BASE,'scores_labels.json')
            json.dump(saved_results, open(save_path, 'w'))
            # saved_results = json.load(open(save_path))
        if logger is not None:
            logger.log(
                'End of Variant:'+cls_type+ str(cls) + ' TS f1: ' + str(test_f1_list)+' TS AffF1: ' + str(test_afff1_list)+' TS PAF1: ' + str(test_paf1_list)+' TS AP: ' + str(test_ap_list)+' TS auc: ' + str(test_auc_list)
                + ' TS train_time/epoch: ' + str(train_time_list) + ' TS infer_time: ' + str(infer_time_list)
                + ' TS train_mem_GB: ' + str(train_mem_list) + ' TS test_mem_GB: ' + str(test_mem_list) )
        print('F1:'+str(np.array(test_f1_list).mean())+' AffF1:'+str(np.array(test_afff1_list).mean())+' PAF1:'+str(np.array(test_paf1_list).mean())+' AUC:'+str(np.array(test_auc_list).mean())
              +' TrainTime/epoch:'+str(np.array(train_time_list).mean())+' InferTime:'+str(np.array(infer_time_list).mean())
              +' TrainMemGB:'+str(np.array(train_mem_list).mean())+' TestMemGB:'+str(np.array(test_mem_list).mean()))
        with open(os.path.join(exp_path, self._RESULTS_FILENAME), 'w') as fp:
            json.dump({'best_config': best_config, 'VAL_AUC': val_auc_list,
                       'TS_F1': test_f1_list,'TS_AffF1': test_afff1_list,'TS_PAF1': test_paf1_list,'TS_AP': test_ap_list,'TS_AUC': test_auc_list,
                       'TS_TrainTimePerEpoch': train_time_list,'TS_InferTime': infer_time_list,
                       'TS_TrainPeakMemGB': train_mem_list,'TS_TestPeakMemGB': test_mem_list}, fp)


