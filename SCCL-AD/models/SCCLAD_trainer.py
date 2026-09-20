import time
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from utils import compute_pre_recall_f1,compute_affiliated_f1,compute_point_adjusted_f1,format_time
class SCCLAD_trainer:

    def __init__(self, model, loss_function,device='cuda'):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def _train(self,train_loader, optimizer):

        self.model.train()

        use_cuda_timing = self.device.type == 'cuda'
        if use_cuda_timing:
            torch.cuda.reset_peak_memory_stats(self.device)

        loss_all = 0
        for data in train_loader:
            try:
                samples, _ = data
            except:
                samples = data

            #print('Sample :', samples.shape)
            z = self.model(samples)

            loss = self.loss_fun(z, samples) # Corrected: pass current batch x instead of x_raw
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            loss_all += loss.sum()

        if use_cuda_timing:
            train_peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        else:
            train_peak_mem_gb = 0.0

        return loss_all.item()/len(train_loader.dataset), train_peak_mem_gb


    def detect_outliers(self, loader,cls):
        model = self.model
        model.eval()

        use_cuda_timing = self.device.type == 'cuda'
        if use_cuda_timing:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        infer_start = time.time()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                try:
                    samples, labels = data
                except:
                    samples = data
                    labels = data.y!=cls
                z= model(samples)
                score = self.loss_fun(z,samples, eval=True)
                loss_in += score[labels == 0].sum()
                loss_out += score[labels == 1].sum()
                target_all.append(labels)
                score_all.append(score)

        if use_cuda_timing:
            torch.cuda.synchronize(self.device)
        infer_time = time.time() - infer_start
        if use_cuda_timing:
            infer_peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        else:
            infer_peak_mem_gb = 0.0

        try:
            score_all = np.concatenate(score_all)
        except:
            score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        afff1 = compute_affiliated_f1(target_all,score_all)
        paf1 = compute_point_adjusted_f1(target_all,score_all)
        ap = average_precision_score(target_all, score_all)
        return auc, ap,f1,afff1,paf1,infer_time,infer_peak_mem_gb,loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum(),score_all,target_all


    def train(self, train_loader,cls = None,max_epochs=100, optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=2):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, val_afff1 = -1, -1, -1
        test_auc, test_f1, test_afff1, test_paf1, test_score = None, None, None, None,None
        test_infer_time, test_peak_mem_gb = None, None
        score,target = None,None

        time_per_epoch = []
        train_peak_mem_per_epoch = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_loss, train_peak_mem_gb = self._train(train_loader, optimizer)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            end = time.time() - start
            time_per_epoch.append(end)
            train_peak_mem_per_epoch.append(train_peak_mem_gb)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                test_auc, test_ap,test_f1,test_afff1,test_paf1,test_infer_time,test_peak_mem_gb, testin_loss,testout_loss,score,target = self.detect_outliers(test_loader,cls)

            if validation_loader is not None:
                val_auc, val_ap,val_f1,val_afff1,val_paf1,val_infer_time,val_peak_mem_gb, valin_loss,valout_loss,_,_ = self.detect_outliers(validation_loader,cls)
                if epoch>5:
                    if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc, test_ap,test_f1,test_afff1,test_paf1,test_infer_time,test_peak_mem_gb,
                                                                        train_loss,score,target):
                        break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss}, TR mem: {train_peak_mem_gb:.3f}GB, VAL loss: {valin_loss,valout_loss}, VL auc: {val_auc} VL ap: {val_ap} VL f1: {val_f1} VL afff1: {val_afff1} VL paf1: {val_paf1} '

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        if early_stopper is not None:
            train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1,test_afff1,test_paf1,test_infer_time,test_peak_mem_gb, best_epoch,score,target \
                = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
                f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1} TS afff1: {test_afff1} TS paf1: {test_paf1} ' \
                f'TS infer_time: {test_infer_time:.4f}s TS peak_mem: {test_peak_mem_gb:.3f}GB'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())
        elapsed = format_time(avg_time_per_epoch)
        train_peak_mem_gb_overall = max(train_peak_mem_per_epoch) if train_peak_mem_per_epoch else 0.0

        summary_msg = f'Avg train time/epoch: {elapsed} ({avg_time_per_epoch:.4f}s), ' \
            f'Peak train mem: {train_peak_mem_gb_overall:.3f}GB, ' \
            f'Test infer time: {test_infer_time:.4f}s, Test peak mem: {test_peak_mem_gb:.3f}GB'
        if logger is not None:
            logger.log(summary_msg)
            print(summary_msg)
        else:
            print(summary_msg)

        return val_loss, val_auc, test_auc, test_ap,test_f1,test_afff1,test_paf1,avg_time_per_epoch,test_infer_time,train_peak_mem_gb_overall,test_peak_mem_gb,score,target