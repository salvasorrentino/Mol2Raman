import torch
import numpy as np
import torch.nn.functional as F
from scipy import signal
from Scripts.utils_model.utils_metrics import count_matched_peaks
import pickle
from typing import Union


class RMSELoss(torch.nn.Module):
    def __init__(self, compute=False):
        self.compute = compute
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = torch.nn.MSELoss()
        if self.compute:
            with open(r"models\scalers\min_max_scaler.pickle",'rb') as f:
                scaler = pickle.load(f)
            x_scaled = scaler.inverse_transform(x.reshape(-1, 1600).cpu().detach().numpy()).reshape(-1)
            y_scaled = scaler.inverse_transform(y.reshape(-1, 1600).cpu().detach().numpy()).reshape(-1)
            loss = torch.sqrt(criterion(torch.Tensor(x_scaled), 
                                        torch.Tensor(y_scaled)))
        else:
            loss = torch.sqrt(criterion(x.reshape(-1), y))
        return loss
    

class BinaryWeightedRMSELoss(torch.nn.Module):

    def __init__(self, Ts=0.4):
        self.Ts = Ts
        super(BinaryWeightedRMSELoss, self).__init__()

    def forward(self, x, y):
        y_pred = x.reshape(-1)
        y_true = y.reshape(-1)
        mse_loss = torch.nn.MSELoss(reduction='none')(y_pred, y_true)
        weight = torch.where(y_true < self.Ts, torch.tensor(0.5), torch.tensor(1.5))
        custom_weighted_loss = weight * mse_loss
        return torch.sqrt(torch.mean(custom_weighted_loss))
    

class WeightedRMSELoss(torch.nn.Module):

    def __init__(self,                  
                 Wtp: float = 8,
                 Wfp: float = 3,
                 Wfn: float = 5,
                 Wtn: float = 1,
                 Ts: Union[str, float] = 0.5,
                 **kwargs):
      
        super(WeightedRMSELoss, self).__init__()
        self.Wtp = Wtp
        self.Wfp = Wfp
        self.Wtn = Wtn
        self.Wfn = Wfn
        if isinstance(Ts, str): 
            assert Ts == 'scipy', "Ts must be either scipy or float"
            self.kwargs = kwargs
        self.Ts = Ts

    def forward(self, x, y):

        y_pred = x.reshape(-1)
        y_true = y.reshape(-1)

        if isinstance(self.Ts, float):
            below_ts_true_mask = y_true <= self.Ts
            below_ts_pred_mask = y_pred <= self.Ts
        else:

            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()

            pred_peaks = signal.find_peaks(y_pred_np, **self.kwargs)
            below_ts_pred_mask_np = np.zeros_like(y_pred_np, dtype=bool)
            below_ts_pred_mask_np[pred_peaks[0]] = False
            below_ts_pred_mask = torch.tensor(below_ts_pred_mask_np, dtype=torch.float64, device='cuda')

            true_peaks = signal.find_peaks(y_true_np, **self.kwargs)
            below_ts_true_mask_np = np.zeros_like(y_true_np, dtype=bool)
            below_ts_true_mask_np[true_peaks[0]] = False
            below_ts_true_mask = torch.tensor(below_ts_true_mask_np, dtype=torch.float64, device='cuda')
        
        t1 = torch.where(below_ts_true_mask & below_ts_pred_mask, self.Wtn, torch.tensor(0.0))
        t2 = torch.where(below_ts_true_mask & ~below_ts_pred_mask, self.Wfp, torch.tensor(0.0))
        t3 = torch.where(~below_ts_true_mask & ~below_ts_pred_mask, self.Wtp, torch.tensor(0.0))
        t4 = torch.where(~below_ts_true_mask & below_ts_pred_mask, self.Wfn, torch.tensor(0.0))
      
        loss = torch.nn.MSELoss(reduction='none')(y_pred, y_true)
        
        weighted_losses = t1 * loss + t2 * loss + t3 * loss + t4 * loss

        sample_losses = torch.mean(weighted_losses)
        
        return torch.sqrt(sample_losses)
    

class PeakAwareLoss(torch.nn.Module):

    def __init__(self, Ts=.4):
        self.Ts = Ts
        super(PeakAwareLoss, self).__init__()

    def forward(self, x, y):
        y_pred = x.reshape(-1)
        y_true = y.reshape(-1)
        mse_loss = torch.nn.MSELoss(reduction='none')(y_pred, y_true)
        weight = torch.where(y_true > self.Ts, torch.tensor(1), torch.tensor(0))
        custom_weighted_loss = weight * mse_loss
        return torch.sqrt(torch.mean(custom_weighted_loss))


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.9, gamma=10, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction}\n Supported "
                f"reduction modes: 'none', 'mean', 'sum'"
            )
        return loss


class HuberLoss(torch.nn.Module):
    def __init__(self, delta=2):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, inputs, targets):

        loss = torch.nn.HuberLoss(reduction='mean', delta=self.delta)(inputs, targets)
        return loss


class SID(torch.nn.Module):
    def __init__(self, n_data_points, threshold: float = 1e-10, eps: float = 1e-8,
                 torch_device: str = 'cpu', **kwargs):
        super(SID, self).__init__()
        self.threshold = threshold
        self.eps = eps
        self.torch_device = torch_device
        self.kwargs = kwargs
        self.n_data_points = n_data_points

    def forward(self, model_spectra, target_spectra):

        target_spectra = target_spectra.reshape(-1, self.n_data_points)
        model_spectra = model_spectra.reshape(-1, self.n_data_points)
        nan_mask = torch.isnan(target_spectra) + torch.isnan(model_spectra)
        nan_mask = nan_mask.to(device=self.torch_device)
        zero_sub = torch.zeros_like(target_spectra, device=self.torch_device)
        model_spectra = model_spectra.to(self.torch_device)
        model_spectra[model_spectra < self.threshold] = self.threshold
        sum_model_spectra = torch.sum(torch.where(nan_mask, zero_sub, model_spectra), axis=1)
        sum_model_spectra = torch.unsqueeze(sum_model_spectra, axis=1)
        model_spectra = torch.div(model_spectra, sum_model_spectra)
        # calculate loss value
        if not isinstance(target_spectra, torch.Tensor):
            target_spectra = torch.tensor(target_spectra)
        target_spectra = target_spectra.to(self.torch_device)
        loss = torch.ones_like(target_spectra)
        loss = loss.to(self.torch_device)
        target_spectra[nan_mask] = 1
        model_spectra[nan_mask] = 1
        model_spectra = model_spectra.clamp(0.000001)
        target_spectra = target_spectra.clamp(0.000001)
        loss = torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra) \
               + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
        loss[nan_mask] = 0
        loss = torch.sum(loss, axis=1)
        mean_loss = torch.mean(loss)
        return mean_loss


class ModifiedRMSELossPeakEmphasis(torch.nn.Module):
    def __init__(self, peak_indices: float , distance: float = 10, emphasis_factor: float = 10, **kwargs):

        super(ModifiedRMSELossPeakEmphasis, self).__init__()
        self.emphasis_factor = emphasis_factor
        self.distance = distance
        self.peak_indices = peak_indices

    def forward(self, pred_spectrum, true_spectrum):
        device = true_spectrum.device
        n = true_spectrum.numel()
        weights = torch.ones(n, device=device)

        for peak in self.peak_indices:
            start = max(0, peak - self.distance // 2)
            end = min(n, peak + self.distance // 2)
            weights[start:end] *= self.emphasis_factor

        squared_errors = weights * (true_spectrum - pred_spectrum) ** 2
        loss = torch.sqrt(torch.mean(squared_errors))

        return loss


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self, width: float = 0.1):
        super(CosineSimilarityLoss, self).__init__()
        self.width=width

    def forward(self, pred_spectrum, true_spectrum):
        gaussian_filter = torch.exp(-torch.linspace(-1, 1, steps=true_spectrum.size(0)) ** 2 / (2 * self.width ** 2)).to(
            true_spectrum.device)

        true_spectrum_filtered = true_spectrum * gaussian_filter
        pred_spectrum_filtered = pred_spectrum * gaussian_filter

        dot_product = torch.sum(true_spectrum_filtered * pred_spectrum_filtered)
        norm_true = torch.sqrt(torch.sum(true_spectrum_filtered ** 2))
        norm_pred = torch.sqrt(torch.sum(pred_spectrum_filtered ** 2))
        cosine_similarity = dot_product / (norm_true * norm_pred + 1e-8)  # Adding epsilon to avoid division by zero

        # Loss is 1 - cosine similarity to emphasize differences
        loss = 1 - cosine_similarity

        return loss


class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, pred, real):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(real, pred))
        return loss


class F1Loss(torch.nn.Module):
    def __init__(self, tolerance=3, prominence=0.1):
        super(F1Loss, self).__init__()
        self.tolerance = tolerance
        self.prominence = prominence

    def forward(self, x, y):
        y_pred = x.reshape(-1)
        y_true = y.reshape(-1)
        y_true_pd = y_true.detach().cpu().numpy()
        y_pred_pd = y_pred.detach().cpu().numpy()
        lst_f1 = []

        #for arr_pred, arr_true in zip(y_pred_pd, y_true_pd):

        # arr_pred = data_s.raman_pred
        # arr_true = data_s.RAMAN_SPECTRUM_1

        mask, _ = signal.find_peaks(y_pred_pd, prominence=self.prominence)
        mask = mask.tolist()
        #raman_pred_mask_int = apply_mask(y_pred, mask)

        mask_true, _ = signal.find_peaks(y_true_pd)
        mask_true = mask_true.tolist()
        #yy_true = apply_mask(y_true, mask_true)

        # tp = count_matched_peaks(yy_true, raman_pred_mask_int, tolerance)
        tp = count_matched_peaks(mask_true, mask, self.tolerance)
        fp = len(mask) - tp
        fn = len(mask_true) - tp

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (fn + tp) != 0:
            recall = tp / (fn + tp)
        else:
            recall = 0

        if tp != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        # lst_f1.append(f1)
        return torch.tensor(f1)