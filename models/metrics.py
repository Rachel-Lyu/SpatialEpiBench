import torch
import torch.nn as nn
import torch.nn.functional as F

#——————————————————————————losses

def get_loss(loss_name = 'mse'):
    """
    Retrieves the specified loss function based on the input loss name. It supports mean squared error (MSE),
    a standardized loss (stan), an epidemic-collaboration specific loss (epi_cola), and cross-entropy loss.

    Parameters
    ----------
    loss_name : str, optional
        Name of the loss function to retrieve. Default is 'mse'.

    Returns
    -------
    callable
        The corresponding loss function as specified by loss_name.
    """
    loss_name = loss_name.lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'mse_filtered':
        return mse_filtered_loss
    elif loss_name == 'mae_filtered':
        return mae_filtered_loss
    elif loss_name == 'stan':
        return stan_loss
    elif loss_name == 'epi_cola':
        return epi_cola_loss
    elif loss_name == 'ce':
        return cross_entropy_loss

def crps_ensemble(samples, target):
    """
    samples: (S, ...) ensemble / bootstrap forecasts
    target:  (...)   observations
    returns scalar CRPS
    """
    S = samples.shape[0]
    samples = samples.reshape(S, -1)        # (S, M)
    target  = target.reshape(1, -1)        # (1, M)
    term1 = (samples - target).abs().mean(dim=0)  # E|X - y|
    diff  = samples.unsqueeze(0) - samples.unsqueeze(1)  # (S, S, M)
    term2 = diff.abs().mean(dim=(0, 1))              # E|X - X'|
    crps = term1 - 0.5 * term2                      # (M,)
    return crps.mean()

def wis_from_quantiles(q, target, alphas):
    """
    q: (1 + 2K, ...) quantiles in order:
       [0.5, alpha1/2, 1-alpha1/2, alpha2/2, 1-alpha2/2, ...]
    target: (...) observations
    alphas: (alpha1, alpha2, ..., alphaK)
    returns scalar WIS
    """
    y = target
    median = q[0]
    ae = (median - y).abs()
    wis_num = 0.5 * ae
    for k, alpha in enumerate(alphas):
        l = q[1 + 2*k]
        u = q[2 + 2*k]
        width = u - l
        below = (y < l).float()
        above = (y > u).float()
        iscore = width + (2.0/alpha) * (l - y) * below + (2.0/alpha) * (y - u) * above
        wis_num = wis_num + 0.5 * alpha * iscore
    denom = 0.5 + len(alphas)
    wis = wis_num / denom
    return wis.mean()

def mse_filtered_loss(pred, target, iqr_mult=1.5):
    pred = pred.reshape_as(target).float()
    target = target.float()
    if torch.all(target == 0): 
        return torch.tensor(0., device=target.device)

    mask = torch.isfinite(target) & (target != 0)
    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    return torch.mean((pred[mask] - target[mask]) ** 2) if mask.any() else torch.tensor(0., device=target.device)

def mae_filtered_loss(pred, target, iqr_mult=1.5):
    pred = pred.reshape_as(target).float()
    target = target.float()
    if torch.all(target == 0): 
        return torch.tensor(0., device=target.device)

    mask = torch.isfinite(target) & (target != 0)
    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    return torch.mean(torch.abs(pred[mask] - target[mask])) if mask.any() else torch.tensor(0., device=target.device)

def _safe_log1p(x):
    return torch.log1p(torch.clamp(x, min=0.0))

def stan_loss(output, label, scale=0.5):
    """
    Calculates a combined mean squared error loss on predicted and physically informed predicted values,
    scaled by a given factor.

    Parameters
    ----------
    output : tuple of torch.Tensor
        The predicted values and the physically informed predicted values.
    label : torch.Tensor
        The ground truth values.
    scale : float, optional
        Scaling factor for the physical informed loss component. Default: 0.5.

    Returns
    -------
    torch.Tensor
        The calculated total loss as a scalar tensor.
    """
    pred_IR, pred_phy_IR = output
    mse = nn.MSELoss()
    total_loss = mse(pred_IR, label) + scale*mse(pred_phy_IR, label)
    return total_loss

def epi_cola_loss(output, label, scale=0.5):
    """
    Calculates a combined L1 and mean squared error loss on the output and an epidemiological output,
    scaled by a given factor.

    Parameters
    ----------
    output : tuple of torch.Tensor
        The primary model output and the epidemiological model output.
    label : torch.Tensor
        The ground truth values.
    scale : float, optional
        Scaling factor for the epidemiological loss component. Default: 0.5.

    Returns
    -------
    torch.Tensor
        The calculated total loss as a scalar tensor.
    """
    output, epi_output = output
    mse = nn.MSELoss()
    total_loss = F.l1_loss(output, label) + scale*mse(epi_output, label)
    return total_loss

def cross_entropy_loss(output, label):
    """
    Computes the cross-entropy loss between the logits and labels, adjusting the label tensor to fit the logits dimensions.

    Parameters
    ----------
    output : torch.Tensor
        The logits from the model.
    label : torch.Tensor
        The ground truth labels, scaled to match the number of classes based on output dimensions.

    Returns
    -------
    torch.Tensor
        The cross-entropy loss as a scalar tensor.
    """
    label = (((label-label.min())/(label.max()-label.min()+1))*output.shape[-1]).int()
    ce = nn.CrossEntropyLoss()
    return ce(output.float().view(-1, output.shape[-1]), label.long().view(-1))


#--------------------metrics------------------
def get_MSE(pred, target):
    """
    Calculates the Mean Absolute Error (MAE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The MAE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    mse_loss = nn.MSELoss(reduction='mean')
    return mse_loss(pred, target)

def get_MAE(pred, target):
    """
    Calculates the Mean Absolute Error (MAE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The MAE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    return torch.mean(torch.absolute(pred - target))

def get_RMSE(pred, target):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.

    Returns
    -------
    torch.Tensor
        The RMSE value as a scalar tensor.
    """
    pred = pred.reshape(target.shape)
    mse_loss = nn.MSELoss(reduction='mean')
    return torch.sqrt(mse_loss(pred, target))

def get_ACC(pred, target):
    """
    Calculates the accuracy of predictions by comparing them to the targets.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    target : torch.Tensor
        True labels.

    Returns
    -------
    torch.Tensor
        The accuracy as a scalar tensor.
    """
    result = pred.eq(target).sum()/len(pred.reshape(-1))
    return result

def get_MSE_filtered(pred, target, iqr_mult=1.5, exclude_zeros=True):
    """Mean Squared Error after filtering target zeros/outliers."""
    pred = pred.reshape_as(target).float()
    target = target.float()

    mask = torch.isfinite(target)
    if exclude_zeros:
        mask &= (target != 0)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    err2 = (pred[mask] - target[mask]) ** 2
    return err2.mean()

def get_RMSE_filtered(pred, target, iqr_mult=1.5, exclude_zeros=True):
    """Root Mean Squared Error after filtering target zeros/outliers."""
    mse = get_MSE_filtered(pred, target, iqr_mult, exclude_zeros)
    return torch.sqrt(mse)

def get_MAE_filtered(pred, target, iqr_mult=1.5, exclude_zeros=True):
    """Mean Absolute Error after filtering target zeros/outliers."""
    pred = pred.reshape_as(target).float()
    target = target.float()

    mask = torch.isfinite(target)
    if exclude_zeros:
        mask &= (target != 0)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    err = (pred[mask] - target[mask]).abs()
    return err.mean()

def get_medAE(pred, target, iqr_mult=1.5, exclude_zeros=True):
    """Median Absolute Error after filtering target zeros/outliers."""
    pred = pred.reshape_as(target).float()
    target = target.float()

    mask = torch.isfinite(target)
    if exclude_zeros:
        mask &= (target != 0)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    err = (pred[mask] - target[mask]).abs()
    return err.median()

def get_medSE(pred, target, iqr_mult=1.5, exclude_zeros=True):
    """Median Squared Error after filtering target zeros/outliers."""
    pred = pred.reshape_as(target).float()
    target = target.float()

    mask = torch.isfinite(target)
    if exclude_zeros:
        mask &= (target != 0)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    v = target[mask]
    q1, q3 = torch.quantile(v, 0.25), torch.quantile(v, 0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr if iqr != 0 else q1
    upper = q3 + iqr_mult * iqr if iqr != 0 else q3
    mask &= (target >= lower) & (target <= upper)
    if mask.sum() == 0:
        return torch.tensor(float('nan'), device=target.device)

    err2 = (pred[mask] - target[mask]) ** 2
    return err2.median()
