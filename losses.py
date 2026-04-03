import torch

def epe_loss(pred_flow, gt_flow, mask=None):
        diff = pred_flow - gt_flow
        epe = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # sum over 2 flow channels (u,v)
        if mask is not None:
            epe = epe * mask
            return epe.sum() / (mask.sum() + 1e-8)
        return epe.mean()
    
def smoothness_loss(pred_flow, img):
    # pred_flow: (B, S-1, 2, H, W) - paired frames
    # img: (B, S, 3, H, W) - original frames
    # Match temporal dimensions: use first S-1 frames of img to align with flow
    img = img[:, :-1, :, :, :]  # (B, S-1, 3, H, W)
    # Compute spatial gradients: dx (horizontal), dy (vertical)
    dx = torch.abs(pred_flow[:, :, :, :, :-1] - pred_flow[:, :, :, :, 1:])  # (B,S-1,2,H,W-1)
    dy = torch.abs(pred_flow[:, :, :, :-1, :] - pred_flow[:, :, :, 1:, :])  # (B,S-1,2,H-1,W)
    # Compute image-based weights by averaging over RGB channels (dim=2)
    img_dx = torch.abs(img[:, :, :, :, :-1] - img[:, :, :, :, 1:])  # (B,S-1,3,H,W-1)
    img_dy = torch.abs(img[:, :, :, :-1, :] - img[:, :, :, 1:, :])  # (B,S-1,3,H-1,W)
    weights_x = torch.exp(-img_dx.mean(dim=2, keepdim=True))  # (B,S-1,1,H,W-1)
    weights_y = torch.exp(-img_dy.mean(dim=2, keepdim=True))  # (B,S-1,1,H-1,W)
    return (dx * weights_x).mean() + (dy * weights_y).mean()

# exclude extremly large displacements
MAX_FLOW = 400
# SUM_FREQ = 100
# VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, isVal=False):
    """ Loss function defined over sequence of flow predictions """
    flow_pred = output.flow
    info_pred = output.flow_info
    n_predictions = len(flow_pred)
    flow_gt = torch.nan_to_num(flow_gt, nan=0.0)
    nf = nf_pred(flow_pred, info_pred, flow_gt.flatten(0,1))
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (mag < max_flow)
    # valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = nf[i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())        

    if isVal:
        # Sync across ranks for logging
        flow_loss = reduce_loss(flow_loss)

    if is_main_rank:
        record_loss([flow_loss.detach()])

    return flow_loss
    
def nf_pred(flow_predictions, info_predictions, flow_gt):
    # exlude invalid pixels and extremely large diplacements
    nf_predictions = []
    use_var = True
    var_min = 0
    var_max = 10
    for i in range(len(info_predictions)):
        if not use_var:
            var_max = var_min = 0
        else:
            var_max = var_max
            var_min = var_min
            
        raw_b = info_predictions[i][:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info_predictions[i][:, :2]
        # Large b Component                
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
        # Small b Component
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
        # term2: [N, 2, m, H, W]
        term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
        # term1: [N, m, H, W]
        term1 = weight - math.log(2) - log_b
        nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
        nf_predictions.append(nf_loss)
    return nf_predictions