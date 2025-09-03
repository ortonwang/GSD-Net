import torch

def dice_coefficient(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dice_type: str = "fg",
    smooth: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Dice coefficient for binary segmentation masks.

    Args:
        y_true (torch.Tensor): Ground truth binary mask.
        y_pred (torch.Tensor): Predicted binary mask.
        dice_type (str): Type of Dice coefficient to compute: "fg" (foreground) or "bg" (background).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Computed Dice coefficient.
    """
    if dice_type == "fg":
        pred = y_pred > 0.5
        label = y_true > 0
    else:
        pred = y_pred < 0.5
        label = y_true == 0

    inter_size = torch.sum(((pred * label) > 0).float())
    sum_size = (torch.sum(pred) + torch.sum(label)).float()
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice

# pre = torch.randn(1,2,4,4)
# pre = torch.softmax(pre,1)[:,1,:,:]
# gt = torch.ones(1,4,4)
# dice_score = dice_coefficient(gt,pre)
# print(dice_score)