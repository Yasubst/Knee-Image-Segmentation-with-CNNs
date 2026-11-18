
import torch
import torch.nn.functional as F


# 1. Soft Dice Loss (for training)
def dice_loss_from_logits(logits, targets, eps=1e-6):

    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (1, 2, 3)

    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)

    return 1.0 - dice.mean()


# 2. Hard Dice Score (for evaluation)
@torch.no_grad()
def dice_score_from_logits(
    logits, targets, threshold=0.5, eps=1e-6
):

    probs = torch.sigmoid(logits)

    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    dims = (1, 2, 3)

    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.mean().item()


# 3. Combined BCE + Dice Loss
def combined_loss(logits, targets):

    targets = targets.float()

    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets
    )

    dice_loss = dice_loss_from_logits(
        logits, targets
    )

    total_loss = bce_loss + dice_loss

    return total_loss


# 4. Test the functions
if __name__ == "__main__":

    logits = torch.randn(2, 1, 256, 256)

    targets = torch.randint(
        0, 2, (2, 1, 256, 256)
    ).float()

    loss = combined_loss(logits, targets)

    dice = dice_score_from_logits(
        logits, targets
    )

    print("Combined Loss:", loss.item())
    print("Dice Score:", dice)
