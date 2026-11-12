
import os
from args import get_args
import torch
import torch.nn as nn
from utils import dice_loss_from_logits, dice_score_from_logits


# 1. Train the model
def train_model(model, train_loader, val_loader, device):

    args = get_args()
    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_dice = -1.0

    # Loop through epochs
    for epoch in range(int(args.epochs)):

        model.train()
        total_loss = 0.0

        # Loop through training batches
        for batch in train_loader:

            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(xrays)

            # Calculate BCE loss
            bce_loss = bce(outputs, masks)
            dice_loss = dice_loss_from_logits(outputs, masks)
            loss = bce_loss + dice_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Average training loss
        train_loss = total_loss / len(train_loader)

        # Validate after each epoch
        val_loss, val_dice = validate_model(
            model, val_loader, device
        )

        # Save the model with the best validation Dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "demo_model.pth"))

        print(
            f"Epoch {epoch + 1}/{int(args.epochs)} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )


# 2. Validate the model
def validate_model(model, val_loader, device):

    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    with torch.no_grad():

        for batch in val_loader:

            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(xrays)

            # Calculate validation loss
            bce_loss = bce(outputs, masks)
            dice_loss = dice_loss_from_logits(outputs, masks)
            loss = bce_loss + dice_loss
            total_loss += loss.item() * xrays.size(0)

            # Calculate Dice score
            dice = dice_score_from_logits(outputs, masks)
            total_dice += dice * xrays.size(0)
            total_images += xrays.size(0)

    val_loss = total_loss / total_images
    val_dice = total_dice / total_images

    return val_loss, val_dice
