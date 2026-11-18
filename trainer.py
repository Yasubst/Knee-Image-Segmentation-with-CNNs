
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from args import get_args
from utils import combined_loss, dice_score_from_logits


# 1. Train the model
def train_model(model, train_loader, val_loader, device):

    args = get_args()
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    # Reduce learning rate when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    os.makedirs(args.out_dir, exist_ok=True)

    best_dice = -1.0

    # Store metrics for plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "learning_rate": []
    }

    # Loop through epochs
    for epoch in range(int(args.epochs)):

        model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_images = 0

        # Training batches
        for batch in train_loader:

            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(xrays)

            # Combined BCE + Dice loss
            loss = combined_loss(outputs, masks)

            # Backpropagation
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = xrays.size(0)

            total_loss += loss.item() * batch_size

            # Training Dice
            with torch.no_grad():
                dice = dice_score_from_logits(
                    outputs.detach(),
                    masks,
                    threshold=0.5
                )

            total_dice += dice * batch_size
            total_images += batch_size

        # Average training metrics
        train_loss = total_loss / total_images
        train_dice = total_dice / total_images

        # Validation
        val_loss, val_dice = validate_model(
            model,
            val_loader,
            device
        )

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update scheduler using validation loss
        scheduler.step(val_loss)

        # Save best model
        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.out_dir,
                    "demo_model.pth"
                )
            )

            print(
                f"Best model saved! "
                f"Validation Dice: {best_dice:.4f}"
            )

        # Save history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["learning_rate"].append(current_lr)

        # Print training progress
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.7f}"
        )

        # Save training history after every epoch
        pd.DataFrame(history).to_csv(
            os.path.join(
                args.out_dir,
                "training_history.csv"
            ),
            index=False
        )

        # Update plots after every epoch
        plot_training_history(history, args.out_dir)

    print("\nTraining completed!")
    print(f"Best Validation Dice: {best_dice:.4f}")

    return history


# 2. Validate the model
def validate_model(model, val_loader, device):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_images = 0

    with torch.no_grad():

        for batch in val_loader:

            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            outputs = model(xrays)

            # Validation loss
            loss = combined_loss(outputs, masks)

            # Validation Dice
            dice = dice_score_from_logits(
                outputs,
                masks,
                threshold=0.5
            )

            batch_size = xrays.size(0)

            total_loss += loss.item() * batch_size
            total_dice += dice * batch_size
            total_images += batch_size

    val_loss = total_loss / total_images
    val_dice = total_dice / total_images

    return val_loss, val_dice


# 3. Plot training history
def plot_training_history(history, out_dir):

    epochs = history["epoch"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9)
    )

    # Plot 1: Training and validation loss
    axes[0, 0].plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        color="blue"
    )

    axes[0, 0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        color="red"
    )

    axes[0, 0].set_title(
        "Training vs Validation Loss"
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Training and validation Dice
    axes[0, 1].plot(
        epochs,
        history["train_dice"],
        label="Training Dice",
        color="green"
    )

    axes[0, 1].plot(
        epochs,
        history["val_dice"],
        label="Validation Dice",
        color="orange"
    )

    axes[0, 1].set_title(
        "Training vs Validation Dice Score"
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: Learning rate
    axes[1, 0].plot(
        epochs,
        history["learning_rate"],
        color="purple"
    )

    axes[1, 0].set_title(
        "Learning Rate Schedule"
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].grid(True)

    # Plot 4: Validation Dice
    axes[1, 1].plot(
        epochs,
        history["val_dice"],
        label="Validation Dice",
        color="darkgreen"
    )

    axes[1, 1].set_title(
        "Validation Dice Progress"
    )
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Dice Score")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            out_dir,
            "training_curves.png"
        ),
        dpi=200
    )

    plt.close(fig)
