import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as T
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from math import pi, sin, cos
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib




# Calculates the Mean Absolute Error (MAE) between predictions and targets
def calculate_mae(preds, targets):
    return torch.mean(torch.abs(preds - targets))


# Mean Squared Error (MSE) as Loss function
criterion_mse = nn.MSELoss()

# Determine the device to use: Use CUDA if available; otherwise, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
FUNCTION FOR MODEL TRAINING 
"""


def train_model(model, optimizer, criterion, dataloader, device):
    
    model.train()  # Set the model to training mode, which enables dropout and batch normalization layers

    epoch_loss = 0.0  # Initialize total loss for the epoch
    epoch_mae = 0.0  # Initialize total mean absolute error for the epoch
    
    for input_batch, target_batch_values in dataloader:
        
        # Move input and target data to the specified device (GPU or CPU)
        src = input_batch.to(device)  # Prepare source sequence for model input
        tgt = target_batch_values.to(device)  # Prepare target sequence for comparison with model output


        optimizer.zero_grad()  # Clear gradients before each backward pass
        
        # Perform a forward pass through the model
        outputs = model(src, tgt)

        # Ensure output dimensions are correct, especially for single-item batches
        if outputs.shape[0] == 1:
            outputs = outputs.squeeze(0)  # Remove singleton dimension if batch size is 1
        elif outputs.shape[0] != tgt.shape[0]:
            # Check for dimension mismatch and raise an error if found
            raise ValueError(f"Output and target batch sizes do not match: {outputs.shape[0]} != {tgt.shape[0]}")

        # Compute the loss between the model's outputs and the target values
        loss = criterion(outputs, tgt)
        
        loss.backward()  # Compute gradient of the loss with respect to model parameters
        optimizer.step()  # Update model parameters based on gradients

        # Accumulate total loss and MAE for the epoch
        epoch_loss += loss.item()  # Sum up batch loss
        epoch_mae += calculate_mae(outputs, tgt).item()  # Sum up batch mean absolute error

    # Calculate average loss and MAE across all batches
    avg_loss = epoch_loss / len(dataloader)
    avg_mae = epoch_mae / len(dataloader)

    return avg_loss, avg_mae





"""
FUNCTION FOR MODEL VALIDATION 
"""




def validate_model(model, criterion, dataloader, device):
    
    model.eval()  # Switch the model to evaluation mode (disables dropout and batch normalization)

    epoch_loss = 0.0  # Initialize total loss for the epoch
    epoch_mae = 0.0  # Initialize total mean absolute error for the epoch

    # Disable gradient computation to speed up the process and reduce memory usage
    with torch.no_grad():
        for input_batch, target_batch_values in dataloader:
            
            # Move input and target data to the specified device (GPU or CPU)
            src = input_batch.to(device)  # Prepare source sequence for model input
            tgt = target_batch_values.to(device)  # Prepare target sequence for comparison with model output

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(src, tgt)

            # Check the shape of the outputs to ensure they match the target's shape
            if outputs.shape[0] == 1:
                outputs = outputs.squeeze(0)  # If batch size is 1, remove singleton dimension
            elif outputs.shape[0] != tgt.shape[0]:
                # If output and target batch sizes do not match, raise an error
                raise ValueError(f"Output and target batch sizes do not match: {outputs.shape[0]} != {tgt.shape[0]}")

            # Calculate the loss and mean absolute error for the batch
            loss = criterion(outputs, tgt)

            # Accumulate the loss and MAE for all batches
            epoch_loss += loss.item()
            epoch_mae += calculate_mae(outputs, tgt).item()

    # Calculate average loss and MAE across all batches
    avg_loss = epoch_loss / len(dataloader)
    avg_mae = epoch_mae / len(dataloader)

    # Return the average loss and MAE for this validation run
    return avg_loss, avg_mae







"""
FUNCTION FOR MODEL EVALUATION
"""



def evaluate_model(model, criterion, dataloader, device):
    
    model.eval()  # Switch the model to evaluation mode, disabling dropout and batch normalization for consistent predictions

    epoch_loss = 0.0  # Initialize total loss for the evaluation
    epoch_mae = 0.0  # Initialize total mean absolute error for the evaluation

    all_predictions = []  # List to store all predictions across batches
    all_targets = []  # List to store all target values across batches
    
    

    with torch.no_grad():  # Context manager to turn off gradient computation, saving memory and computations
        
        for input_batch, target_batch_values in dataloader:
            src = input_batch.to(device)  # Move input batch to the specified device
            tgt = target_batch_values.to(device)  # Move target batch to the specified device

            outputs = model(src, tgt)  # Forward pass: compute model's predictions

            # Check if the output shape matches the target shape and adjust if necessary
            if outputs.shape[0] == 1:
                outputs = outputs.squeeze(0)  # Remove unnecessary dimension if batch size is 1
            elif outputs.shape[0] != tgt.shape[0]:
                # Ensure the predicted batch size matches the target batch size
                raise ValueError(f"Output and target batch sizes do not match: {outputs.shape[0]} != {tgt.shape[0]}")
            

            loss = criterion(outputs, tgt)  # Calculate loss for the current batch
            epoch_loss += loss.item()  # Accumulate loss over the epoch
            epoch_mae += calculate_mae(outputs, tgt).item()  # Accumulate MAE over the epoch

            all_predictions.append(outputs.cpu().numpy())  # Store predictions (move to CPU and convert to NumPy array)
            all_targets.append(tgt.cpu().numpy())  # Store targets similarly
            

    # Calculate average loss and MAE for the epoch
    avg_loss = epoch_loss / len(dataloader)
    avg_mae = epoch_mae / len(dataloader)

    # Concatenate all batch predictions and targets into single arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Return the average loss, MAE, and the complete sets of predictions and targets
    return avg_loss, avg_mae, all_predictions, all_targets
