"""
Neural network for quantum cryptography eavesdropping detection.

Uses a 2-hidden layer neural network to classify whether a quantum key
distribution transmission has been eavesdropped based on bit error patterns.
Built with PyTorch.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EavesdropperDetectorNN(nn.Module):
    """
    Neural network for quantum cryptography eavesdropping detection.
    
    3-hidden layer architecture for binary classification of eavesdropped vs normal
    quantum key distribution transmissions.
    """
    
    def __init__(self, input_dim: int, hidden_units_1: int = 16, hidden_units_2: int = 12, 
                 hidden_units_3: int = 8):
        """
        Initialize the eavesdropper detector neural network.
        
        Args:
            input_dim: Number of input features (24 for our dataset)
            hidden_units_1: Number of neurons in first hidden layer (default 16)
            hidden_units_2: Number of neurons in second hidden layer (default 12)
            hidden_units_3: Number of neurons in third hidden layer (default 8)
        """
        super(EavesdropperDetectorNN, self).__init__()
        
        # First hidden layer
        self.dense1 = nn.Linear(input_dim, hidden_units_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        # Second hidden layer
        self.dense2 = nn.Linear(hidden_units_1, hidden_units_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        # Third hidden layer
        self.dense3 = nn.Linear(hidden_units_2, hidden_units_3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer (binary classification)
        self.output_layer = nn.Linear(hidden_units_3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x: Input features
        
        Returns:
            Model predictions
        """
        # First hidden layer with dropout
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second hidden layer with dropout
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Third hidden layer with dropout
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x


def load_dataset(filepath: str) -> tuple:
    """
    Load dataset from CSV and split into features and labels.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Tuple of (X, y) as NumPy arrays
    """
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop('eavesdropped', axis=1).to_numpy()
    y = df['eavesdropped'].to_numpy()
    
    print(f"Loaded dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def build_model(input_dim: int, hidden_units_1: int = 16, hidden_units_2: int = 12, 
                hidden_units_3: int = 8, device='cpu'):
    """
    Create an eavesdropper detector model.
    
    Args:
        input_dim: Number of input features (24 for our dataset)
        hidden_units_1: Number of neurons in first hidden layer
        hidden_units_2: Number of neurons in second hidden layer
        hidden_units_3: Number of neurons in third hidden layer
        device: Device to place model on ('cpu' or 'cuda')
    
    Returns:
        EavesdropperDetectorNN model on specified device
    """
    model = EavesdropperDetectorNN(input_dim, hidden_units_1, hidden_units_2, hidden_units_3)
    model = model.to(device)
    
    return model


def train_model(model: EavesdropperDetectorNN, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 50, learning_rate: float = 0.001, device='cpu') -> tuple:
    """
    Train the neural network.
    
    Args:
        model: EavesdropperDetectorNN model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
    
    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses


def evaluate_model(model: EavesdropperDetectorNN, test_loader: DataLoader, device='cpu'):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained EavesdropperDetectorNN model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Test loss value
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            test_loss += loss.item()
            
            # Binary predictions (threshold 0.5)
            predictions = (outputs.squeeze() > 0.5).int()
            
            total += batch_y.size(0)
            correct += (predictions == batch_y).sum().item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix and classification report
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(all_labels, all_preds)}")
    
    return avg_test_loss


def plot_training_loss(train_losses: list, val_losses: list):
    """
    Plot training and validation loss vs epochs.
    
    Args:
        train_losses: List of training loss values per epoch
        val_losses: List of validation loss values per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', linewidth=2, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='s', linestyle='-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_plot.png', dpi=300)
    print("\nTraining and validation loss plot saved to 'training_loss_plot.png'")
    plt.show()


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading quantum BB84 dataset...")
    X, y = load_dataset('quantum_bb84_dataset.csv')
    
    # Split into train, validation, and test sets
    print("\nSplitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Build model
    print("\nBuilding neural network...")
    model = build_model(input_dim=X_train.shape[1], hidden_units_1=16, hidden_units_2=12, 
                       hidden_units_3=8, device=device)
    print(model)
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=25, learning_rate=0.001, device=device)
    
    # Plot training and validation loss
    print("\nGenerating training and validation loss visualization...")
    plot_training_loss(train_losses, val_losses)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss = evaluate_model(model, test_loader, device=device)
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'eavesdropping_detector_model.pth')
    print("Model saved to eavesdropping_detector_model.pth")
