# Eavesdropper Detector for Quantum Cryptography

A neural network-based system for detecting eavesdropping in quantum key distribution (QKD) transmissions using the BB84 protocol. This project uses PyTorch to train a deep learning model that analyzes quantum transmission patterns and identifies potential eavesdropping attempts with high accuracy.

## Project Overview

### What is Quantum Cryptography?

Quantum cryptography, specifically the BB84 protocol, is a method for secure key exchange that leverages quantum mechanics. Unlike classical cryptography, any attempt to eavesdrop on a quantum transmission introduces detectable anomalies in the bit error patterns.

### How It Works

The eavesdropping detector analyzes quantum transmission data including:
- Photon transmission and reception bits
- Basis information (rectilinear or diagonal)
- Basis matching between sender and receiver
- Bit error rates
- Eavesdropping status labels (training data)

A 3-layer neural network trained on this data learns to classify whether a transmission has been eavesdropped based on statistical anomalies in the bit error patterns.

## Getting Started

### Prerequisites

- Python 3.13+
- PyTorch
- NumPy, Pandas, scikit-learn, Matplotlib
- VS Code (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Eavesdropper-Detector-for-Quantum-Cryptography
```

2. Set up the Python environment (macOS/Linux/WSL):
```bash
bash scripts/bootstrap-linux.sh
```

Or for Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap-windows.ps1
```

3. Verify PyTorch installation:
```bash
python scripts/test-pytorch.py
```

## Project Structure

```
.
├── README.md                           # This file
├── pyproject.toml                     # Python dependencies
├── quantum_bb84_dataset.csv           # Generated training dataset
├── eavesdropping_detector_model.pth   # Trained model weights
├── scripts/
│   ├── bootstrap-linux.sh
│   ├── bootstrap-macos.sh
│   ├── bootstrap-windows.ps1
│   └── test-pytorch.py
├── src/
│   ├── data-generator.py              # Generates synthetic BB84 data
│   ├── eavesdropping-detector-nn.py   # Main training script
│   └── hello.py
└── tests/
    └── README.md
```

## Running the Project

### Step 1: Generate Dataset

Create synthetic quantum BB84 transmission data:

```bash
python src/data-generator.py
```

**Output:** `quantum_bb84_dataset.csv`

**Dataset Composition:**
- 500 total samples
- Approximately 250 eavesdropped transmissions (label=1)
- Approximately 250 non-eavesdropped transmissions (label=0)
- 24 input features per sample
- Roughly balanced binary classification problem (exact distribution varies due to stochastic generation)

**Features:**
- 4 photons × 6 attributes each (transmission bit, basis, reception bit, reception basis, basis match, bit error)

### Step 2: Train the Eavesdropping Detector

Train the neural network model on the generated dataset:

```bash
python src/eavesdropping-detector-nn.py
```

**What the script does:**
1. Loads `quantum_bb84_dataset.csv`
2. Splits data: 60% training, 20% validation, 20% test
3. Normalizes features using StandardScaler
4. Builds and trains neural network
5. Generates training loss visualization
6. Evaluates on test set
7. Saves trained model weights

**Outputs:**
- `training_loss_plot.png` - Training and validation loss curves
- `eavesdropping_detector_model.pth` - Trained model weights
- Console output with test metrics and classification report

## Model Architecture

### Neural Network Structure

```
Input Layer (24 features)
    ↓
Dense Layer 1 (16 neurons)
    + ReLU Activation
    + Dropout (20%)
    ↓
Dense Layer 2 (12 neurons)
    + ReLU Activation
    + Dropout (20%)
    ↓
Dense Layer 3 (8 neurons)
    + ReLU Activation
    + Dropout (20%)
    ↓
Output Layer (1 neuron)
    + Sigmoid Activation
    ↓
Binary Classification Output (0 or 1)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy (BCE) |
| Regularization | Dropout (0.2 each layer) |

## Evaluation Metrics

### Classification Report

The model evaluation displays:

- **Precision**: Of all cases flagged as eavesdropped, how many actually were
- **Recall**: Of all actual eavesdropped cases, how many did the model catch
- **F1-Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Accuracy**: Overall correct predictions across all samples
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives

### Example Output

```
Test Loss: 0.3245
Test Accuracy: 0.8750

Confusion Matrix:
[[85  5]
 [ 9 51]]

Classification Report:
              precision    recall  f1-score   support
           0       0.90      0.94      0.92        90
           1       0.91      0.85      0.88        60
    accuracy                           0.88       150
```

## Understanding the Results

### Precision vs Recall Trade-off

- **High Precision**: Few false alarms (when we say eavesdropped, we're usually right)
  - Good when false positives are costly
  
- **High Recall**: Few missed threats (we catch most eavesdropping attempts)
  - Good when false negatives are costly

The model balances both for quantum security applications.

## Code Overview

### data-generator.py
Generates synthetic quantum BB84 transmission data with realistic patterns for normal and eavesdropped scenarios.

### eavesdropping-detector-nn.py
Main training script with:
- Data loading and preprocessing
- Train/validation/test split
- Feature normalization
- Neural network definition and training
- Model evaluation and visualization
- Model persistence

## Dependencies

Core dependencies (installed via bootstrap):
- torch (PyTorch)
- numpy
- pandas
- scikit-learn
- matplotlib

See `pyproject.toml` for complete dependency list and versions.

## Troubleshooting

### PyTorch Not Installed
```bash
python scripts/test-pytorch.py
```

### Reset Python Environment
```bash
rm -rf .venv
uv sync
```

Then restart VS Code.

### CUDA/GPU Issues
The code automatically falls back to CPU if CUDA is unavailable. Training will be slower but still functional.

## Future Enhancements

- GPU acceleration with CUDA
- Hyperparameter tuning for improved accuracy
- Real quantum transmission data integration
- Model interpretability analysis
- Deployment as REST API
- Extended to other QKD protocols (E91, SARG04)

## References

- Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing"
- PyTorch Documentation: https://pytorch.org/
- Quantum Cryptography: https://en.wikipedia.org/wiki/Quantum_cryptography

## License

This project is part of the Engineering Artificial Intelligence course.

---

**Happy quantum computing! 🚀🔐**
