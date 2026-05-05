"""
Dataset generator for quantum cryptography eavesdropping detection.

Generates a dataset with ~500 samples following BB84 quantum key distribution protocol.
Each sample contains a photon series of length 4, with 6 features per photon:
- tx_bit: transmitted bit (0 or 1)
- t_basis: transmitted basis (0=rectilinear, 1=diagonal)
- rx_bit: received bit (0 or 1)
- rx_basis: received basis (0=rectilinear, 1=diagonal)
- basis_match: whether rx_basis matches t_basis (0 or 1)
- bit_error: error detected when basis_match and rx_bit != tx_bit (0 or 1)

Labels: eavesdropped (0=no eavesdropping, 1=eavesdropping detected)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_bb84_transmission(eavesdropped: bool, photon_count: int = 4) -> dict:
    """
    Generate a single BB84 photon series transmission.
    
    Args:
        eavesdropped: Whether an eavesdropper is present
        photon_count: Number of photons in the series (default 4)
    
    Returns:
        Dictionary containing features for all photons and the label
    """
    features = {}
    error_count = 0
    
    for i in range(photon_count):
        # Alice (sender) randomly generates bit and basis
        tx_bit = np.random.randint(0, 2)
        t_basis = np.random.randint(0, 2)
        
        # Bob (receiver) randomly chooses basis
        rx_basis = np.random.randint(0, 2)
        
        # Check if bases match
        basis_match = 1 if rx_basis == t_basis else 0
        
        if eavesdropped:
            # Eve (eavesdropper) intercepts and measures
            eve_basis = np.random.randint(0, 2)
            eve_bit = np.random.randint(0, 2)  # Eve's measurement
            
            # Eve retransmits what she measured
            if eve_basis == t_basis:
                # Eve guessed correctly, retransmits correctly
                retransmitted_bit = tx_bit
            else:
                # Eve guessed wrong, may have wrong bit
                # 50% chance Eve measured the wrong bit
                retransmitted_bit = tx_bit if np.random.random() > 0.5 else 1 - tx_bit
            
            rx_bit = retransmitted_bit
            
            # Bit error occurs when basis matches but bit doesn't
            if basis_match and rx_bit != tx_bit:
                error_count += 1
        else:
            # No eavesdropper: normal BB84 protocol
            if basis_match:
                # Correct basis: transmit without error
                rx_bit = tx_bit
            else:
                # Wrong basis: random guess (50/50)
                rx_bit = np.random.randint(0, 2)
        
        bit_error = 1 if (basis_match and rx_bit != tx_bit) else 0
        
        # Store features for this photon
        features[f'photon_{i}_tx_bit'] = tx_bit
        features[f'photon_{i}_t_basis'] = t_basis
        features[f'photon_{i}_rx_bit'] = rx_bit
        features[f'photon_{i}_rx_basis'] = rx_basis
        features[f'photon_{i}_basis_match'] = basis_match
        features[f'photon_{i}_bit_error'] = bit_error
    
    # Determine label based on error threshold
    # Eavesdropping typically causes higher error rates
    bit_error_rate = error_count / photon_count
    
    if eavesdropped:
        # Add noise to make classification realistic
        # Eavesdropping usually causes ~25% error rate (Eve wrong basis ~50% of time,
        # and when wrong, has 50% chance of introducing error)
        label = 1 if bit_error_rate > 0.05 else np.random.binomial(1, 0.3)
    else:
        # Normal channel should have very low error rates
        label = 0 if bit_error_rate < 0.1 else np.random.binomial(1, 0.1)
    
    features['eavesdropped'] = label
    
    return features


def generate_dataset(num_samples: int = 500, eavesdrop_ratio: float = 0.5) -> pd.DataFrame:
    """
    Generate the complete dataset.
    
    Args:
        num_samples: Number of samples to generate (default 500)
        eavesdrop_ratio: Proportion of samples with eavesdropping (default 0.5 for balance)
    
    Returns:
        DataFrame with 24 features + 1 label column
    """
    num_eavesdropped = int(num_samples * eavesdrop_ratio)
    num_normal = num_samples - num_eavesdropped
    
    data = []
    
    # Generate normal transmissions
    for _ in range(num_normal):
        data.append(generate_bb84_transmission(eavesdropped=False))
    
    # Generate eavesdropped transmissions
    for _ in range(num_eavesdropped):
        data.append(generate_bb84_transmission(eavesdropped=True))
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Eavesdropped samples: {df['eavesdropped'].sum()} ({df['eavesdropped'].mean()*100:.1f}%)")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset statistics:\n{df.describe()}")


if __name__ == "__main__":
    # Generate dataset
    print("Generating quantum cryptography eavesdropping detection dataset...")
    dataset = generate_dataset(num_samples=500, eavesdrop_ratio=0.5)
    
    # Save to CSV
    save_dataset(dataset, "quantum_bb84_dataset.csv")
