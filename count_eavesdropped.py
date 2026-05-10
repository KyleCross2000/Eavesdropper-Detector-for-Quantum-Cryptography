import pandas as pd

# Load the dataset
df = pd.read_csv('quantum_bb84_dataset.csv')

# Get basic info
total_samples = len(df)
eavesdropped_count = (df['eavesdropped'] == 1).sum()
not_eavesdropped_count = (df['eavesdropped'] == 0).sum()

# Display results
print(f"Total samples: {total_samples}")
print(f"Eavesdropped (1): {eavesdropped_count}")
print(f"Not eavesdropped (0): {not_eavesdropped_count}")
print(f"\nPercentage eavesdropped: {(eavesdropped_count/total_samples)*100:.2f}%")
print(f"Percentage not eavesdropped: {(not_eavesdropped_count/total_samples)*100:.2f}%")
