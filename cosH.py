import json
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram

def custom_hadamard_matrix(x, nd):
    factor = 1/np.sqrt(x**(2*nd) - nd)
    return [[factor, factor], [factor, -factor]]

# Given values:
x = 2.5000000000000004  # Explicitly set value for x
nd = 0.5  # Sample value for nd

matrix = custom_hadamard_matrix(x, nd)

# Ensure the matrix is unitary:
product = np.dot(matrix, np.conj(matrix).T)
if not np.allclose(product, np.eye(2)):
    raise ValueError("The matrix is not unitary.")

# Create a quantum circuit with 1 qubit
qc = QuantumCircuit(192)
qc.unitary(matrix, [0], label='custom_hadamard')
qc.measure_all()  # Add measurement

# Test the circuit
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit).result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts).show()
with open('results.json', 'w') as f:
    json.dump(counts, f)
# Convert counts to DataFrame
df = pd.DataFrame(list(counts.items()), columns=['Eigenstate', 'Count'])

# If you want to break down the 192-qubit eigenstate to 128 columns
df_eigenstate = df['Eigenstate'].apply(lambda x: pd.Series(list(x)))
df = pd.concat([df, df_eigenstate], axis=1)
df.drop(columns=['Eigenstate'], inplace=True)

# If you want to truncate or limit data to 32000 instances
df = df.head(32000)
df.to_csv('results.csv', index=False)
