import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# Load your dataset from the CSV file
data_path = r'C:\Users\aman\Desktop\hackathon\balanced_creditcard.csv'  # Update this to your actual file path
data = pd.read_csv(data_path)

# Assuming the last column is the target (class label) and the rest are features
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values    # The last column as the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check number of features
num_features = X_train.shape[1]
print(f'Number of features: {num_features}')

# Set the number of qubits based on the number of features
num_qubits = min(num_features, 4)  # Adjust to your device's limitations

# Dimensionality reduction if necessary
if num_features > num_qubits:
    pca = PCA(n_components=num_qubits)  # Set to the number of qubits
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f'PCA applied. Reduced number of features to {num_qubits}')

# Define the quantum device
dev = qml.device('default.qubit', wires=num_qubits)

# Initialize weights for the circuit
weight_shapes = {"w": (num_qubits,)}
dev_params = np.random.uniform(low=-np.pi, high=np.pi, size=weight_shapes["w"])

# Define the quantum circuit with parameters
@qml.qnode(dev)
def circuit(x, weights):
    # Encode classical data into quantum states
    for i in range(len(x)):
        if i < num_qubits:  # Ensure we only map to available qubits
            qml.RX(x[i], wires=i)

    # Example of a simple variational circuit with weights
    for i in range(num_qubits):
        qml.RY(weights[i], wires=i)  # Variational parameter
    qml.CNOT(wires=[0, 1])
    if num_qubits > 2:
        qml.CNOT(wires=[1, 2])
    if num_qubits > 3:
        qml.CNOT(wires=[2, 3])

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Define the loss function
def loss_fn(predictions, y_true):
    return np.mean((predictions - y_true) ** 2)  # Mean Squared Error

# Training function with learning
def train_quantum_model(X_train, y_train, epochs, learning_rate=0.01):
    global dev_params
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            # Forward pass
            predictions = circuit(x, dev_params)
            pred_value = predictions[0]  # Use the first qubit's measurement
            
            # Compute loss
            loss = loss_fn(pred_value, y)
            total_loss += loss
            
            # Compute gradients (finite difference approximation)
            for j in range(len(dev_params)):
                original_param = dev_params[j]
                dev_params[j] += 0.01  # Small step for numerical gradient
                loss_plus = loss_fn(circuit(x, dev_params)[0], y)
                dev_params[j] = original_param - 0.01  # Small step in the opposite direction
                loss_minus = loss_fn(circuit(x, dev_params)[0], y)
                
                # Numerical gradient
                grad = (loss_plus - loss_minus) / 0.02
                dev_params[j] += learning_rate * grad  # Update parameters

            # Print updates for each batch/sample
            print(f'Epoch: {epoch + 1}/{epochs}, Sample: {i + 1}/{len(X_train)}, Loss: {loss:.4f}')
        
        avg_loss = total_loss / len(X_train)
        print(f'Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}\n')

# Evaluation function with detailed updates
def evaluate_model(X_test, y_test):
    predictions = []
    total_correct = 0
    for i, (x, y_true) in enumerate(zip(X_test, y_test)):
        pred = circuit(x, dev_params)
        pred_value = pred[0]  # Use the first qubit's measurement
        pred_label = np.sign(pred_value)  # Assuming binary classification
        predictions.append(pred_label)

        # Compare prediction with actual label
        correct = pred_label == y_true
        total_correct += correct

        # Print updates for each test sample
        print(f'Test Sample: {i + 1}/{len(X_test)}, Predicted: {pred_label}, True: {y_true}, Correct: {correct}')
    
    accuracy = total_correct / len(X_test)
    print(f'\nEvaluation completed. Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, predictions))

# Save the model parameters
def save_model(params, filename="quantum_model_params.npy"):
    np.save(filename, params)
    print(f"Model saved to {filename}")

# Load the model parameters
def load_model(filename="quantum_model_params.npy"):
    params = np.load(filename)
    print(f"Model loaded from {filename}")
    return params

# Train the model
print("Starting training...\n")
train_quantum_model(X_train, y_train, epochs=10)

# Save the model parameters after training
print("Saving model...")
save_model(dev_params, filename="quantum_model_params.npy")

# Load the model parameters before evaluation (optional)
print("Loading model for evaluation...")
dev_params = load_model("quantum_model_params.npy")

# Evaluate the model
print("Starting evaluation...\n")
evaluate_model(X_test, y_test)
