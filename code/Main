import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pickle
import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

# Load the trained model parameters from a pickle file
def load_model(filename=r"C:\Users\aman\quantum_model.pkl"):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model loaded from {filename}")
    return params

# Define the quantum device and circuit
def create_circuit(num_qubits):
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit(x, weights):
        for i in range(len(x)):
            if i < num_qubits:  # Ensure we only map to available qubits
                qml.RX(x[i], wires=i)

        for i in range(num_qubits):
            qml.RY(weights[i], wires=i)  # Variational parameter

        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
    
    return circuit

# Traditional method: Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

# Process the uploaded CSV file
def process_csv(file_path):
    # Load the CSV data
    data = pd.read_csv(file_path)
    
    # Assuming the last column is the target (class label) and the rest are features
    X = data.iloc[:, :-1].values  # All columns except the last one as features
    y = data.iloc[:, -1].values    # The last column as the target
    
    # Load quantum model parameters
    global dev_params, circuit
    dev_params = load_model(r"C:\Users\aman\quantum_model.pkl")  # Update the path if needed
    num_qubits = min(X.shape[1], 4)  # Number of qubits based on features
    circuit = create_circuit(num_qubits)

    # Train traditional model
    traditional_model, traditional_time = train_logistic_regression(X, y)

    # Make predictions for the quantum model
    predictions_quantum = []
    for x in X:
        pred = circuit(x, dev_params)  # Get the prediction from the quantum circuit
        pred_value = pred[0].numpy()  # Extract the expectation value
        pred_label = np.sign(pred_value)  # Get the class label based on the sign of the prediction
        predictions_quantum.append(pred_label)

    # Make predictions for the traditional model
    predictions_traditional = traditional_model.predict(X)

    # Print and display the results
    print("Quantum Model Predictions:", predictions_quantum)
    print("Traditional Model Predictions:", predictions_traditional)
    print("Quantum Model Classification Report:")
    print(classification_report(y, predictions_quantum))
    print("Traditional Model Classification Report:")
    print(classification_report(y, predictions_traditional))

    # Accuracy Calculation
    accuracy_quantum = np.mean(predictions_quantum == y)
    accuracy_traditional = np.mean(predictions_traditional == y)

    # Speed Comparison
    speed_comparison(accuracy_quantum, accuracy_traditional, traditional_time)

# Function to compare speed and accuracy
def speed_comparison(accuracy_quantum, accuracy_traditional, traditional_time):
    # Assuming quantum model takes a fixed time (you can measure it if needed)
    quantum_time = 2  # Dummy value for demonstration, replace with actual timing if needed

    # Prepare data for the graph
    methods = ['Quantum Model', 'Traditional Model']
    accuracies = [accuracy_quantum, accuracy_traditional]
    times = [quantum_time, traditional_time]

    # Plotting Accuracy Comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.bar(methods, accuracies, color=['blue', 'orange'])
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    
    # Plotting Speed Comparison
    plt.subplot(1, 3, 2)
    plt.bar(methods, times, color=['blue', 'orange'])
    plt.title('Speed Comparison')
    plt.ylabel('Time (seconds)')
    
    # Scalability can be shown with different dataset sizes. For demonstration purposes,
    # you can use a fixed example of how accuracy scales with dataset size.
    dataset_sizes = [100, 200, 300, 400, 500]
    quantum_accuracies = np.random.uniform(0.6, 0.9, len(dataset_sizes))  # Simulated values
    traditional_accuracies = np.random.uniform(0.7, 0.95, len(dataset_sizes))  # Simulated values

    plt.subplot(1, 3, 3)
    plt.plot(dataset_sizes, quantum_accuracies, label='Quantum Model', marker='o')
    plt.plot(dataset_sizes, traditional_accuracies, label='Traditional Model', marker='o')
    plt.title('Scalability Comparison')
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Function to select file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        process_csv(file_path)

# Create the GUI
root = tk.Tk()
root.title("CSV File Uploader")
root.geometry("300x150")

upload_button = tk.Button(root, text="Upload CSV", command=select_file)
upload_button.pack(expand=True)

# Run the application
root.mainloop()
