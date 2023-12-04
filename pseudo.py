
import numpy as np
import matplotlib.pyplot as plt

def time_evolution_operator(hamiltonian, time, h_bar=1.0):
    return np.exp(-1j * hamiltonian * time / h_bar)

def evolve_state(initial_state, time_evolution_operator):
    return np.dot(time_evolution_operator, initial_state)

def plot_evolution(initial_state, hamiltonian, timesteps=100, h_bar=1.0):
    times = np.linspace(0.0, 2.0 * np.pi, timesteps)
    states = []

    for t in times:
        time_evolution_op = time_evolution_operator(hamiltonian, t, h_bar)
        evolved_state = evolve_state(initial_state, time_evolution_op)
        states.append(evolved_state)

    states = np.array(states)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(initial_state)):
        plt.plot(times, np.real(states[:, i]), label=f'Real part of State {i + 1}')

    plt.title('Temporal Evolution of Quantum State')
    plt.xlabel('Time')
    plt.ylabel('State Amplitude')
    plt.legend()
    plt.show()

# Example Hamiltonian (2x2 matrix)
hamiltonian = np.array([[1.0, 0.0], [0.0, -1.0]])

# Example initial state (column vector)
initial_state = np.array([1.0, 0.0])

# Plot the temporal evolution
plot_evolution(initial_state, hamiltonian)
