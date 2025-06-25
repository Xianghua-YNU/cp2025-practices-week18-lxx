import numpy as np
import matplotlib.pyplot as plt

def generate_px_samples(N):
    """Generate random numbers following p(x) = 1/(2√x) distribution"""
    u = np.random.rand(N)  # Uniform random numbers in [0,1]
    x = u**2  # Transform to p(x) distribution
    return x

def integrand(x):
    """The integrand function x^(-1/2)/(e^x + 1)"""
    return 1 / (np.sqrt(x) * (np.exp(x) + 1))

def importance_sampling(N):
    """Estimate the integral using importance sampling"""
    x_samples = generate_px_samples(N)
    f_over_p = integrand(x_samples) / (1/(2*np.sqrt(x_samples)))
    integral = np.mean(f_over_p)
    variance = np.var(f_over_p)
    error = np.sqrt(variance/N)
    return integral, error

# Set parameters
N = 1000000

# Calculate integral and error
integral, error = importance_sampling(N)

print(f"Integral estimate: {integral:.6f}")
print(f"Estimated error: {error:.6f}")

# Visualization of the sampling distribution
plt.figure(figsize=(10, 6))
x_plot = np.linspace(0.01, 1, 1000)
plt.plot(x_plot, integrand(x_plot), 'r-', label='Integrand: $x^{-1/2}/(e^x+1)$')
plt.plot(x_plot, 1/(2*np.sqrt(x_plot)), 'b-', label='Weight function: $1/(2\sqrt{x})$')
plt.xlabel('x', fontsize=12)
plt.ylabel('Function value', fontsize=12)
plt.title('Integrand and Weight Function Comparison', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
