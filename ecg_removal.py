import numpy as np
from collections import deque

class StreamingNLMSFilter:
    """
    Streaming NLMS filter for continuous real-time processing
    """
    
    def __init__(self, filter_order=32, mu=0.01, epsilon=1e-10):
        """
        Parameters:
        -----------
        filter_order : int
            Filter order
        mu : float
            Step size (0.0 - 2.0, typically 0.5-1.5 for NLMS)
        epsilon : float
            Regularization term to prevent division by zero
        """
        self.filter_order = filter_order
        self.mu = mu
        self.epsilon = epsilon
        self.weights = np.zeros(filter_order)
        self.buffer = deque(maxlen=filter_order)
        self.stats = {'samples_processed': 0, 'avg_error': 0}
    
    def process(self, emg_noisy, ecg_ref):
        """
        Process a single sample
        
        Returns (cleaned_emg, estimated_ecg, error)
        """
        # Fill buffer
        self.buffer.appendleft(ecg_ref)
        
        if len(self.buffer) < self.filter_order:
            return emg_noisy, 0, emg_noisy
        
        # Convert to array for computation
        x = np.array(self.buffer)
        
        # Estimated ECG artifact
        y = np.dot(self.weights, x)
        
        # Error (cleaned EMG)
        e = emg_noisy - y
        
        # Normalized step size
        x_power = np.dot(x, x) + self.epsilon
        mu_n = self.mu / x_power
        
        # Update weights
        self.weights += mu_n * e * x
        
        # Track statistics
        self.stats['samples_processed'] += 1
        self.stats['avg_error'] = (self.stats['avg_error'] * 
                                   (self.stats['samples_processed'] - 1) + 
                                   e) / self.stats['samples_processed']
        
        return e, y, mu_n
    
    def get_stats(self):
        return self.stats


# Example: Streaming from data source
def streaming_example():
    """
    Simulate real-time data streaming
    """
    # Initialize filter
    filt = StreamingNLMSFilter(filter_order=32, mu=0.5)
    
    # Simulate streaming data
    fs = 1000
    duration = 2
    t = np.arange(0, duration, 1/fs)
    
    emg_clean = 0.5 * np.sin(2 * np.pi * 50 * t)
    ecg = 1.0 * np.sin(2 * np.pi * 1 * t)
    emg_noisy = emg_clean + ecg + 0.05 * np.random.randn(len(t))
    
    # Process sample by sample (real-time)
    output = []
    for i in range(len(emg_noisy)):
        cleaned, est_ecg, mu_n = filt.process(emg_noisy[i], ecg[i])
        output.append(cleaned)
    
    print(f"Processed {filt.stats['samples_processed']} samples")
    print(f"Average error: {filt.stats['avg_error']:.6f}")
    
    return np.array(output), emg_clean


if __name__ == "__main__":
    output, ground_truth = streaming_example()
    
    # Calculate error
    error = output - ground_truth
    print(f"MSE: {np.mean(error**2):.6f}")
