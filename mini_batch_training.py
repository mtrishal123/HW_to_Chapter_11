import numpy as np

def mini_batch_generator(X, y, batch_size=32):
    
    data_size = X.shape[0]
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        if start_idx < data_size:
            excerpt = indices[start_idx:end_idx]
            yield X[excerpt], y[excerpt]

# Example usage
if __name__ == "__main__":
    # Random dataset for example purposes
    X = np.random.rand(1000, 20)  # 1000 samples, 20 features each
    y = np.random.randint(0, 2, 1000)  # 1000 binary labels
    
    batch_size = 32
    for X_batch, y_batch in mini_batch_generator(X, y, batch_size):
        print(f"Mini-batch X shape: {X_batch.shape}, y shape: {y_batch.shape}")
