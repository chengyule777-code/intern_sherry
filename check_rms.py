import pandas as pd
import numpy as np

df = pd.read_csv('./test_data/rr_20260520_094816/cleaned_result.csv')

rest_raw = df['raw'].iloc[2718:3598]
rest_cleaned = df['cleaned'].iloc[2718:3598]

print(f"Raw RMS: {np.sqrt(np.mean(rest_raw**2))}")
print(f"Cleaned RMS: {np.sqrt(np.mean(rest_cleaned**2))}")