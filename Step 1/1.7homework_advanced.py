import numpy as np
import pandas as pd

class SmartDataFrame:
    def __init__(self, df):
        self.df = df
    
    def optimize_types(self):
        self.df['Amount'] = self.df['Amount'].astype(np.int16)
        self.df['Price'] = self.df['Price'].astype(np.float32)

    def get_revenue_by_position(self, row_idx):
        return self.df.iloc[row_idx, 1] * self.df.iloc[row_idx, 2]