import numpy as np
import pandas as pd
 
class MarketAnalyzer:
    def __init__(self, df_orders, df_categories):
        self.df_orders = df_orders
        self.df_categories = df_categories
    
    def clean_and_merge(self):
        df_sort = self.df_orders[self.df_orders['price'] > 0]
        self.merged = pd.merge(df_sort, self.df_categories, on='item_id', how='inner')

    def get_category_revenue(self):
        return self.merged.groupby('category')['price'].sum()
    