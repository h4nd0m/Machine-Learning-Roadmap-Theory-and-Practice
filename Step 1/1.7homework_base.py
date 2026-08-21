import pandas as pd
import numpy as np

class SalesDataset:
    def __init__(self, products, amount, prices):
        self.df = pd.DataFrame({
            'Product': products,
            'Amount': amount,
            'Price': prices
        })

    def get_preview(self):
        return self.df.head(2)
    
    def get_stats(self):
        return self.df.describe()
    
if __name__ == '__main__':

    prod_list = ['Монитор', 'Клавиатура', 'Мышь', 'Ноутбук', 'Наушники']
    amount_list = [10, 50, 65, 5, 32]
    price_list = [25000.0, 3500.0, 1800.0, 95000.0, 5500.0]

    dataset = SalesDataset(prod_list, amount_list, price_list)

    print(dataset.get_preview())

    print(dataset.get_stats())