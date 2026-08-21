import numpy as np
class SalesAnalyzer:
    def __init__(self, sales_matrix):
        self.sales = sales_matrix
    
    def get_average_sales(self):
        return np.mean(self.sales, axis = 0)
    
    def center_data(self):
        return self.sales - self.get_average_sales()

if __name__ == "__main__":
    sales = np.array([
        [100, 500],
        [200, 600],
        [300,700]
    ])
    
    analyzer = SalesAnalyzer(sales)

    print(analyzer.get_average_sales())

    print(analyzer.center_data())