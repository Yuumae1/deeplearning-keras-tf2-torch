class Company:
    def __init__(self, sales, cost, persons):
        self.sales = sales
        self.cost = cost
        self.persons = persons

    def get_profit(self):
        return self.sales - self.cost


company_A = Company(100, 80, 10)    # Companyクラスのインスタンスを生成、初期化。引数は sales, cost, persons で変数となる部分
company_B = Company(40, 60, 20)

print(company_A.sales)      # Companyクラスのインスタンスの sales 変数を表示
print(company_A.get_profit())

sales_A = company_A.sales
sales_B = company_B.sales
print(sales_A, sales_B)