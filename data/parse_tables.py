import pandas as pd


if __name__ == '__main__':
    main_dict = pd.read_csv('table-a-1.csv', index_col='Substance').to_dict(orient='index')

    Ps_table = pd.read_csv('table-a-2.csv', index_col='Substance')
