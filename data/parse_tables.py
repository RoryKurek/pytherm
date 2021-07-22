import pandas as pd
import json


def add_corrs(d: dict, table_path: str, corr_type: str, source: str, corr_name: str):
    table = pd.read_csv(table_path, index_col='Substance')
    table['Type'] = corr_type
    table['Source'] = source
    for substance, row in table.iterrows():
        d[substance].setdefault(corr_name, []).append(row.to_dict())


if __name__ == '__main__':
    main_table = pd.read_csv('table-a-1.csv', index_col='Substance')
    main_table['Source'] = 'GKKR book'
    main_dict = main_table.to_dict(orient='index')

    add_corrs(main_dict, table_path='table-a-2.csv', corr_type='Wagner5', source='GKKR book', corr_name='Ps_corr')
    add_corrs(main_dict, table_path='table-a-3.csv', corr_type='PPDSLiqDens', source='GKKR book', corr_name='rhoL_corr')

    print(json.dumps(main_dict, indent=2))
