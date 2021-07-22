import pandas as pd
import json


def add_corrs(d: dict, table_path: str, corr_type: str, source: str, corr_name: str):
    table = pd.read_csv(table_path, index_col='Substance')
    table['Type'] = corr_type
    table['Source'] = source
    for substance, row in table.iterrows():
        d[substance].setdefault(corr_name, []).append(row.to_dict())


if __name__ == '__main__':
    main_table = pd.read_csv('general-props.csv', index_col='Substance')
    main_table['Source'] = 'GKKR book'
    main_dict = main_table.to_dict(orient='index')

    add_corrs(main_dict, table_path='Psat.csv', corr_type='Wagner5', source='GKKR book', corr_name='Psat_corr')
    add_corrs(main_dict, table_path='rhoL.csv', corr_type='PPDSLiqDens', source='GKKR book', corr_name='rhoL_corr')
    add_corrs(main_dict, table_path='h_vap.csv', corr_type='PPDSHVap', source='GKKR book', corr_name='h_vap_corr')
    add_corrs(main_dict, table_path='cpL.csv', corr_type='PPDSCpL', source='GKKR book', corr_name='cpL_corr')
    add_corrs(main_dict, table_path='etaL.csv', corr_type='PPDSetaL', source='GKKR book', corr_name='etaL_corr')
    add_corrs(main_dict, table_path='eta_id.csv', corr_type='DIPPReta_id', source='GKKR book', corr_name='eta_id_corr')

    table_pt1 = pd.read_csv('cp_id-pt1.csv', index_col='Substance')
    table_pt2 = pd.read_csv('cp_id-pt2.csv', index_col='Substance')
    for substance, row in table_pt1.iterrows():
        corr_dict = {'Source': 'GKKR book'}
        corr_dict.update(row['A':'D'].to_dict())
        if row['Equation number'] == 'A.1':
            corr_dict['Type'] = 'PPDSCp_id'
            indices = ['E', 'F', 'G', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        elif row['Equation number'] == 'A.2':
            corr_dict['Type'] = 'AlyLee'
            indices = ['E', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        else:
            raise ValueError

        corr_dict.update(table_pt2.loc[substance][indices].to_dict())
        main_dict[substance].setdefault('cp_id_corr', []).append(corr_dict)

    table = pd.read_csv('lambdaL.csv', index_col='Substance')
    for substance, row in table.iterrows():
        corr_dict = {'Source': 'GKKR book'}
        if row['Equation number'] == 'A.3':
            corr_dict['Type'] = 'Jamieson'
            indices = ['A', 'B', 'C', 'D', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        elif row['Equation number'] == 'A.4':
            corr_dict['Type'] = 'Poly4thDeg'
            indices = ['A', 'B', 'C', 'D', 'E', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        elif row['Equation number'] == 'A.4B':
            corr_dict['Type'] = 'Poly1stDeg'
            indices = ['A', 'B', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        else:
            raise ValueError

        corr_dict.update(row[indices].to_dict())
        main_dict[substance].setdefault('lambdaL_corr', []).append(corr_dict)

    print(json.dumps(main_dict, indent=2))
