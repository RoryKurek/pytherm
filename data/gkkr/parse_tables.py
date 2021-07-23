import pandas as pd
import json


def add_corrs(d: dict, table_path: str, corr_type: str, source: str, corr_name: str):
    table = pd.read_csv(table_path, index_col='Substance')
    table['Type'] = corr_type
    table['Source'] = source
    for substance, row in table.iterrows():
        d[substance].setdefault('Correlations', {}).setdefault(corr_name, []).append(row.to_dict())


if __name__ == '__main__':
    main_table = pd.read_csv('general_props.csv', index_col='Substance')
    main_table['Source'] = 'GKKR book'
    main_dict = main_table.to_dict(orient='index')

    # TODO: Fix units in general properties

    add_corrs(main_dict, table_path='vapor_pressure.csv', corr_type='Wagner 2.5-5 Form', source='GKKR book', corr_name='Vapor Pressure')
    add_corrs(main_dict, table_path='liquid_density.csv', corr_type='PPDS Liquid Density', source='GKKR book', corr_name='Liquid Density')
    add_corrs(main_dict, table_path='heat_of_vaporization.csv', corr_type='PPDS Heat of Vaporization', source='GKKR book', corr_name='Heat of Vaporization')
    add_corrs(main_dict, table_path='liquid_cp.csv', corr_type='PPDS Liquid Heat Capacity', source='GKKR book', corr_name='Liquid cp')
    add_corrs(main_dict, table_path='liquid_viscosity.csv', corr_type='PPDS Liquid Viscosity', source='GKKR book', corr_name='Liquid Viscosity')
    add_corrs(main_dict, table_path='ideal_vapor_viscosity.csv', corr_type='DIPPR Ideal Vapor Viscosity', source='GKKR book', corr_name='Ideal Vapor Viscosity')
    add_corrs(main_dict, table_path='vapor_thermal_conductivity.csv', corr_type='PPDS Vapor Thermal Conductivity', source='GKKR book', corr_name='Vapor Thermal Conductivity')
    add_corrs(main_dict, table_path='surface_tension.csv', corr_type='Watson', source='GKKR book', corr_name='Surface Tension')

    table_pt1 = pd.read_csv('ideal_cp_pt1.csv', index_col='Substance')
    table_pt2 = pd.read_csv('ideal_cp_pt2.csv', index_col='Substance')
    for substance, row in table_pt1.iterrows():
        corr_dict = {'Source': 'GKKR book'}
        corr_dict.update(row['A':'D'].to_dict())
        if row['Equation number'] == 'A.1':
            corr_dict['Type'] = 'PPDS Ideal cp'
            indices = ['E', 'F', 'G', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        elif row['Equation number'] == 'A.2':
            corr_dict['Type'] = 'Aly-Lee'
            indices = ['E', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        else:
            raise ValueError

        corr_dict.update(table_pt2.loc[substance][indices].to_dict())
        main_dict[substance].setdefault('Correlations', {}).setdefault('Ideal Vapor cp', []).append(corr_dict)

    table = pd.read_csv('liquid_thermal_conductivity.csv', index_col='Substance')
    for substance, row in table.iterrows():
        corr_dict = {'Source': 'GKKR book'}
        if row['Equation number'] == 'A.3':
            corr_dict['Type'] = 'Jamieson'
            indices = ['A', 'B', 'C', 'D', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        elif row['Equation number'] == 'A.4':
            corr_dict['Type'] = 'Polynomial'
            indices = ['A', 'B', 'C', 'D', 'E', 'T_min', 'T_max', 'T_verify', 'prop_verify']
        else:
            raise ValueError

        corr_dict.update(row[indices].to_dict())
        main_dict[substance].setdefault('Correlations', {}).setdefault('Liquid Thermal Conductivity', []).append(corr_dict)

    with open('../gkkr.json', 'w') as f:
        print(json.dump(main_dict, f, indent=2))
