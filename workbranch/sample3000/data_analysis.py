import pandas as pd
import json
import xlrd
from icecream import ic


def extract_data(*args):
    df = pd.read_excel('store_visit_report.xls')
    print(df.columns)
    df_t = df[df['is_exist'] == 't'].copy()
    column_dict = {}
    for col in args:
        column_dict[col] = list()
    df_t["report_json"].apply(getname, args=(column_dict,))
    for col in args:
        df_t[col] = column_dict[col]
    concat = pd.concat((df_t, df[df['is_exist'] == 'f']), axis=0)
    concat.to_csv('new_sv_report.csv')


def getname(df, col_dict):
    value = json.loads(df)
    for k, v in col_dict.items():
        v.append(value[k])


if __name__ == '__main__':
    extract_data('name', 'storeType', 'drinkTypes', 'stackingState')
