import pandas as pd
import numpy as np
import gc


# Функция для проверки типа данных: float int, функция ищет знаки после точки, если в них есть что-то отличное от
# нуля, значит это flaot, если нет то int
def check_fraction(list_of_numbers: list):
    list_of_numbers = [str(i) for i in list_of_numbers]
    flag = 0
    for i in list_of_numbers:
        if '.' in i:
            integer_part, fractional_part = i.split('.')
            # Проверяем, есть ли в дробной части число не равное нулю
            if int(fractional_part) != 0:
                flag += 1
            else:
                pass
    if flag == 0:
        return False, float(max(list_of_numbers)), float(min(list_of_numbers))
    else:
        return True, float(max(list_of_numbers)), float(min(list_of_numbers))


# Функция для оптимизации типа данных в датафрейме, поиск идет по предельным значениям, соответсвующим границам типов данных, идет в связке с функцией check_fraction
def type_transform(df: pd.DataFrame):
    unit8 = []
    unit16 = []
    unit32 = []
    int8 = []
    int16 = []
    int32 = []
    float32 = []
    df['id'] = df['id'].astype('uint32')
    df['flag'] = df['flag'].astype('uint8')
    for i in [i for i in df.columns if i not in ['id', 'flag']]:
        if check_fraction(list(df[i].unique()))[0] == True:
            float32.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 255 and check_fraction(list(df[i].unique()))[2] >= 0:
            unit8.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 65535 and check_fraction(list(df[i].unique()))[2] >= 0:
            unit16.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 4294967295 and check_fraction(list(df[i].unique()))[2] >= 0:
            unit32.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 127 and check_fraction(list(df[i].unique()))[2] >= -128:
            int8.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 32767 and check_fraction(list(df[i].unique()))[2] >= -32767:
            int16.append(i)
        elif check_fraction(list(df[i].unique()))[1] <= 2147483647 and check_fraction(list(df[i].unique()))[
            2] >= -2147483647:
            int32.append(i)
    if len(unit8) != 0:
        df[unit8] = df[unit8].astype('uint8')
    else:
        pass
    if len(unit16) != 0:
        df[unit16] = df[unit16].astype('uint16')
    else:
        pass
    if len(unit32) != 0:
        df[unit32] = df[unit32].astype('uint32')
    else:
        pass
    if len(int8) != 0:
        df[int8] = df[int8].astype('int8')
    else:
        pass
    if len(int16) != 0:
        df[int16] = df[unit8].astype('int16')
    else:
        pass
    if len(int32) != 0:
        df[int32] = df[unit8].astype('int32')
    else:
        pass
    if len(float32) != 0:
        df[float32] = df[float32].astype('float32')
    else:
        pass
    return df


# функция проверяющая кол-во выпавших из границы int32 данных
def validate_for_reduction(df):
    df = df.drop(columns='id')
    min_int32 = -32768
    max_int32 = 32767
    possibble_overflow = 0
    for col in df.columns:
        if df[col].min() < min_int32 or df[col].max() > max_int32:
            possibble_overflow += 1
    return possibble_overflow


# функция создания валидационной выборки
def find_spec(df: pd.DataFrame, n_samples=30000):
    num_samples = n_samples
    prop = {}
    for i, j in zip(df['flag'].value_counts(normalize=True).keys().values,
                    df['flag'].value_counts(normalize=True).values):
        prop[i] = j

    groups = df.groupby('flag')

    selected_rows = []

    for flag, group in groups:
        num_samples_group = int(num_samples * prop[flag])
        if num_samples_group >= len(group):
            selected_rows += group.index.tolist()
        else:
            selected_rows += np.random.choice(group.index, num_samples_group, replace=False).tolist()

    random_df = df.loc[selected_rows]
    return random_df


target = pd.read_csv('/data/train_target.csv')
df = pd.DataFrame()
is_zero_loans = [
    'is_zero_loans5',
    'is_zero_loans530',
    'is_zero_loans3060',
    'is_zero_loans6090',
    'is_zero_loans90'
]
# Изначально из датафрейма удаляются столбцы с pre_loans_total_overdue, далее находится кол-во кредитов у клиента,
# данные присоединяются к изначальной выборке
for num in range(0, 12):
    tmp_df = pd.read_parquet(f'G:/train_data/train_data_{num}.pq')
    tmp_df = tmp_df.drop(['pre_loans_total_overdue'], axis=1)
    df_id_max = tmp_df[['id', 'rn']].groupby('id', as_index=False).max('rn').rename(columns={'rn': 'count_of_loans'})
    tmp_df = tmp_df.merge(df_id_max, on='id')
    tmp_df[is_zero_loans] = tmp_df[is_zero_loans].replace([0, 1], [1, 0])
    # Иходятся средние описательные статистики по данным enc_paym, данные добоволяются в соответсвующие столбцы
    enc_ = [i for i in tmp_df.columns if 'enc_paym' in i]
    means = tmp_df[enc_].mean(axis=1)
    median = tmp_df[enc_].median(axis=1)
    std = tmp_df[enc_].std(axis=1)
    tmp_df['enc_paym_means'] = means
    tmp_df['enc_paym_median'] = median
    tmp_df['enc_paym_std'] = std
    # Создается список из фичей, на которые нужно использовать кодирование
    df_court_dumm = [i for i in tmp_df.columns if
                     i not in ['id', 'rn', 'count_of_loans', 'enc_paym_std', 'enc_paym_means', 'enc_paym_median']]
    df_court_dumm_id = [i for i in tmp_df.columns if
                        i not in ['rn', 'count_of_loans', 'enc_paym_std', 'enc_paym_means', 'enc_paym_median']]
    df_court_dummes = pd.get_dummies(tmp_df[df_court_dumm_id], columns=df_court_dumm)
    df_court_dummes_generated = [i for i in df_court_dummes.columns if i not in ['id']]

    # Первичная работа с get_dummes
    tmp_df = pd.concat([tmp_df, df_court_dummes.drop('id', axis=1)], axis=1)
    tmp_df = tmp_df.drop(df_court_dumm, axis=1)

    # Создание фичей с влиянием количества кредитов у клиента

    tmp_df['weight'] = tmp_df['rn'] / tmp_df['count_of_loans']
    df_total_weight = tmp_df.groupby('id')['weight'].sum().reset_index(drop=False).rename(
        columns={'weight': 'total_weight'})
    tmp_df = tmp_df.merge(df_total_weight, on='id')

    generated_features = [i for i in tmp_df.columns if i not in ['id', 'rn', 'count_of_loans'
                                                                             'weight', 'total_weight', 'enc_paym_std',
                                                                 'enc_paym_means', 'enc_paym_median']]
    sec_gen_features = ['enc_paym_std', 'enc_paym_means', 'enc_paym_median']

    enc_p_df = tmp_df[['id', 'enc_paym_std', 'enc_paym_means', 'enc_paym_median']]

    for feature in generated_features:
        tmp_df[feature] = tmp_df[feature] * tmp_df['weight'] / tmp_df['total_weight']

    # Непосредственно создание финального датасета, по каждой фиче у клиента высчитывается количество соответсвующих
    # значений

    tmp_df.groupby("id")
    enc_p_df = enc_p_df.groupby("id")[sec_gen_features].mean().reset_index(drop=False)
    tmp_df = tmp_df.groupby("id")[generated_features].sum().reset_index(drop=False)
    df_court_dummes = df_court_dummes.groupby("id")[df_court_dummes_generated].sum().reset_index(drop=False)
    tmp_df = tmp_df.merge(df_court_dummes, on='id')
    tmp_df = tmp_df.merge(df_total_weight, on='id')
    tmp_df = tmp_df.merge(enc_p_df, on='id')
    df = pd.concat([df, tmp_df])
    print(f'dataframe number {num} sucesfully concatenated.')
df.fillna(np.uint8(0), inplace=True)
df = type_transform(df)
gc.collect()

df.to_parquet(f'/data/train_target.pq')
