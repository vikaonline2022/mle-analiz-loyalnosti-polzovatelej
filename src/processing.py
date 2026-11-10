
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataProcessor:
    """Класс для загрузки и предобработки данных о покупках."""

    def __init__(self):
        self.df = None
        self.tenge_df = None
        self.user_profile = None
        self.user_profile_original = None
        self.df_original = None
        self.retention_results = None
        self.hypothesis_results = None
    

    def load_data(self, file_path='data/orders.csv'):
        """Загружает данные о заказах."""

        self.df = pd.read_csv(file_path)

        return self.df

    def load_currency_rates(
        self,
        file_path='data/final_tickets_tenge_df.csv'
    ):
        """Загружает данные о курсах валют."""

        self.tenge_df = pd.read_csv(file_path)

        return self.tenge_df

    def preprocess_data(self):
        """Выполняет предобработку данных и переводит выручку в рубли."""

        if self.df is None:
            raise ValueError('Сначала загрузите данные о заказах.')

        if self.tenge_df is None:
            raise ValueError('Сначала загрузите курсы валют.')

        df = self.df.copy()

        # Очистка строковых значений от пробелов по краям
        for column in df.select_dtypes(include='object').columns:
            df[column] = df[column].str.strip()

        # Преобразование дат
        df['order_dt'] = pd.to_datetime(df['order_dt'])
        df['order_ts'] = pd.to_datetime(df['order_ts'])

        # Преобразование числовых столбцов
        df['days_since_prev'] = df['days_since_prev'].astype('Int64')
        df['order_id'] = pd.to_numeric(
            df['order_id'], downcast='integer'
        )
        df['event_id'] = pd.to_numeric(
            df['event_id'], downcast='integer'
        )
        df['tickets_count'] = pd.to_numeric(
            df['tickets_count'], downcast='integer'
        )

        # Подготовка курсов тенге
        tenge_rates = self.tenge_df.loc[
            self.tenge_df['cdx'] == 'kzt',
            ['data', 'nominal', 'curs']
        ].copy()

        tenge_rates['data'] = (
            pd.to_datetime(tenge_rates['data']).dt.normalize()
        )

        df['order_dt'] = df['order_dt'].dt.normalize()

        # Объединение заказов с курсами валют
        df = df.merge(
            tenge_rates,
            left_on='order_dt',
            right_on='data',
            how='left',
            validate='many_to_one'
        )

        # Перевод выручки в рубли
        df['revenue_rub'] = df['revenue']

        mask = df['currency_code'] == 'kzt'

        df.loc[mask, 'revenue_rub'] = (
            df.loc[mask, 'revenue']
            * df.loc[mask, 'curs']
            / df.loc[mask, 'nominal']
        )

        print('Количество покупок в тенге:', mask.sum())
        print(
            'Покупок в тенге без найденного курса:',
            df.loc[mask, 'curs'].isna().sum()
        )

        # Удаление временных столбцов курсов
        df = df.drop(columns=['data', 'nominal', 'curs'])

        # Фильтрация выбросов по 99-му перцентилю
        rows_before = len(df)

        revenue_99 = df['revenue_rub'].quantile(0.99)

        df = df[df['revenue_rub'] <= revenue_99].copy()

        print('99-й перцентиль выручки:', revenue_99)
        print('Строк до фильтрации:', rows_before)
        print('Строк после фильтрации:', len(df))
        print('Удалено строк:', rows_before - len(df))

        
        # Удаление неявных дубликатов
        duplicate_cols = [
            'user_id',
            'order_ts',
            'device_type_canonical',
            'service_name',
            'event_id',
            'currency_code',
            'revenue',
            'tickets_count'
        ]

        rows_before_duplicates = len(df)

        df = df.drop_duplicates(
            subset=duplicate_cols,
            keep='first'
        ).copy()

        print(
            'Удалено неявных дубликатов:',
            rows_before_duplicates - len(df)
        )

        print('Строк после удаления дубликатов:', len(df))

        print(
            'Оставшиеся неявные дубликаты:',
            df.duplicated(subset=duplicate_cols).sum()
        )

        print(
            'Повторяющиеся order_id:',
            df['order_id'].duplicated().sum()
        )

        self.df = df

        return self.df

    
    def create_user_profile(self):
        """Создаёт профиль каждого пользователя."""

        if self.df is None:
            raise ValueError(
                'Сначала выполните предобработку данных.'
            )

        # Сортируем данные и оставляем уникальные заказы
        orders_unique = (
            self.df.sort_values(
                ['user_id', 'order_ts', 'order_id']
            )
            .drop_duplicates(
                subset=['user_id', 'order_id']
            )
            .copy()
        )

        # Рассчитываем интервалы между заказами
        orders_unique['days_between_orders'] = (
            orders_unique.groupby('user_id')['order_dt']
            .diff()
            .dt.days
        )

        # Создаём профиль пользователя
        user_profile = orders_unique.groupby('user_id').agg(
            first_order_dt=('order_dt', 'first'),
            last_order_dt=('order_dt', 'last'),
            first_device=('device_type_canonical', 'first'),
            first_region=('region_name', 'first'),
            first_event_type=('event_type_main', 'first'),
            orders_count=('order_id', 'nunique'),
            avg_revenue=('revenue_rub', 'mean'),
            avg_days_between_orders=('days_between_orders', 'mean')
        ).reset_index()

        # Добавляем признак повторных покупок
        user_profile['is_two'] = (
            user_profile['orders_count'] >= 2
        ).astype('int8')

                # Определяем день недели первой покупки
        weekday_names = {
            0: 'Понедельник',
            1: 'Вторник',
            2: 'Среда',
            3: 'Четверг',
            4: 'Пятница',
            5: 'Суббота',
            6: 'Воскресенье'
        }

        user_profile['first_order_weekday'] = (
            user_profile['first_order_dt']
            .dt.dayofweek
            .map(weekday_names)
        )

        self.user_profile = user_profile

        print(
            'Количество пользователей:',
            user_profile['user_id'].nunique()
        )

        print(
            'Доля пользователей с повторными заказами:',
            f"{user_profile['is_two'].mean() * 100:.2f}%"
        )

        return self.user_profile

    
    def clean_user_profile(self):
        """Исключает аномальные профили из основной выборки."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте профиль пользователей.'
            )

        # Сохраняем исходные данные
        if self.user_profile_original is None:
            self.user_profile_original = self.user_profile.copy()

        if self.df_original is None:
            self.df_original = self.df.copy()

        original_profile = self.user_profile_original.copy()
        original_df = self.df_original.copy()

        # Определяем 99-й перцентиль количества заказов
        orders_threshold = original_profile[
            'orders_count'
        ].quantile(0.99)

        high_activity = (
            original_profile['orders_count'] > orders_threshold
        )

        negative_avg_revenue = (
            original_profile['avg_revenue'] < 0
        )

        # Проверяем профили с отрицательной средней выручкой
        negative_users = original_profile.loc[
            negative_avg_revenue, 'user_id'
        ]

        negative_orders = original_df[
            original_df['user_id'].isin(negative_users)
        ]

        negative_orders_check = (
            negative_orders.groupby('user_id')
            .agg(
                negative_orders=(
                    'revenue_rub', lambda x: (x < 0).sum()
                ),
                zero_orders=(
                    'revenue_rub', lambda x: (x == 0).sum()
                ),
                positive_orders=(
                    'revenue_rub', lambda x: (x > 0).sum()
                )
            )
            .reset_index()
        )

        # Формируем очищенную выборку
        valid_profiles = (
            (original_profile['orders_count'] <= orders_threshold)
            & (original_profile['avg_revenue'] >= 0)
        )

        filtered_profile = original_profile.loc[
            valid_profiles
        ].copy()

        excluded_users = original_profile.loc[
            ~valid_profiles, 'user_id'
        ]

        filtered_df = original_df[
            original_df['user_id'].isin(
                filtered_profile['user_id']
            )
        ].copy()

        # Сравниваем показатели до и после очистки
        def profile_metrics(profile):
            return pd.Series({
                'Количество пользователей':
                    profile['user_id'].nunique(),
                'Доля повторных покупателей, %':
                    profile['is_two'].mean() * 100,
                'Среднее количество заказов':
                    profile['orders_count'].mean(),
                'Медианное количество заказов':
                    profile['orders_count'].median(),
                'Максимальное количество заказов':
                    profile['orders_count'].max(),
                'Средняя выручка на пользователя, руб.':
                    profile['avg_revenue'].mean(),
                'Медианная выручка на пользователя, руб.':
                    profile['avg_revenue'].median(),
                'Средний интервал между заказами, дней':
                    profile['avg_days_between_orders'].mean()
            })

        comparison = pd.DataFrame({
            'До очистки': profile_metrics(original_profile),
            'После очистки': profile_metrics(filtered_profile)
        })

        # Сохраняем очищенные данные
        self.user_profile = filtered_profile
        self.df = filtered_df

        # Выводим результаты
        total_users = len(original_profile)
        excluded_count = len(excluded_users)

        print('\nОчистка аномальных профилей')
        print('99-й перцентиль заказов:', orders_threshold)
        print('Профилей с высокой активностью:', high_activity.sum())
        print(
            'Профилей с отрицательной средней выручкой:',
            negative_avg_revenue.sum()
        )
        print(
            'Пользователей с отрицательной средней выручкой:',
            len(negative_orders_check)
        )
        print(
            'Исключено пользователей:',
            excluded_count
        )
        print(
            'Доля исключённых пользователей:',
            f'{excluded_count / total_users * 100:.2f}%'
        )
        print('Размер очищенного профиля:', self.user_profile.shape)
        print('Размер очищенных заказов:', self.df.shape)

        print('\nСравнение показателей до и после очистки:')
        print(comparison.round(2).to_string())

        print('\nПроверка заказов пользователей с отрицательной выручкой:')
        print(negative_orders_check.head(20).to_string(index=False))

        return self.user_profile



    
    def calculate_retention(self):
        """Рассчитывает долю повторных покупателей по сегментам."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте профиль пользователей.'
            )

        user_profile = self.user_profile.copy()

        # Средняя доля повторных покупателей
        overall_return_share = (
            user_profile['is_two'].mean() * 100
        )

        print(
            '\nСредняя доля повторных покупателей:',
            f'{overall_return_share:.2f}%'
        )

        # Расчёт возвратов по сегментам
        results = {}

        for column in [
            'first_event_type',
            'first_device',
            'first_region'
        ]:
            stats = (
                user_profile
                .groupby(column)
                .agg(
                    users_count=('user_id', 'nunique'),
                    repeat_users=('is_two', 'sum')
                )
                .reset_index()
            )

            stats['return_share'] = (
                stats['repeat_users']
                / stats['users_count'] * 100
            )

            stats['above_average'] = (
                stats['return_share'] > overall_return_share
            )

            stats = stats.sort_values(
                'users_count', ascending=False
            )

            results[column] = stats

            print(f'\nВозвраты по признаку {column}:')
            print(stats.head(10).to_string(index=False))

        
        # Retention по дням недели первой покупки
        weekday_stats = (
            user_profile
            .groupby('first_order_weekday')
            .agg(
                users_count=('user_id', 'nunique'),
                repeat_users=('is_two', 'sum')
            )
            .reset_index()
        )

        weekday_stats['return_share'] = (
            weekday_stats['repeat_users']
            / weekday_stats['users_count'] * 100
        )

        weekday_order = [
            'Понедельник',
            'Вторник',
            'Среда',
            'Четверг',
            'Пятница',
            'Суббота',
            'Воскресенье'
        ]

        weekday_stats['first_order_weekday'] = pd.Categorical(
            weekday_stats['first_order_weekday'],
            categories=weekday_order,
            ordered=True
        )

        weekday_stats = weekday_stats.sort_values(
            'first_order_weekday'
        )

        results['first_order_weekday'] = weekday_stats

        print('\nВозвраты по дням недели первой покупки:')
        print(weekday_stats.to_string(index=False))

        self.retention_results = results

        return results

    
    def analyze_product_hypotheses(self):
        """Проверяет продуктовые гипотезы по повторным покупкам."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте и очистите профиль пользователей.'
            )

        user_profile = self.user_profile.copy()

        # Гипотеза 1. Сравнение спорта и концертов
        sport_concert = user_profile[
            user_profile['first_event_type'].isin(
                ['спорт', 'концерты']
            )
        ].copy()

        sport_stats = (
            sport_concert.groupby('first_event_type')
            .agg(
                repeat_users=('is_two', 'sum'),
                users_count=('is_two', 'count')
            )
        )

        sport_stats['return_share'] = (
            sport_stats['repeat_users']
            / sport_stats['users_count'] * 100
        )

        print('\nГипотеза 1. Спорт и концерты:')
        print(sport_stats.round(2).to_string())

        if {'спорт', 'концерты'}.issubset(sport_stats.index):
            difference = (
                sport_stats.loc['спорт', 'return_share']
                - sport_stats.loc['концерты', 'return_share']
            )

            print(
                '\nРазница долей возврата (спорт минус концерты):',
                f'{difference:.2f} п.п.'
            )

        # Гипотеза 2. Сравнение активных и менее активных регионов
        top_10_regions = (
            user_profile['first_region']
            .value_counts()
            .head(10)
            .index
        )

        user_profile['region_group'] = user_profile[
            'first_region'
        ].apply(
            lambda x: 'Активные регионы'
            if x in top_10_regions
            else 'Менее активные регионы'
        )

        region_stats = (
            user_profile.groupby('region_group')
            .agg(
                repeat_users=('is_two', 'sum'),
                users_count=('is_two', 'count'),
                return_share=('is_two', 'mean')
            )
            .reset_index()
        )

        region_stats['return_share'] *= 100

        print('\nГипотеза 2. Активные и менее активные регионы:')
        print(region_stats.round(2).to_string(index=False))

        self.hypothesis_results = {
            'sport_concert': sport_stats,
            'regions': region_stats
        }

        return self.hypothesis_results

    
    def analyze_revenue_distribution(self):
        """Сравнивает среднюю выручку пользователей с одним и повторными заказами."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте и очистите профиль пользователей.'
            )

        revenue_stats = (
            self.user_profile.groupby('is_two')['avg_revenue']
            .agg(
                users_count='count',
                mean_revenue='mean',
                median_revenue='median',
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75)
            )
            .rename(index={
                0: 'Один заказ',
                1: 'Повторные заказы'
            })
        )

        print('\nСредняя выручка по типу покупателей:')
        print(revenue_stats.round(2).to_string())

        self.revenue_results = revenue_stats

        return revenue_stats

    
    def plot_revenue_distribution(self):
        """Строит гистограммы средней выручки для двух групп покупателей."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте и очистите профиль пользователей.'
            )

        one_order = self.user_profile.loc[
            self.user_profile['is_two'] == 0, 'avg_revenue'
        ]

        repeat_orders = self.user_profile.loc[
            self.user_profile['is_two'] == 1, 'avg_revenue'
        ]

        bins = np.histogram_bin_edges(
            self.user_profile['avg_revenue'],
            bins=30
        )

        plt.figure(figsize=(12, 6))

        plt.hist(
            one_order,
            bins=bins,
            alpha=0.6,
            density=True,
            label='Один заказ'
        )

        plt.hist(
            repeat_orders,
            bins=bins,
            alpha=0.6,
            density=True,
            label='Повторные заказы'
        )

        plt.title('Распределение средней выручки пользователей')
        plt.xlabel('Средняя выручка, руб.')
        plt.ylabel('Плотность')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()


    def analyze_phik(self):
        """Рассчитывает корреляции Phi-k между характеристиками пользователей."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте и очистите профиль пользователей.'
            )

        from phik import phik_matrix

        features = self.user_profile[
            [
                'orders_count',
                'first_device',
                'first_region',
                'first_event_type',
                'avg_revenue',
                'avg_days_between_orders'
            ]
        ].copy()

        features = features.rename(
            columns={'orders_count': 'total_orders'}
        )

        interval_cols = [
            'total_orders',
            'avg_revenue',
            'avg_days_between_orders'
        ]

        correlation_matrix = features.phik_matrix(
            interval_cols=interval_cols
        )

        print('\nМатрица корреляций Phi-k:')
        print(correlation_matrix.round(2).to_string())

        self.phik_results = correlation_matrix

        return correlation_matrix

    
    def analyze_order_groups(self):
        """Группирует пользователей по количеству заказов и анализирует связи."""

        if self.user_profile is None:
            raise ValueError(
                'Сначала создайте и очистите профиль пользователей.'
            )

        from phik import phik_matrix

        profile = self.user_profile.copy()

        profile['orders_group'] = pd.cut(
            profile['orders_count'],
            bins=[0, 1, 2, 4, float('inf')],
            labels=['1 заказ', '2 заказа', '3–4 заказа', '5+ заказов']
        )

        group_stats = (
            profile.groupby('orders_group', observed=False)
            .agg(
                users_count=('orders_count', 'count'),
                avg_revenue=('avg_revenue', 'mean'),
                median_revenue=('avg_revenue', 'median'),
                avg_days_between_orders=(
                    'avg_days_between_orders', 'mean'
                )
            )
        )

        print('\nХарактеристики групп пользователей:')
        print(group_stats.round(2).to_string())

        features = profile[
            [
                'orders_group',
                'first_device',
                'first_region',
                'first_event_type',
                'avg_revenue',
                'avg_days_between_orders'
            ]
        ]

        correlation_matrix = features.phik_matrix(
            interval_cols=[
                'avg_revenue',
                'avg_days_between_orders'
            ]
        )

        print('\nМатрица Phi-k с учётом групп заказов:')
        print(correlation_matrix.round(2).to_string())

        self.order_group_results = group_stats
        self.order_group_phik_results = correlation_matrix

        return group_stats, correlation_matrix








