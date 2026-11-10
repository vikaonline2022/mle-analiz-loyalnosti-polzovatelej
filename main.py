
from src.processing import DataProcessor


def main():
    """Запускает полный цикл обработки и анализа данных."""

    processor = DataProcessor()

    # 1. Загрузка и предобработка данных
    processor.load_data()
    processor.load_currency_rates()
    processor.preprocess_data()

    # 2. Создание и очистка профиля
    processor.create_user_profile()
    processor.clean_user_profile()

    # 3. Анализ Retention
    processor.calculate_retention()

    # 4. Продуктовые гипотезы
    processor.analyze_product_hypotheses()

    # 5. Анализ выручки
    processor.analyze_revenue_distribution()
    processor.plot_revenue_distribution()

    # 6. Корреляционный анализ
    processor.analyze_phik()
    processor.analyze_order_groups()

    # 7. Итоговый отчёт
    print('\n' + '=' * 60)
    print('ИТОГОВЫЙ ОТЧЁТ')
    print('=' * 60)

    print(f'Количество пользователей: {len(processor.user_profile)}')

    repeat_share = (
        processor.user_profile['is_two'].mean() * 100
    )
    print(f'Доля повторных покупателей: {repeat_share:.2f}%')

    print(
        f"Средняя выручка на пользователя: "
        f"{processor.user_profile['avg_revenue'].mean():.2f} руб."
    )

    print(
        f"Медианная выручка на пользователя: "
        f"{processor.user_profile['avg_revenue'].median():.2f} руб."
    )

    print(
        f"Среднее количество заказов: "
        f"{processor.user_profile['orders_count'].mean():.2f}"
    )

   
    print('\nОсновные результаты:')

    # Доля повторных покупателей
    print(
        f'- Доля повторных покупателей составила '
        f'{repeat_share:.2f}%.'
    )

    # Мероприятие с максимальной долей повторных покупок
    event_stats = processor.retention_results['first_event_type']

    best_event = event_stats.loc[
        event_stats['return_share'].idxmax()
    ]

    print(
        f"- Наибольшая доля повторных покупок наблюдается "
        f"среди пользователей, начавших с категории "
        f"«{best_event['first_event_type']}»: "
        f"{best_event['return_share']:.2f}%."
    )

    # Сравнение активных и менее активных регионов
    region_stats = (
        processor.hypothesis_results['regions']
        .set_index('region_group')['return_share']
    )

    active_share = region_stats['Активные регионы']
    less_active_share = region_stats['Менее активные регионы']
    region_difference = active_share - less_active_share

    print(
        f'- Доля повторных покупателей в активных регионах: '
        f'{active_share:.2f}%, в менее активных: '
        f'{less_active_share:.2f}%. Разница: '
        f'{region_difference:.2f} п.п.'
    )

    # Сравнение медианной выручки
    revenue_stats = processor.revenue_results

    one_order_median = revenue_stats.loc[
        'Один заказ', 'median_revenue'
    ]

    repeat_median = revenue_stats.loc[
        'Повторные заказы', 'median_revenue'
    ]

    median_difference = repeat_median - one_order_median

    print(
        f'- Медианная выручка повторных покупателей: '
        f'{repeat_median:.2f} руб., покупателей с одним заказом: '
        f'{one_order_median:.2f} руб. Разница: '
        f'{median_difference:.2f} руб.'
    )


    print('=' * 60)


if __name__ == '__main__':
    main()
    