
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
    print('- Доля повторных покупателей составляет около 61%.')
    print('- Наиболее высокая доля возврата наблюдается у покупателей,')
    print('  начавших с выставок и театральных мероприятий.')
    print('- В активных регионах доля повторных покупателей выше.')
    print('- Медианная выручка повторных покупателей выше,')
    print('  чем покупателей с одним заказом.')

    print('=' * 60)


if __name__ == '__main__':
    main()
    