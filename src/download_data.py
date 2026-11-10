
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def download_data():
    """Загружает данные о покупках из PostgreSQL и сохраняет в CSV."""

    load_dotenv()

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PWD"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        database=os.getenv("DB_NAME"),
        query={"sslmode": "require"}
    )

    engine = create_engine(connection_url)

    query = '''
    SELECT 
        p.user_id, p.device_type_canonical, p.order_id, p.created_dt_msk as order_dt, p.created_ts_msk as order_ts, p.currency_code, p.revenue, p.tickets_count,
        (p.created_dt_msk::date - LAG(p.created_dt_msk::date) OVER (PARTITION BY p.user_id ORDER BY p.created_dt_msk::date))::integer AS days_since_prev, p.event_id, e.event_name_code as event_name, e.event_type_main, service_name, r.region_name, c.city_name
    FROM 
        afisha.purchases p 
    INNER JOIN afisha.events e on p.event_id = e.event_id
    INNER JOIN afisha.city c on e.city_id = c.city_id
    INNER JOIN afisha.regions r on c.region_id = r.region_id
    WHERE 
        device_type_canonical IN ('mobile', 'desktop')
        AND event_type_main not in ('фильм')
    ORDER BY 
        user_id ASC
    '''

    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(text(query), connection)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/orders.csv", index=False)

        print("Данные успешно загружены!")
        print("Размер таблицы:", df.shape)
        print("Файл сохранён: data/orders.csv")

    finally:
        engine.dispose()


if __name__ == "__main__":
    download_data()
