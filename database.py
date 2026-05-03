from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:password@db:5432/churn_db"

engine = create_engine(DATABASE_URL)