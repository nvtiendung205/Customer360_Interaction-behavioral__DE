import os, json
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, lower, trim, regexp_replace, when,
    row_number, lit, split, size, length, sum as _sum
)

# ==== CONFIG ====
BASE_LOG_DIR = r"C:\Users\admin\OneDrive\Study_de\Dataset\log_search\log_search"
CONFIG_DIR = r"C:\Users\admin\OneDrive\Study_de\Bigdata\360_project_DE\etl_logsearch\most_search_staging_config"
CONFIG_PATH = os.path.join(CONFIG_DIR, "mysql_staging_config.json")

MYSQL = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "driver": "com.mysql.cj.jdbc.Driver"
}

spark = (SparkSession.builder.appName("etl_mostsearch").config("spark.driver.memory", "8g").getOrCreate())

# ==== HELPERS ====
def mysql_url(db): 
    return f"jdbc:mysql://{MYSQL['host']}:{MYSQL['port']}/{db}"
#==== Đọc File ====
def read_parquet_range(start, end):
    s, e = datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")
    paths = [os.path.join(BASE_LOG_DIR, (s + timedelta(days=i)).strftime("%Y%m%d"))
             for i in range((e - s).days + 1)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError("Không có parquet hợp lệ")
    return spark.read.parquet(*paths)

# ====CLEAN KEYWORD =====
def clean_keyword(df):
    df = (
        df.select("user_id", "keyword").filter(col("user_id").isNotNull()).withColumn("kw", lower(trim(col("keyword"))))
          .filter(col("kw").isNotNull())
          .filter(col("kw") != "")
          .withColumn("kw", regexp_replace(col("kw"), r"\s+", " "))
    )

    letters  = length(regexp_replace(col("kw"), r"[^\p{L}]", ""))
    digits   = length(regexp_replace(col("kw"), r"[^0-9]", ""))
    specials = length(regexp_replace(col("kw"), r"[\p{L}0-9\s]", ""))

    df = (
        df.withColumn("n_letters", letters)
          .withColumn("n_digits", digits)
          .withColumn("n_specials", specials)
          .withColumn("word_count", size(split(col("kw"), " ")))
          .filter(col("word_count") <= 4)
          .filter(col("n_letters") >= 3)
          .filter(col("n_digits") * 4 < col("n_letters"))
          .filter(col("n_specials") * 4 < col("n_letters"))
          .withColumn("keyword_norm", regexp_replace(col("kw"), r"[^\p{L}\p{N}\s]", ""))
          .filter(length(col("keyword_norm")) >= 3)
    )

    return df.select(col("user_id"), col("keyword").alias("keyword_raw"), col("keyword_norm"))

# ==== MOST SEARCH ====
def most_search(df):
    cnt = df.groupBy("user_id", "keyword_norm", "keyword_raw").count()

    top_norm = (cnt.groupBy("user_id", "keyword_norm")
           .agg(_sum("count").alias("s"))
           .withColumn("rn", row_number().over(Window.partitionBy("user_id").orderBy(col("s").desc())))
           .filter(col("rn") == 1)
           .select("user_id", "keyword_norm")
    )

    return (cnt.join(top_norm, ["user_id", "keyword_norm"])
           .withColumn("rn", row_number().over(Window.partitionBy("user_id").orderBy(col("count").desc())))
           .filter(col("rn") == 1)
           .select("user_id", col("keyword_raw").alias("Most_Search"), col("keyword_norm").alias("Most_Search_Norm"))
    )

# ==== MAIN =====
def main():
    print("[START] etl_mostsearch")
    t6 = clean_keyword(read_parquet_range(input("T6 start: "), input("T6 end: ")))
    t7 = clean_keyword(read_parquet_range(input("T7 start: "), input("T7 end: ")))
    print('')
    print('[Most Search] clean keyword')
    m6 = most_search(t6)
    m7 = most_search(t7)
    m6.show(10)
    m7.show(10)
    print('')
    print('[Join Most Search] join and mysql')
    full = (m6.alias("a").join(m7.alias("b"), "user_id", "outer").select("user_id", col("a.Most_Search").alias("most_search_t6"), 
                                                                         col("a.Most_Search_Norm").alias("most_search_norm_t6"),
                                                                         col("b.Most_Search").alias("most_search_t7"),
                                                                         col("b.Most_Search_Norm").alias("most_search_norm_t7")
                                                                         ))

    full.show(20, False)

    if input("Ghi MySQL? (y/n): ").lower() == "y":
        db = input("Database: ")
        table = input("Table: ")
        full.write.format("jdbc") \
            .option("url", mysql_url(db)) \
            .option("driver", MYSQL["driver"]) \
            .option("dbtable", table) \
            .option("user", MYSQL["user"]) \
            .option("password", MYSQL["password"]) \
            .mode("overwrite").save()

        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({**MYSQL, "db_name": db, "table_name": table}, f, indent=2)

    print("[DONE] etl_mostsearch")

if __name__ == "__main__":
    main()
