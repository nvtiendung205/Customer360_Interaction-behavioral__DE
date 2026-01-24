import os
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col, when, lit, concat_ws, greatest,
    input_file_name, regexp_extract, to_date
)

spark = (
    SparkSession.builder
    .appName("ETL_APP")
    .config("spark.driver.memory", "8g")
    .getOrCreate()
)

# Date helpers
def range_date(start_yyyymmdd: str, end_yyyymmdd: str):
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return out


# Transform steps
def select_fields(df):
    return df.select("_source.*")


def transform_category(df):
    df = df.withColumn(
        "Type",
        when(col("AppName").isin("CHANNEL", "DSHD", "KPLUS", "KPlus"), "Truyen Hinh")
        .when(col("AppName").isin("VOD", "FIMS_RES", "BHD_RES", "VOD_RES", "FIMS", "BHD", "DANET"), "Phim Truyen")
        .when(col("AppName") == "RELAX", "Giai Tri")
        .when(col("AppName") == "CHILD", "Thieu Nhi")
        .when(col("AppName") == "SPORT", "The Thao")
        .otherwise("Error")
    )
    df = df.filter(col("Contract") != "0").filter(col("Type") != "Error")
    return df.select("Contract", "Type", "TotalDuration", "Mac")


def add_date_from_filename(df):
    # Mỗi record lấy Date theo tên file YYYYMMDD.json
    return (
        df.withColumn("file_path", input_file_name())
          .withColumn("Date", to_date(regexp_extract("file_path", r"(\d{8})\.json", 1), "yyyyMMdd"))
          .drop("file_path")
    )


def pivot_duration(df_clean):
    # Pivot duration theo Type -> ra 5 cột duration
    return (
        df_clean.groupBy("Contract")
                .pivot("Type", ["Truyen Hinh", "Phim Truyen", "Giai Tri", "Thieu Nhi", "The Thao"])
                .sum("TotalDuration")
                .na.fill(0)
    )


def add_mostwatch_and_taste(pivot_df):
    pivot_df = pivot_df.withColumn(
        "MostWatchValue",
        greatest(
            col("Truyen Hinh"), col("Phim Truyen"), col("Giai Tri"), col("Thieu Nhi"), col("The Thao")
        )
    )

    pivot_df = pivot_df.withColumn(
        "MostWatch",
        when(col("MostWatchValue") == col("Truyen Hinh"), lit("Truyen Hinh"))
        .when(col("MostWatchValue") == col("Phim Truyen"), lit("Phim Truyen"))
        .when(col("MostWatchValue") == col("The Thao"), lit("The Thao"))
        .when(col("MostWatchValue") == col("Thieu Nhi"), lit("Thieu Nhi"))
        .when(col("MostWatchValue") == col("Giai Tri"), lit("Giai Tri"))
        .otherwise(lit("Truyen Hinh"))
    ).drop("MostWatchValue")

    pivot_df = pivot_df.withColumn(
        "Taste",
        concat_ws(
            "-",
            when(col("Giai Tri") > 0, lit("Giai Tri")),
            when(col("Phim Truyen") > 0, lit("Phim Truyen")),
            when(col("The Thao") > 0, lit("The Thao")),
            when(col("Thieu Nhi") > 0, lit("Thieu Nhi")),
            when(col("Truyen Hinh") > 0, lit("Truyen Hinh")),
        )
    )
    return pivot_df


def agg_active_and_devices(df_clean):
    # 1 bảng aggregate duy nhất: ActiveDays + TotalDevices
    return (
        df_clean.groupBy("Contract")
                .agg(
                    F.countDistinct("Date").alias("ActiveDays"),
                    F.countDistinct("Mac").alias("TotalDevices")
                )
    )


def add_dynamic(df_result):
    return df_result.withColumn(
        "Dynamic",
        when(col("ActiveDays") < 5, lit("Low"))
        .when((col("ActiveDays") >= 5) & (col("ActiveDays") <= 10), lit("Medium"))
        .otherwise(lit("High"))
    )


# Read files
def read_selected_files(path: str, filenames: list[str]):
    df_union = None
    for fn in filenames:
        file_path = os.path.join(path, fn)
        if not os.path.exists(file_path):
            print("Khong ton tai file:", file_path)
            continue

        print("-----------------------------")
        print("Read data file:", fn)
        print("-----------------------------")

        df_d = spark.read.json(file_path)
        df_union = df_d if df_union is None else df_union.unionByName(df_d)

    return df_union


def read_by_date_range(path: str, start_yyyymmdd: str, end_yyyymmdd: str):
    date_list = range_date(start_yyyymmdd, end_yyyymmdd)
    filenames = [f"{d}.json" for d in date_list]
    return read_selected_files(path, filenames)

# Finalize 
def finalize_with_logs(df_raw):
    print("-----------------------------")
    print("Union du lieu xong - show top 10")
    print("-----------------------------")
    df_raw.show(10, truncate=False)

    print("-----------------------------")
    print("Select _source.*")
    print("-----------------------------")
    df_source = select_fields(df_raw)
    df_source.show(10, truncate=False)

    print("-----------------------------")
    print("Transform category (AppName -> Type) + filter")
    print("-----------------------------")
    df_cat = transform_category(df_source)
    df_cat.show(10, truncate=False)

    print("-----------------------------")
    print("Add Date from filename")
    print("-----------------------------")
    df_clean = add_date_from_filename(df_cat)
    df_clean.show(10, truncate=False)

    print("-----------------------------")
    print("Pivot duration theo Type")
    print("-----------------------------")
    pivot_df = pivot_duration(df_clean)
    pivot_df.show(10, truncate=False)

    print("-----------------------------")
    print("Tinh MostWatch + Taste")
    print("-----------------------------")
    pivot_df2 = add_mostwatch_and_taste(pivot_df)
    pivot_df2.show(10, truncate=False)

    print("-----------------------------")
    print("Tinh ActiveDays + TotalDevices (agg)")
    print("-----------------------------")
    agg_df = agg_active_and_devices(df_clean)
    agg_df.show(10, truncate=False)

    print("-----------------------------")
    print("Join pivot + agg + tinh Dynamic")
    print("-----------------------------")
    result = (
        pivot_df2.join(agg_df, "Contract", "left")
                 .na.fill({"ActiveDays": 0, "TotalDevices": 1})
    )
    result = add_dynamic(result)

    final_cols = [
        "Contract", "TotalDevices", "ActiveDays", "Dynamic",
        "MostWatch", "Taste",
        "Truyen Hinh", "Phim Truyen", "Giai Tri", "Thieu Nhi", "The Thao"
    ]
    result = result.select(*final_cols)

    print("-----------------------------")
    print("Ket qua cuoi cung - show top 10")
    print("-----------------------------")
    result.show(10, truncate=False)

    return result


# Import to MySQL (DEFAULT DB = etl_data)
def import_to_mysql(result, db_name: str, table_name: str):
    url = (
        f"jdbc:mysql://localhost:3306/{db_name}"
        "?useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true"
    )
    driver = "com.mysql.cj.jdbc.Driver"
    user = "root"
    password = ""

    result.write.format("jdbc") \
        .option("url", url) \
        .option("driver", driver) \
        .option("dbtable", table_name) \
        .option("user", user) \
        .option("password", password) \
        .option("batchsize", 20000) \
        .mode("append") \
        .save()

    print(f"Data imported to MySQL -> DB: {db_name}, Table: {table_name}")



# Main task
def maintask(path: str):
    task_start = datetime.now()

    print("=============================")
    print("ETL APP (Spark)")
    print("=============================")

    print("-----------------------------")
    print("Chon cach ETL")
    print("1) ETL theo khoang ngay (YYYYMMDD -> YYYYMMDD)")
    print("2) ETL theo danh sach file (nhap ten file, cach nhau boi dau phay)")
    print("-----------------------------")
    choice = input("Ban chon (1/2): ").strip()

    df_raw = None

    if choice == "1":
        start_date = input("Nhap start_date (yyyymmdd): ").strip()
        end_date = input("Nhap end_date (yyyymmdd): ").strip()
        df_raw = read_by_date_range(path, start_date, end_date)
    elif choice == "2":
        raw = input("Nhap danh sach file (vd: 20220401.json,20220403.json): ").strip()
        filenames = [x.strip() for x in raw.split(",") if x.strip()]
        df_raw = read_selected_files(path, filenames)
    else:
        print("Ban chi duoc chon 1 hoac 2.")
        return None

    if df_raw is None:
        print("Khong doc duoc file nao.")
        return None

    if df_raw.limit(1).count() == 0:
        print("Du lieu rong.")
        return None

    print("-----------------------------")
    print("Bat dau ETL")
    print("-----------------------------")
    result = finalize_with_logs(df_raw)

    export_mysql = input("Ban co muon xuat du lieu vao MySQL khong? (y/n): ").strip().lower()
    if export_mysql == "y":
        db_name = input("Nhap ten database (ENTER = etl_data): ").strip()
        if db_name == "":
            db_name = "etl_data"

        table_name = input("Nhap ten bang (ENTER = customer_content_stats): ").strip()
        if table_name == "":
            table_name = "customer_content_stats"

        print('-----------------------------')
        print('Import result to mysql')
        print('-----------------------------')
        import_to_mysql(result, db_name, table_name)

    else:
        print("Khong export vao MySQL.")

    print("Finished job")

    task_end = datetime.now()
    duration = task_end - task_start
    print("-----------------------------")
    print("Task start time:", task_start.strftime("%Y-%m-%d %H:%M:%S"))
    print("Task end time  :", task_end.strftime("%Y-%m-%d %H:%M:%S"))
    print("Total runtime  : %.2f seconds" % duration.total_seconds())
    print("-----------------------------")

    return result


if __name__ == "__main__":
    path = r"C:\Users\admin\OneDrive\Study_de\Dataset\log_content"
    maintask(path)
