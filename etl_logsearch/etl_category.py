import os
import json
import time
import requests

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import (col, lit, row_number, lpad, concat, when, trim, concat_ws, regexp_replace)


# CONFIG PATH
CONFIG_DIR = r"C:\Users\admin\OneDrive\Study_de\Bigdata\360_project_DE\etl_logsearch\most_search_staging_config"
CONFIG_PATH = os.path.join(CONFIG_DIR, "mysql_staging_config.json")

# Lưu tiến độ classify (mapping keyword_norm -> category)
MAPPING_STORE_PATH = os.path.join(CONFIG_DIR, "keyword_category_mapping.json")

# MySQL connection 
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DRIVER = "com.mysql.cj.jdbc.Driver"
#== CẤU HÌNH database ==
def mysql_url(db_name: str):
    return f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{db_name}"

spark = (SparkSession.builder.appName("etl_category").config("spark.driver.memory", "8g").getOrCreate())


# Load config + read MySQL
# Config chỉ cần:
# {"db_name":"etl_data","table_name":"mostsearch_full"}
def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            "Khong tim thay mysql_staging_config.json.\n"
            "Hay chay etl_mostsearch.py va ghi MOST_SEARCH_FULL vao MySQL truoc.\n"
            f"Missing: {CONFIG_PATH}"
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "db_name" not in cfg or "table_name" not in cfg:
        raise ValueError(
            "Config thieu key. Can co dang:\n"
            '{"db_name":"etl_data","table_name":"mostsearch_full"}'
        )
    return cfg

def read_mostsearch_full(cfg):
    return (
        spark.read.format("jdbc")
        .option("url", mysql_url(cfg["db_name"]))   
        .option("dbtable", cfg["table_name"])
        .option("user", MYSQL_USER)
        .option("password", MYSQL_PASSWORD)
        .option("driver", MYSQL_DRIVER)
        .load()
    )


# ====Mapping store (json)====
def load_mapping_store(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}

def save_mapping_store(path: str, mapping: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def mapped_keys_to_spark_df(keys):
    rows = [(k,) for k in keys]
    schema = StructType([StructField("Most_Search_Norm", StringType(), True)])
    return spark.createDataFrame(rows, schema)

def mapping_to_spark_df(mapping_dict):
    rows = [(k, v) for k, v in mapping_dict.items()]
    schema = StructType([
        StructField("Most_Search_Norm", StringType(), True),
        StructField("category", StringType(), True),
    ])
    return spark.createDataFrame(rows, schema)


# Build Bang category: content_id, content, category
def build_category_table_from_store(mapping_store: dict):
    df = spark.createDataFrame([(k, v) for k, v in mapping_store.items()], ["content", "category"])
    df = df.filter(col("content").isNotNull()).filter(col("content") != "").dropDuplicates(["content"])
    w = Window.orderBy(col("content"))
    df = df.withColumn("rn", row_number().over(w))
    df = df.withColumn("content_id", concat(lit("ct"), lpad(col("rn").cast("string"), 6, "0")))
    return df.select("content_id", "content", "category")


# ====LM Studio client + classify====
def make_lmstudio_client():
    return {"base_url": "http://localhost:1234/v1", "model": "google/gemma-3n-e4b"}

def classify_batch_lmstudio(client, norm_keywords):
    MODEL_NAME = client["model"]
    base_url = client["base_url"].rstrip("/")

    CATEGORIES = [
    "Action/Adventure", "Romance", "Comedy", "Horror/Thriller",
    "Drama", "Fantasy", "Sci-Fi",
    "Animation", "Documentary",
    "Sports", "News", "Music",
    "Reality Show", "TV Channel",
    "Other"
    ]

    cat_map = {c.lower(): c for c in CATEGORIES}

    if not norm_keywords:
        return {}

    # strip keyword để tránh key mismatch do space đầu/cuối
    norm_keywords_clean = [str(k).strip()[:120] for k in norm_keywords]
    keyword_block = "\n".join(norm_keywords_clean)

    # PROMPT
    user_prompt = (
    "Task: Classify each keyword into EXACTLY ONE category from the list below.\n"
    "Return ONLY a single JSON object mapping each input line to one category.\n"
    "No markdown. No explanation.\n\n"

    "Allowed categories (use ONLY these, no others):\n"
    + ", ".join(CATEGORIES)
    + "\n\n"

    "STRICT RULES:\n"
    "- Keys MUST match the input lines EXACTLY.\n"
    "- Values MUST be EXACTLY one of the allowed categories above (character-for-character).\n"
    "- NEVER invent new categories (Crime, Adventure, Biography, Historical, Anime, Kpop, etc. are NOT allowed).\n"
    "- Use 'Other' ONLY if no reasonable category can be inferred.\n"
    "- Mentally fix obvious typos or missing accents.\n\n"

    "MULTI-LANGUAGE & ALIAS UNDERSTANDING (VERY IMPORTANT):\n"
    "- A keyword may be a localized title, translated title, or alternative name of the SAME movie/series in different countries.\n"
    "- The keyword can be in English, Vietnamese, Chinese, Korean, or mixed language.\n"
    "- You MUST infer the underlying work if possible and classify by its CONTENT TYPE, not by language.\n"
    "- Examples:\n"
    "  • 'Stranger Things' (English), 'Cậu bé mất tích' (Vietnamese), '怪奇物语' (Chinese) → SAME TV series.\n"
    "  • 'Money Heist', 'Phi vụ triệu đô', 'La Casa de Papel' → SAME series.\n"
    "- Do NOT classify such localized names as 'Other' if the work is recognizable.\n\n"

    "CRITICAL DEFINITIONS (ABSOLUTE RULES):\n"
    "- Animation: ALL animated works, INCLUDING ALL Japanese anime.\n"
    "- If a keyword refers to anime/animation/cartoon/animated series/film, it MUST be classified as 'Animation' ONLY.\n"
    "- 'Animation' OVERRIDES ALL other categories (Romance, Fantasy, Action/Adventure, Drama, Sci-Fi, etc.).\n"
    "- Fantasy: LIVE-ACTION (human-acted) fantasy ONLY (magic, dragons, vampires, mythical worlds), and ONLY when NOT animated.\n"
    "- Horror/Thriller: includes Horror, Thriller, AND ALL Crime-related titles (crime, mafia, murder, investigation, suspense).\n"
    "- Documentary: includes Documentary, Biography, and Historical, but ALWAYS output 'Documentary' only.\n"
    "- Action/Adventure: includes action OR adventure/journey/exploration titles.\n\n"

    "MUSIC DETECTION (VERY IMPORTANT, MULTI-LANGUAGE):\n"
    "- Classify as 'Music' if the keyword indicates a SONG, MUSIC TRACK, or MUSIC VIDEO.\n"
    "- Vietnamese music signals (choose Music): 'bài hát', 'bài ca', 'ca khúc', 'bản nhạc', 'nhạc', 'lời bài hát', 'nghe nhạc', 'nhạc trẻ', 'nhạc remix'.\n"
    "- English signals (choose Music): 'song', 'lyrics', 'music video', 'mv', 'official audio', 'official mv', 'karaoke', 'remix', 'cover', 'live concert', 'album', 'OST', 'soundtrack'.\n"
    "- Asian signals (choose Music): '歌曲', '歌詞/歌词', '音乐/音樂', 'mv', '카카오케/노래방(karoke)', '가사(lyrics)', 'ost'.\n"
    "- IMPORTANT: If the keyword starts with or contains 'bài ca ...' or 'bài hát ...' THEN it is Music.\n"
    "- EXCEPTIONS (NOT Music): 'bài tập', 'bài học', 'bài giảng', 'bài thi', 'bài kiểm tra', 'bài toán' → NOT Music.\n\n"

    "PRIORITY RULES (FOLLOW STRICTLY IN THIS ORDER):\n"
    "1) If animated/anime/cartoon → Animation (NO EXCEPTIONS).\n"
    "2) If any MUSIC signal above applies → Music.\n"
    "3) If sports events/leagues/teams/matches/scores/highlights/live → Sports.\n"
    "4) If news/current affairs (tin tức, thời sự, breaking news) → News.\n"
    "5) If TV station/channel name only (vtv, htv, sctv, hbo, live tv) → TV Channel.\n"
    "6) If crime/mafia/murder/investigation/suspense/thriller/horror → Horror/Thriller.\n"
    "7) If biography/historical/real-life factual/documentary → Documentary.\n"
    "8) Otherwise classify as live-action film/series genre: Action/Adventure, Drama, Comedy, Romance, Fantasy, Sci-Fi.\n\n"

    "OUTPUT FORMAT (STRICT):\n"
    "- Output ONLY one valid JSON object.\n"
    "- Each value MUST be EXACTLY one of the allowed categories (no extra spaces).\n\n"

    "INPUT (one keyword per line):\n"
    f"{keyword_block}\n"
    )


    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator. Output ONLY one valid JSON object. No extra text."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 3000,
        "stream": False,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {k: "Other" for k in norm_keywords_clean}

    try:
        parsed = json.loads(text[start:end + 1])
    except Exception:
        return {k: "Other" for k in norm_keywords_clean}

    #  chuẩn hóa key output (strip) để khớp với input clean
    parsed_norm = {}
    for kk, vv in parsed.items():
        parsed_norm[str(kk).strip()] = vv

    def normalize_cat(v):
        if v is None:
            return "Other"
        s = str(v).strip()
        #  gộp nhiều spaces thành 1 space
        s = " ".join(s.split())
        # unify case
        s_low = s.lower()
        # một vài sửa hay gặp
        s_low = s_low.replace("tvchannel", "tv channel")
        s_low = s_low.replace("realityshow", "reality show")
        return cat_map.get(s_low, "Other")

    out = {}
    for k in norm_keywords_clean:
        out[k] = normalize_cat(parsed_norm.get(k, "Other"))

    return out

#=== CHUẨN HÓA KEYWORD OTHER ====
def classify_all_keywords(client, all_norm_keywords, batch_size):
    SLEEP_SEC = 0.2
    mapping_all = {}

    def run_batch_safely(batch):
        if len(batch) == 1:
            try:
                return classify_batch_lmstudio(client, batch)
            except Exception:
                return {str(batch[0]): "Other"}
        try:
            return classify_batch_lmstudio(client, batch)
        except Exception as e:
            print("[WARN] Batch failed -> split smaller. Err:", str(e)[:200])
            mid = len(batch) // 2
            left = run_batch_safely(batch[:mid])
            right = run_batch_safely(batch[mid:])
            left.update(right)
            return left

    for i in range(0, len(all_norm_keywords), batch_size):
        batch = all_norm_keywords[i:i + batch_size]
        print("LM Studio batch", i // batch_size + 1, "size", len(batch))
        result_map = run_batch_safely(batch)
        mapping_all.update(result_map)
        time.sleep(SLEEP_SEC)

    return mapping_all

#==== Build FINAL OLAP OUTPUT====
def build_final(most_full_df, mapping_df):
    # join T6 category
    t6 = (
        most_full_df
        .select("user_id", "most_search_t6", col("most_search_norm_t6").alias("Most_Search_Norm"))
        .join(mapping_df, "Most_Search_Norm", "left")
        .drop("Most_Search_Norm")
        .withColumnRenamed("category", "category_t6")
    )

    # join T7 category
    t7 = (
        most_full_df
        .select("user_id", "most_search_t7", col("most_search_norm_t7").alias("Most_Search_Norm"))
        .join(mapping_df, "Most_Search_Norm", "left")
        .drop("Most_Search_Norm")
        .withColumnRenamed("category", "category_t7")
    )

    df = t6.join(t7, "user_id", "outer")

    df = df.withColumn(
        "trending_type",
        when(col("most_search_t6").isNotNull() & col("most_search_t7").isNotNull() &(trim(col("category_t6")) == trim(col("category_t7"))),lit("Unchanged"))
        .when(col("most_search_t6").isNotNull() & col("most_search_t7").isNotNull(),lit("Changed"))
        .when(col("most_search_t6").isNotNull() & col("most_search_t7").isNull(),lit("Only search T6"))
        .when(col("most_search_t6").isNull() & col("most_search_t7").isNotNull(),lit("Only search T7"))
        )

    df = df.withColumn(
        "previous",
        when(col("trending_type") == lit("Unchanged"), lit("Unchanged"))
        .when(col("trending_type") == lit("Changed"), concat_ws("-", col("category_t6"), col("category_t7")))
        .when(col("trending_type") == lit("Only search T6"), concat_ws("-", col("category_t6"), lit("NULL")))
        .when(col("trending_type") == lit("Only search T7"), concat_ws("-", lit("NULL"), col("category_t7")))
    )

    return df.select("user_id","most_search_t6", "category_t6","most_search_t7", "category_t7","trending_type", "previous")


#==== MySQL write (user chọn db/table)==== 
def write_df_to_mysql(df, db_name: str, table_name: str, mode: str):
    (
        df.write.format("jdbc")
        .option("url", mysql_url(db_name))              
        .option("driver", MYSQL_DRIVER)
        .option("dbtable", table_name)
        .option("user", MYSQL_USER)
        .option("password", MYSQL_PASSWORD)
        .mode(mode)
        .save()
    )

def ask_write_mysql(df, title: str, default_mode: str, cfg):
    print("")
    ans = input(f"{title} -> Bạn có muốn ghi vào database không? (y/n): ").strip().lower()
    if ans != "y":
        return

    db_input = input(f"Nhập tên database (Enter = dùng DB [{cfg['db_name']}]): ").strip()
    db_name = cfg["db_name"] if db_input == "" else db_input

    table_name = input("Nhập tên bảng: ").strip()

    mode = input(f"Nhập mode (append/overwrite) - mặc định {default_mode}: ").strip().lower()
    if mode not in ["append", "overwrite"]:
        mode = default_mode

    print(f"[STEP] Dang ghi MySQL: db={db_name}, table={table_name}, mode={mode}")
    write_df_to_mysql(df, db_name, table_name, mode)
    print("[DONE] Ghi MySQL thanh cong.")

#====MAIN====
def main():
    print("[START] etl_category.py")

    cfg = load_config()
    print(f"[CONFIG] Đọc MySQL: db={cfg['db_name']}, table={cfg['table_name']}")
    most_full = read_mostsearch_full(cfg)

    # DISTINCT keyword_norm 
    need_norm_df = (
        most_full.select(regexp_replace(trim(col("most_search_norm_t6")),r"\s+"," ").alias("Most_Search_Norm"))
        .union(most_full.select(regexp_replace(trim(col("most_search_norm_t7")),r"\s+"," ").alias("Most_Search_Norm")))
        .filter(col("Most_Search_Norm").isNotNull())
        .filter(col("Most_Search_Norm") != "")
        .distinct()
    )

    total_need = need_norm_df.count()
    print(f"[INFO] Tổng số keyword_norm cần gắn category: {total_need}")

    mapping_store = load_mapping_store(MAPPING_STORE_PATH)
    mapped_keys = set(mapping_store.keys())

    if mapped_keys:
        mapped_df = mapped_keys_to_spark_df(list(mapped_keys))
        remaining_df = need_norm_df.join(mapped_df, "Most_Search_Norm", "left_anti")
    else:
        remaining_df = need_norm_df

    total_remaining = remaining_df.count()
    total_done = total_need - total_remaining
    print(f"[PROGRESS] Đã gắn: {total_done}/{total_need} | Còn lại: {total_remaining}")

    # classify theo phiên/ngày 
    if total_remaining > 0:
        while True:
            raw_limit = input(f"Số lượng keyword cần phân loại? (1..{total_remaining}, hoac 0 = ALL): ").strip()
            try:
                LIMIT_KEYWORDS = int(raw_limit)
                if LIMIT_KEYWORDS == 0:
                    LIMIT_KEYWORDS = total_remaining
                    break
                if 1 <= LIMIT_KEYWORDS <= total_remaining:
                    break
                print(f"[WARN] Giá trị gắn cần nằm trong [1 đến {total_remaining}] hoac 0.")
            except ValueError:
                print("[WARN] Vui lòng nhập số nguyên.")

        while True:
            raw_bs = input("Nhap BATCH_SIZE: ").strip()
            try:
                BATCH_SIZE = int(raw_bs)
                if BATCH_SIZE > 0:
                    break
                print("[WARN] BATCH_SIZE phai > 0.")
            except ValueError:
                print("[WARN] BATCH_SIZE phai la so nguyen.")

        today_norm_df = remaining_df.orderBy(lit(1)).limit(LIMIT_KEYWORDS)
        today_norm = [r[0] for r in today_norm_df.collect()]

        print(f"[LMSTUDIO] Phân loại {len(today_norm)} keyword_norm")
        client = make_lmstudio_client()
        mapping_today = classify_all_keywords(client, today_norm, BATCH_SIZE)

        mapping_store.update(mapping_today)
        save_mapping_store(MAPPING_STORE_PATH, mapping_store)
        print("[SAVE] Da luu mapping vao:", MAPPING_STORE_PATH)

        # bảng category unique content
        bang2_full = build_category_table_from_store(mapping_store)
        bang2_full.show(30, truncate=False)
        ask_write_mysql(bang2_full, "[BANG 2] content_id-content-category (FULL UNIQUE)", default_mode="overwrite", cfg=cfg)

        # check remaining
        mapped_keys2 = set(mapping_store.keys())
        mapped_df2 = mapped_keys_to_spark_df(list(mapped_keys2))
        remaining_df2 = need_norm_df.join(mapped_df2, "Most_Search_Norm", "left_anti")
        total_remaining2 = remaining_df2.count()
        print(f"[PROGRESS] Con lai can gan: {total_remaining2}")

        if total_remaining2 > 0:
            print("[STOP] Chưa gắn xong hết keyword.")
            return

    # ==== build FINAL OLAP OUTPUT ==== 
    print("[STEP] Đã gắn xong hết keyword -> build FINAL")
    mapping_df_full = mapping_to_spark_df(mapping_store)
    final_df = build_final(most_full, mapping_df_full)
    final_df.show(50, truncate=False)

    ask_write_mysql(final_df, "[FINAL TABLE] trending_table", default_mode="overwrite", cfg=cfg)
    print("[DONE] etl_category.py")

if __name__ == "__main__":
    main()
