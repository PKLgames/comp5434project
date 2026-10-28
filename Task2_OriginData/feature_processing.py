import json
import time

# 复用已有环境配置，不会执行清洗。
from clean_data import PROJECT_DIR

from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_DIR = PROJECT_DIR / "output"

# 明确列出模型可使用的基础数值特征。
# id、rating、用户和产品 ID 不包含在这里。
NUMERIC_FEATURES = [
    "title_char_count",
    "comment_char_count",
    "title_word_count",
    "comment_word_count",
    "votes",
    "votes_log",
    "purchased",
    "review_year",
    "review_month",
    "comment_missing",
    "votes_invalid",
]

# 提取字母、数字组成的词，保留词内的撇号，例如 don't。
TOKEN_PATTERN = r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*"


def add_features(df):
    """固定规则特征处理，无需从数据中拟合参数。"""

    # 原始 title/comment 保留。
    # 仅对分词用的临时文本去除 HTML 标签、整理空白。
    for field in ["title", "comment"]:
        text = F.coalesce(F.col(field), F.lit(""))
        text = F.regexp_replace(text, r"<[^>]*>", " ")
        text = F.trim(F.regexp_replace(text, r"\s+", " "))

        df = df.withColumn(f"_{field}_text", text)

    title_tokenizer = RegexTokenizer(
        inputCol="_title_text",
        outputCol="title_tokens",
        pattern=TOKEN_PATTERN,
        gaps=False,
        minTokenLength=1,
        toLowercase=True,
    )

    comment_tokenizer = RegexTokenizer(
        inputCol="_comment_text",
        outputCol="comment_tokens",
        pattern=TOKEN_PATTERN,
        gaps=False,
        minTokenLength=1,
        toLowercase=True,
    )

    df = title_tokenizer.transform(df)
    df = comment_tokenizer.transform(df)

    df = (
        df
        # 字符数按保留的原始文本计算。
        .withColumn("title_char_count", F.length("title"))
        .withColumn("comment_char_count", F.length("comment"))
        # 词数按上述固定分词规则计算。
        .withColumn("title_word_count", F.size("title_tokens"))
        .withColumn("comment_word_count", F.size("comment_tokens"))
        .withColumn(
            "votes_log",
            F.log1p(F.col("votes").cast("double")),
        )
        .withColumn(
            "_review_timestamp",
            F.timestamp_millis(F.col("time")),
        )
        .withColumn(
            "review_year", F.year("_review_timestamp")
        )
        .withColumn(
            "review_month", F.month("_review_timestamp")
        )
        .drop("_title_text", "_comment_text", "_review_timestamp")
    )

    return df


def process_dataset(
    spark,
    input_name,
    output_name,
    expected_rows,
):
    print(f"\n===== 开始处理 {input_name} =====", flush=True)
    started = time.perf_counter()

    source = spark.read.parquet(str(OUTPUT_DIR / input_name))
    featured = add_features(source)

    output_path = OUTPUT_DIR / output_name

    print("正在生成并保存特征……", flush=True)

    # 只覆盖本步骤生成的特征目录。
    # 不修改清洗结果或原来的训练/验证数据。
    featured.write.mode("overwrite").parquet(str(output_path))

    processing_seconds = time.perf_counter() - started

    print("正在回读并检查……", flush=True)
    validation_started = time.perf_counter()

    saved = spark.read.parquet(str(output_path))

    # 检查数值特征是否为空或 NaN。
    invalid = F.lit(False)

    for column in NUMERIC_FEATURES:
        invalid = (
            invalid
            | F.col(column).isNull()
            | F.isnan(F.col(column).cast("double"))
        )

    invalid = (
        invalid
        | (F.col("review_month") < 1)
        | (F.col("review_month") > 12)
        | (F.col("votes") < 0)
        | F.col("title_tokens").isNull()
        | F.col("comment_tokens").isNull()
    )

    stats = saved.agg(
        F.count("*").alias("rows"),
        F.sum(
            F.when(invalid, 1).otherwise(0)
        ).alias("invalid_feature_rows"),
        F.min("review_year").alias("min_year"),
        F.max("review_year").alias("max_year"),
        F.avg("title_word_count").alias("avg_title_words"),
        F.avg("comment_word_count").alias("avg_comment_words"),
    ).first().asDict()

    if stats["rows"] != expected_rows:
        raise ValueError(
            f"记录数不一致：预期 {expected_rows}，"
            f"实际 {stats['rows']}"
        )

    if stats["invalid_feature_rows"] != 0:
        raise ValueError(
            f"发现异常特征记录：{stats['invalid_feature_rows']}"
        )

    print(f"记录数：{stats['rows']:,}", flush=True)
    print(
        f"异常特征记录数：{stats['invalid_feature_rows']}",
        flush=True,
    )
    print(
        f"年份范围：{stats['min_year']}—{stats['max_year']}",
        flush=True,
    )
    print(
        f"标题平均词数：{stats['avg_title_words']:.2f}",
        flush=True,
    )
    print(
        f"正文平均词数：{stats['avg_comment_words']:.2f}",
        flush=True,
    )

    print("\n分词样例：", flush=True)
    saved.select(
        "id",
        "title",
        "title_tokens",
        "title_word_count",
        "comment_word_count",
    ).show(3, truncate=100)

    validation_seconds = time.perf_counter() - validation_started

    print(
        f"特征生成与保存耗时：{processing_seconds:.2f} 秒",
        flush=True,
    )
    print(
        f"回读验证与展示耗时：{validation_seconds:.2f} 秒",
        flush=True,
    )
    print(f"输出目录：{output_path}", flush=True)

    return {
        "source": input_name,
        "output": output_name,
        **stats,
        "processing_and_write_seconds": round(
            processing_seconds, 2
        ),
        "validation_and_display_seconds": round(
            validation_seconds, 2
        ),
    }


def main():
    spark = (
        SparkSession.builder
        .appName("Task2BasicFeatureProcessing")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        # 年、月统一按 UTC 提取。
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        train_stats = process_dataset(
            spark,
            input_name="train_80",
            output_name="train_features_base",
            expected_rows=6_104_774,
        )

        validation_stats = process_dataset(
            spark,
            input_name="validation_20",
            output_name="validation_features_base",
            expected_rows=1_526_196,
        )

        manifest = {
            "numeric_features": NUMERIC_FEATURES,
            "text_token_columns": [
                "title_tokens",
                "comment_tokens",
            ],
            "identifier_column": "id",
            "target_column": "rating",
            "timezone": "UTC",
            "token_pattern": TOKEN_PATTERN,
            "token_lowercase": True,
            "remove_html_tags_for_tokenization": True,
            "remove_stopwords": False,
            "learned_parameters": [],
            "spark_version": spark.version,
            "datasets": [train_stats, validation_stats],
        }

        manifest_path = OUTPUT_DIR / "feature_processing_summary.json"

        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

        print(f"\n特征说明与统计：{manifest_path}", flush=True)
        print("BASE_FEATURES_OK", flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()