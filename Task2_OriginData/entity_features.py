import json
import time

# 复用现有环境和基础特征清单，不执行之前的 main()。
from clean_data import PROJECT_DIR
from feature_processing import NUMERIC_FEATURES

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_DIR = PROJECT_DIR / "output"
STATISTICS_DIR = OUTPUT_DIR / "entity_statistics"

TRAIN_ROWS = 6_104_774
VALIDATION_ROWS = 1_526_196

# 原始标识字段、特征名前缀。
ENTITY_SPECS = [
    ("user_id", "user"),
    ("prod_id", "product"),
    ("parent_prod_id", "parent_product"),
]

ENTITY_FEATURES = [
    f"{prefix}_{suffix}"
    for _, prefix in ENTITY_SPECS
    for suffix in [
        "review_count",
        "review_count_log",
        "unseen",
    ]
]


def build_statistics(spark, train_metadata):
    """仅从训练部分计算三份数量统计表。"""
    tables = {}
    summaries = []

    for key, prefix in ENTITY_SPECS:
        print(f"\n正在计算 {key} 数量统计……", flush=True)
        started = time.perf_counter()

        count_column = f"{prefix}_review_count"
        table_path = STATISTICS_DIR / prefix

        table = (
            train_metadata
            .groupBy(key)
            .agg(F.count("*").alias(count_column))
        )

        # 只覆盖本步骤的统计表。
        table.write.mode("overwrite").parquet(str(table_path))

        # 回读，后续训练和验证均使用保存好的同一份统计表。
        saved_table = spark.read.parquet(str(table_path))

        stats = saved_table.agg(
            F.count("*").alias("entities"),
            F.sum(count_column).alias("total_reviews"),
            F.max(count_column).alias("max_review_count"),
            F.sum(
                F.when(
                    F.col(key).isNull()
                    | (F.length(F.trim(F.col(key))) == 0)
                    | F.col(count_column).isNull()
                    | (F.col(count_column) <= 0),
                    1,
                ).otherwise(0)
            ).alias("bad_rows"),
        ).first()

        if stats["total_reviews"] != TRAIN_ROWS:
            raise ValueError(f"{key} 的统计数量合计不正确")

        if stats["bad_rows"] != 0:
            raise ValueError(f"{key} 的统计表存在异常")

        elapsed = time.perf_counter() - started

        print(f"不同标识数量：{stats['entities']:,}", flush=True)
        print(f"评论数量合计：{stats['total_reviews']:,}", flush=True)
        print(f"统计与保存验证耗时：{elapsed:.2f} 秒", flush=True)

        tables[prefix] = saved_table

        summaries.append({
            "key": key,
            "prefix": prefix,
            "entities": stats["entities"],
            "total_reviews": stats["total_reviews"],
            "max_review_count": stats["max_review_count"],
            "seconds_including_validation": round(elapsed, 2),
            "path": str(table_path),
        })

    return tables, summaries


def add_statistics(metadata, tables):
    """将训练部分的统计关联到指定数据，不重新计算统计。"""
    result = metadata

    for key, prefix in ENTITY_SPECS:
        count_column = f"{prefix}_review_count"
        unseen_column = f"{prefix}_unseen"
        log_column = f"{prefix}_review_count_log"

        result = result.join(
            tables[prefix],
            on=key,
            how="left",
        )

        # 没有匹配到训练统计表，表示该标识未在训练部分出现。
        result = (
            result
            .withColumn(
                unseen_column,
                F.col(count_column).isNull().cast("int"),
            )
            .withColumn(
                count_column,
                F.coalesce(
                    F.col(count_column),
                    F.lit(0).cast("long"),
                ),
            )
            .withColumn(
                log_column,
                F.log1p(F.col(count_column).cast("double")),
            )
        )

    return result.select("id", *ENTITY_FEATURES)


def process_dataset(
    spark,
    metadata,
    tables,
    feature_source_name,
    output_name,
    expected_rows,
    is_training,
):
    print(f"\n===== 处理 {output_name} =====", flush=True)
    started = time.perf_counter()

    # 先处理较小的 ID 和数值字段，
    # 避免在三次实体关联中反复搬运高维文本向量。
    mapping = add_statistics(metadata, tables).persist(
        StorageLevel.DISK_ONLY
    )

    try:
        invalid = F.lit(False)

        for column in ENTITY_FEATURES:
            invalid = (
                invalid
                | F.col(column).isNull()
                | F.isnan(F.col(column).cast("double"))
            )

        for _, prefix in ENTITY_SPECS:
            count_column = f"{prefix}_review_count"
            unseen_column = f"{prefix}_unseen"

            invalid = (
                invalid
                | (F.col(count_column) < 0)
                | ~F.col(unseen_column).isin(0, 1)
                | (
                    (F.col(unseen_column) == 1)
                    & (F.col(count_column) != 0)
                )
                | (
                    (F.col(unseen_column) == 0)
                    & (F.col(count_column) < 1)
                )
            )

        checks = [
            F.count("*").alias("rows"),
            F.countDistinct("id").alias("unique_ids"),
            F.sum(
                F.when(invalid, 1).otherwise(0)
            ).alias("invalid_rows"),
        ]

        for _, prefix in ENTITY_SPECS:
            checks.append(
                F.sum(f"{prefix}_unseen")
                .alias(f"{prefix}_unseen_rows")
            )

        stats = mapping.agg(*checks).first().asDict()

        if (
            stats["rows"] != expected_rows
            or stats["unique_ids"] != expected_rows
        ):
            raise ValueError("实体特征映射存在记录缺失或重复 ID")

        if stats["invalid_rows"] != 0:
            raise ValueError("实体特征存在非法值")

        if is_training:
            for _, prefix in ENTITY_SPECS:
                if stats[f"{prefix}_unseen_rows"] != 0:
                    raise ValueError("训练记录未匹配到自己的统计表")

        # 原来的 TF-IDF 文件没有用户/产品 ID，
        # 因此通过评论 id 加入新增特征。
        existing = spark.read.parquet(
            str(OUTPUT_DIR / feature_source_name)
        )

        result = (
            existing
            .join(mapping, on="id", how="left")
            .select(
                *existing.columns,
                *ENTITY_FEATURES,
            )
        )

        output_path = OUTPUT_DIR / output_name

        print("正在保存合并后的特征……", flush=True)

        result.write.mode("overwrite").parquet(str(output_path))

        # 回读，确认合并未丢记录，所有新增特征均已匹配。
        saved = spark.read.parquet(str(output_path))

        missing = F.lit(False)
        for column in ENTITY_FEATURES:
            missing = missing | F.col(column).isNull()

        saved_stats = saved.agg(
            F.count("*").alias("rows"),
            F.countDistinct("id").alias("unique_ids"),
            F.sum(
                F.when(missing, 1).otherwise(0)
            ).alias("missing_feature_rows"),
        ).first()

        if (
            saved_stats["rows"] != expected_rows
            or saved_stats["unique_ids"] != expected_rows
            or saved_stats["missing_feature_rows"] != 0
        ):
            raise ValueError("保存后的特征数量或关联结果不正确")

        print(f"保存后记录数：{saved_stats['rows']:,}", flush=True)
        print(
            f"未匹配新增特征的记录数："
            f"{saved_stats['missing_feature_rows']}",
            flush=True,
        )

        for _, prefix in ENTITY_SPECS:
            unseen_rows = stats[f"{prefix}_unseen_rows"]
            print(
                f"{prefix} 未在训练部分出现："
                f"{unseen_rows:,} 条，"
                f"占 {unseen_rows / expected_rows:.2%}",
                flush=True,
            )

        print("\n新增特征样例：", flush=True)
        saved.select(
            "id",
            "user_review_count",
            "product_review_count",
            "parent_product_review_count",
            "user_unseen",
            "product_unseen",
            "parent_product_unseen",
        ).show(5, truncate=False)

        elapsed = time.perf_counter() - started
        print(f"处理、保存及检查耗时：{elapsed:.2f} 秒", flush=True)
        print(f"输出目录：{output_path}", flush=True)

        return {
            "source_features": feature_source_name,
            "output": str(output_path),
            **stats,
            "seconds_including_validation": round(elapsed, 2),
        }

    finally:
        mapping.unpersist()


def main():
    spark = (
        SparkSession.builder
        .appName("Task2EntityStatisticsFeatures")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    started = time.perf_counter()

    try:
        STATISTICS_DIR.mkdir(parents=True, exist_ok=True)

        key_columns = [
            "id",
            "user_id",
            "prod_id",
            "parent_prod_id",
        ]

        # 明确只读取标识字段，不读取评分用于统计。
        train_metadata = spark.read.parquet(
            str(OUTPUT_DIR / "train_80")
        ).select(*key_columns)

        validation_metadata = spark.read.parquet(
            str(OUTPUT_DIR / "validation_20")
        ).select(*key_columns)

        tables, table_summaries = build_statistics(
            spark, train_metadata
        )

        train_summary = process_dataset(
            spark=spark,
            metadata=train_metadata,
            tables=tables,
            feature_source_name="train_features_tfidf",
            output_name="train_features_with_entity",
            expected_rows=TRAIN_ROWS,
            is_training=True,
        )

        validation_summary = process_dataset(
            spark=spark,
            metadata=validation_metadata,
            tables=tables,
            feature_source_name="validation_features_tfidf",
            output_name="validation_features_with_entity",
            expected_rows=VALIDATION_ROWS,
            is_training=False,
        )

        summary = {
            "statistics_fit_source": "train_80 only",
            "rating_used_for_statistics": False,
            "validation_used_for_statistics": False,
            "training_counts_include_current_record": True,
            "unseen_count_fill_value": 0,
            "new_features": ENTITY_FEATURES,
            "all_candidate_numeric_features":
                NUMERIC_FEATURES + ENTITY_FEATURES,
            "text_features": ["title_tfidf", "comment_tfidf"],
            "identifier": "id",
            "target": "rating",
            "numeric_features_scaled": False,
            "statistics_tables": table_summaries,
            "datasets": [train_summary, validation_summary],
            "spark_version": spark.version,
            "total_seconds_including_validation": round(
                time.perf_counter() - started, 2
            ),
        }

        summary_path = OUTPUT_DIR / "entity_features_summary.json"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"\n特征说明与统计：{summary_path}", flush=True)
        print("ENTITY_FEATURES_OK", flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()