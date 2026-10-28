import json
import os
import sys
import time
from pathlib import Path

import pyspark

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
HADOOP_HOME = Path(r"D:\HADOOP")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_HOME"] = str(Path(pyspark.__file__).resolve().parent)
os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
os.environ["PATH"] = (
    str(HADOOP_HOME / "bin") + os.pathsep + os.environ.get("PATH", "")
)

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def read_csv(spark, filename):
    """读取原始 CSV，保留字符串类型，支持正文中的换行。"""
    return (
        spark.read
        .option("header", True)
        .option("encoding", "UTF-8")
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "FAILFAST")
        .csv(str(PROJECT_DIR / filename))
    )


def clean_records(df):
    """保留第一次清洗的全部规则。"""

    missing_comment = (
        F.col("comment").isNull()
        | (F.length(F.trim(F.col("comment"))) == 0)
    )

    df = (
        df
        .withColumn("comment_missing", missing_comment.cast("int"))
        .withColumn(
            "comment",
            F.when(missing_comment, F.lit(""))
            .otherwise(F.col("comment")),
        )
        .fillna({"title": ""})
    )

    votes_number = F.expr("try_cast(votes as bigint)")
    votes_invalid = votes_number.isNull() | (votes_number < 0)

    df = (
        df
        .withColumn("votes_invalid", votes_invalid.cast("int"))
        .withColumn(
            "votes",
            F.when(votes_invalid, F.lit(0)).otherwise(votes_number),
        )
        .withColumn("time", F.expr("try_cast(time as bigint)"))
        .withColumn(
            "purchased",
            F.when(
                F.lower(F.trim(F.col("purchased"))) == "true", 1
            ).when(
                F.lower(F.trim(F.col("purchased"))) == "false", 0
            ).otherwise(F.lit(None).cast("int")),
        )
    )

    if "rating" in df.columns:
        df = df.withColumn(
            "rating", F.expr("try_cast(rating as int)")
        )

    return df


def main():
    for path in [
        PROJECT_DIR / "train.csv",
        HADOOP_HOME / "bin" / "winutils.exe",
        HADOOP_HOME / "bin" / "hadoop.dll",
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"找不到文件：{path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = OUTPUT_DIR / "clean_train_latest"
    removed_path = OUTPUT_DIR / "train_removed_latest"
    summary_path = OUTPUT_DIR / "clean_train_latest_summary.json"

    spark = (
        SparkSession.builder
        .appName("Task2LatestTrainingCleaning")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    duplicate_info = None
    removed = None
    started = time.perf_counter()

    try:
        print("\n===== 1. 检查原始训练集 =====", flush=True)
        raw = read_csv(spark, "train.csv")

        rating_number = F.expr("try_cast(rating as int)")
        source_stats = raw.agg(
            F.count("*").alias("rows"),
            F.sum(
                F.when(
                    rating_number.isNull()
                    | ~rating_number.isin(1, 2, 3, 4, 5),
                    1,
                ).otherwise(0)
            ).alias("invalid_ratings"),
        ).first()

        original_count = source_stats["rows"]

        if original_count != 7_684_645:
            raise ValueError(
                f"记录数与之前的数据不同：{original_count}"
            )

        if source_stats["invalid_ratings"] != 0:
            raise ValueError("原始评分存在非法值，需要先检查")

        print(f"原始记录数：{original_count:,}", flush=True)

        # 用原始字段判断重复，避免清洗改变原值后产生误判。
        input_columns = [
            "title",
            "comment",
            "user_id",
            "prod_id",
            "parent_prod_id",
            "time",
            "votes",
            "purchased",
        ]

        keyed = raw.withColumn(
            "_input_key",
            F.sha2(
                F.to_json(
                    F.struct(*[F.col(c) for c in input_columns]),
                    options={"ignoreNullFields": "false"},
                ),
                256,
            ),
        )

        print("\n===== 2. 识别重复和评分冲突 =====", flush=True)

        duplicate_info = (
            keyed.groupBy("_input_key")
            .agg(
                F.count("*").alias("_records"),
                F.min(
                    F.col("rating").cast("int")
                ).alias("_min_rating"),
                F.max(
                    F.col("rating").cast("int")
                ).alias("_max_rating"),
                # 评分一致时，固定保留字符串排序最小的 ID。
                F.min("id").alias("_keep_id"),
            )
            .filter(F.col("_records") > 1)
            .withColumn(
                "_conflict",
                F.col("_min_rating") != F.col("_max_rating"),
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        counts = duplicate_info.agg(
            F.coalesce(
                F.sum(
                    F.when(
                        F.col("_conflict"), F.col("_records")
                    ).otherwise(0)
                ),
                F.lit(0),
            ).alias("conflict_removed"),
            F.coalesce(
                F.sum(
                    F.when(
                        ~F.col("_conflict"),
                        F.col("_records") - 1,
                    ).otherwise(0)
                ),
                F.lit(0),
            ).alias("duplicate_removed"),
        ).first()

        conflict_removed = counts["conflict_removed"]
        duplicate_removed = counts["duplicate_removed"]
        total_removed = conflict_removed + duplicate_removed
        expected_remaining = original_count - total_removed

        print(
            f"评分冲突：删除 {conflict_removed:,} 条",
            flush=True,
        )
        print(
            f"完全重复且评分相同：删除多余的 "
            f"{duplicate_removed:,} 条",
            flush=True,
        )
        print(f"合计删除：{total_removed:,} 条", flush=True)
        print(f"预计保留：{expected_remaining:,} 条", flush=True)

        # 冲突记录全部删除；评分一致的重复记录只留一条。
        removed = (
            keyed.select("id", "rating", "_input_key")
            .join(duplicate_info, on="_input_key", how="inner")
            .filter(
                F.col("_conflict")
                | (F.col("id") != F.col("_keep_id"))
            )
            .select(
                "id",
                F.col("rating").alias("original_rating"),
                F.when(
                    F.col("_conflict"),
                    F.lit("rating_conflict"),
                ).otherwise(
                    F.lit("identical_duplicate")
                ).alias("reason"),
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        if removed.count() != total_removed:
            raise ValueError("删除清单与预计删除数量不一致")

        print("\n===== 3. 清洗并保存保留的数据 =====", flush=True)

        remaining_raw = (
            keyed.join(
                removed.select("id"),
                on="id",
                how="left_anti",
            )
            .select(*raw.columns)
        )

        # 从原始数据重建完整流程，执行与第一次相同的清洗规则。
        cleaned = clean_records(remaining_raw)

        # 仅覆盖最新结果目录，不覆盖旧清洗结果和原始文件。
        cleaned.repartition(16).write.mode("overwrite").parquet(
            str(result_path)
        )

        removed.repartition(2).write.mode("overwrite").parquet(
            str(removed_path)
        )

        print("\n===== 4. 回读并验证 =====", flush=True)

        saved = spark.read.parquet(str(result_path))

        invalid = (
            F.col("id").isNull()
            | (F.length(F.trim(F.col("id"))) == 0)
            | F.col("title").isNull()
            | F.col("comment").isNull()
            | F.col("votes").isNull()
            | (F.col("votes") < 0)
            | F.col("time").isNull()
            | (F.col("time") <= 0)
            | F.col("purchased").isNull()
            | ~F.col("purchased").isin(0, 1)
            | F.col("rating").isNull()
            | ~F.col("rating").isin(1, 2, 3, 4, 5)
        )

        final_stats = saved.agg(
            F.count("*").alias("rows"),
            F.sum(
                F.when(invalid, 1).otherwise(0)
            ).alias("bad_rows"),
            F.sum("comment_missing").alias("missing_comments"),
            F.sum("votes_invalid").alias("invalid_votes"),
        ).first().asDict()

        if final_stats["rows"] != expected_remaining:
            raise ValueError("最终记录数与预计数量不一致")

        if final_stats["bad_rows"] != 0:
            raise ValueError("最终数据仍存在非法字段")

        if (
            saved.join(
                removed.select("id"),
                on="id",
                how="left_semi",
            ).limit(1).count()
            != 0
        ):
            raise ValueError("最终数据中仍有应删除的 ID")

        elapsed = time.perf_counter() - started

        summary = {
            "original_rows": original_count,
            "conflicting_records_removed": conflict_removed,
            "identical_duplicates_removed": duplicate_removed,
            "total_removed": total_removed,
            "remaining_rows": final_stats["rows"],
            "remaining_missing_comment_flags":
                final_stats["missing_comments"],
            "remaining_invalid_votes_flags":
                final_stats["invalid_votes"],
            "invalid_rows_after_cleaning": final_stats["bad_rows"],
            "total_seconds_including_validation": round(elapsed, 2),
            "spark_version": spark.version,
            "master": spark.sparkContext.master,
            "input_fields_for_duplicate_detection": input_columns,
            "output": str(result_path),
            "removed_records": str(removed_path),
        }

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"最终记录数：{final_stats['rows']:,}", flush=True)
        print(f"非法记录数：{final_stats['bad_rows']}", flush=True)
        print(
            f"保留数据中的正文缺失标记："
            f"{final_stats['missing_comments']:,}",
            flush=True,
        )
        print(
            f"保留数据中的点赞数异常标记："
            f"{final_stats['invalid_votes']:,}",
            flush=True,
        )
        print(f"最新训练数据：{result_path}", flush=True)
        print(f"删除清单：{removed_path}", flush=True)
        print(f"统计记录：{summary_path}", flush=True)
        print(f"总耗时（含回读验证）：{elapsed:.2f} 秒", flush=True)
        print("LATEST_CLEANING_OK", flush=True)

    finally:
        if removed is not None:
            removed.unpersist()
        if duplicate_info is not None:
            duplicate_info.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()