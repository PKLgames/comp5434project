import os
import sys
import time
from pathlib import Path

import pyspark

# Share the portable Spark environment setup without launching a Spark session.
from clean_data import PROJECT_DIR, DATA_DIR

from pyspark.sql import SparkSession
from pyspark.sql import functions as F



def read_csv(spark, filename):
    return (
        spark.read
        .option("header", True)
        .option("encoding", "UTF-8")
        # 评论可能包含换行，以及用双引号转义的引号。
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        # 遇到无法正常解析的记录时停止，避免静默丢弃。
        .option("mode", "FAILFAST")
        # 暂时全部按字符串读取，之后再验证和转换类型。
        .csv(str(DATA_DIR / filename))
    )


def summarize(df, name):
    print(f"\n===== {name}：记录数与缺失值 =====", flush=True)

    expressions = [F.count("*").alias("total_rows")]

    for column in df.columns:
        missing = (
            F.col(column).isNull()
            | (F.length(F.trim(F.col(column))) == 0)
        )
        expressions.append(
            F.sum(F.when(missing, 1).otherwise(0))
            .alias(f"{column}_missing")
        )

    result = df.agg(*expressions).first().asDict()

    for key, value in result.items():
        print(f"{key}: {value}", flush=True)


def main():
    spark = (
        SparkSession.builder
        .appName("Task2DataAnalysis")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    started = time.perf_counter()

    try:
        print("开始分析。训练集较大，请等待……", flush=True)

        train = read_csv(spark, "train.csv")
        test = read_csv(spark, "test.csv")

        summarize(train, "训练集")
        summarize(test, "测试集")
        print("\n===== 训练集字段有效性与文本长度 =====", flush=True)

        checked = (
            train
            .withColumn(
                "_rating", F.expr("try_cast(rating as double)")
            )
            .withColumn(
                "_votes", F.expr("try_cast(votes as bigint)")
            )
            .withColumn(
                "_time", F.expr("try_cast(time as bigint)")
            )
        )

        def count_invalid(condition, name):
            return F.sum(
                F.when(condition, 1).otherwise(0)
            ).alias(name)

        quality = checked.agg(
            count_invalid(
                F.col("_rating").isNull()
                | ~F.col("_rating").isin(1, 2, 3, 4, 5),
                "invalid_rating",
            ),
            count_invalid(
                F.col("_votes").isNull() | (F.col("_votes") < 0),
                "invalid_votes",
            ),
            count_invalid(
                F.col("_time").isNull() | (F.col("_time") <= 0),
                "invalid_time",
            ),
            count_invalid(
                F.col("purchased").isNull()
                | ~F.lower(F.trim(F.col("purchased"))).isin(
                    "true", "false"
                ),
                "invalid_purchased",
            ),
            F.avg(
                F.length(F.coalesce(F.col("title"), F.lit("")))
            ).alias("avg_title_chars"),
            F.max(F.length("title")).alias("max_title_chars"),
            F.avg(
                F.length(F.coalesce(F.col("comment"), F.lit("")))
            ).alias("avg_comment_chars"),
            F.max(F.length("comment")).alias("max_comment_chars"),
        ).first().asDict()

        for key, value in quality.items():
            print(f"{key}: {value}", flush=True)
        print("\n===== 异常 votes 记录 =====", flush=True)

        checked.filter(
            F.col("_votes").isNull() | (F.col("_votes") < 0)
        ).select(
            "id", "votes", "rating", "title"
        ).show(20, truncate=100)

        print("\n===== 各数据集内部的 ID 重复检查 =====", flush=True)

        for name, df in [("训练集", train), ("测试集", test)]:
            id_stats = df.agg(
                F.count("*").alias("total_rows"),
                F.countDistinct("id").alias("unique_ids"),
            ).first()

            extra_rows = (
                id_stats["total_rows"] - id_stats["unique_ids"]
            )

            print(
                f"{name}: 总记录数={id_stats['total_rows']}, "
                f"唯一ID数={id_stats['unique_ids']}, "
                f"重复ID多出的记录数={extra_rows}",
                flush=True,
            )
        print("\n===== 测试集字段有效性 =====", flush=True)

        test_checked = (
            test
            .withColumn("_votes", F.expr("try_cast(votes as bigint)"))
            .withColumn("_time", F.expr("try_cast(time as bigint)"))
        )

        bad_votes = (
            F.col("_votes").isNull() | (F.col("_votes") < 0)
        )
        bad_time = (
            F.col("_time").isNull() | (F.col("_time") <= 0)
        )
        bad_purchased = (
            F.col("purchased").isNull()
            | ~F.lower(F.trim(F.col("purchased"))).isin(
                "true", "false"
            )
        )

        test_checked.agg(
            count_invalid(bad_votes, "invalid_votes"),
            count_invalid(bad_time, "invalid_time"),
            count_invalid(bad_purchased, "invalid_purchased"),
        ).show(truncate=False)

        test_checked.filter(
            bad_votes | bad_time | bad_purchased
        ).select(
            "id", "votes", "time", "purchased"
        ).show(20, truncate=False)
        print("\n===== 训练集评分分布 =====", flush=True)
        train.groupBy("rating").count().orderBy("rating").show(
            30, truncate=False
        )

        elapsed = time.perf_counter() - started
        print(f"\n本次分析耗时：{elapsed:.2f} 秒", flush=True)
        print("ANALYSIS_OK", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
