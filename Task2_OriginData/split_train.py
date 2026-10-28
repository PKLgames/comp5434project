import json
import time

# 复用已有环境设置，不会执行清洗。
from clean_data import PROJECT_DIR

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


SEED = 42
OUTPUT_DIR = PROJECT_DIR / "output"

SOURCE_PATH = OUTPUT_DIR / "clean_train_latest"
TRAIN_PATH = OUTPUT_DIR / "train_80"
VALIDATION_PATH = OUTPUT_DIR / "validation_20"
ASSIGNMENTS_PATH = OUTPUT_DIR / "split_assignments"
SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"


def main():
    spark = (
        SparkSession.builder
        .appName("Task2TrainValidationSplit")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    started = time.perf_counter()

    try:
        print("\n===== 1. 读取最新训练数据 =====", flush=True)

        source = spark.read.parquet(str(SOURCE_PATH))

        source_stats = source.agg(
            F.count("*").alias("rows"),
            F.countDistinct("id").alias("unique_ids"),
        ).first()

        total = source_stats["rows"]

        if total != 7_630_970:
            raise ValueError(
                f"数据条数与最新清洗结果不同：{total}"
            )

        if source_stats["unique_ids"] != total:
            raise ValueError("存在缺失或重复 ID，需要先检查")

        print(f"待划分记录数：{total:,}", flush=True)

        print("\n===== 2. 按评分划分 80/20 =====", flush=True)

        # 只对 ID 和评分进行排序，避免搬运大段评论文本。
        ids = source.select("id", "rating")

        # 用 ID 和固定参数生成可重复的伪随机排序。
        # 相同输入和 Spark 版本下，不依赖文件中的行顺序。
        random_order = F.xxhash64(F.col("id"), F.lit(SEED))

        ordering = Window.partitionBy("rating").orderBy(
            random_order, F.col("id")
        )
        class_window = Window.partitionBy("rating")

        assignments = (
            ids
            .withColumn("_position", F.row_number().over(ordering))
            .withColumn(
                "_class_count",
                F.count("*").over(class_window),
            )
            # 每个评分取 floor(总数 × 4/5) 条作为训练数据。
            .withColumn(
                "_train_count",
                F.expr("(_class_count * 4) DIV 5"),
            )
            .withColumn(
                "subset",
                F.when(
                    F.col("_position") <= F.col("_train_count"),
                    F.lit("train"),
                ).otherwise(F.lit("validation")),
            )
            .select("id", "rating", "subset")
        )

        # 保存划分清单，A 和 B 使用同一份。
        assignments.repartition(8).write.mode("overwrite").parquet(
            str(ASSIGNMENTS_PATH)
        )

        saved_assignments = spark.read.parquet(
            str(ASSIGNMENTS_PATH)
        )

        assignment_stats = saved_assignments.agg(
            F.count("*").alias("rows"),
            F.countDistinct("id").alias("unique_ids"),
        ).first()

        if (
            assignment_stats["rows"] != total
            or assignment_stats["unique_ids"] != total
        ):
            raise ValueError("划分清单出现记录缺失或重复")

        print("\n各评分的划分条数：", flush=True)

        distribution = (
            saved_assignments
            .groupBy("rating", "subset")
            .count()
            .orderBy("rating", "subset")
            .collect()
        )

        for row in distribution:
            print(
                f"评分 {row['rating']}："
                f"{row['subset']} = {row['count']:,} 条",
                flush=True,
            )

        print("\n===== 3. 保存训练和验证数据 =====", flush=True)

        train_ids = saved_assignments.filter(
            F.col("subset") == "train"
        ).select("id")

        validation_ids = saved_assignments.filter(
            F.col("subset") == "validation"
        ).select("id")

        train = source.join(
            train_ids, on="id", how="left_semi"
        ).select(*source.columns)

        validation = source.join(
            validation_ids, on="id", how="left_semi"
        ).select(*source.columns)

        train.repartition(16).write.mode("overwrite").parquet(
            str(TRAIN_PATH)
        )

        validation.repartition(8).write.mode("overwrite").parquet(
            str(VALIDATION_PATH)
        )

        print("\n===== 4. 回读并检查 =====", flush=True)

        saved_train = spark.read.parquet(str(TRAIN_PATH))
        saved_validation = spark.read.parquet(
            str(VALIDATION_PATH)
        )

        train_count = saved_train.count()
        validation_count = saved_validation.count()

        if train_count + validation_count != total:
            raise ValueError("划分后的总记录数不一致")

        # 检查两部分有没有相同 ID。
        overlap_exists = (
            saved_train.select("id")
            .join(
                saved_validation.select("id"),
                on="id",
                how="inner",
            )
            .limit(1)
            .count()
        )

        if overlap_exists:
            raise ValueError("训练部分和验证部分存在相同 ID")

        expected_train = sum(
            row["count"]
            for row in distribution
            if row["subset"] == "train"
        )
        expected_validation = sum(
            row["count"]
            for row in distribution
            if row["subset"] == "validation"
        )

        if (
            train_count != expected_train
            or validation_count != expected_validation
        ):
            raise ValueError("保存后的条数与划分清单不一致")

        elapsed = time.perf_counter() - started

        summary = {
            "source": str(SOURCE_PATH),
            "method": "rating-stratified deterministic hash ordering",
            "seed_parameter": SEED,
            "spark_version": spark.version,
            "total_rows": total,
            "train_rows": train_count,
            "validation_rows": validation_count,
            "train_ratio": train_count / total,
            "validation_ratio": validation_count / total,
            "id_overlap": 0,
            "rating_distribution": [
                row.asDict() for row in distribution
            ],
            "total_seconds_including_validation": round(elapsed, 2),
        }

        with SUMMARY_PATH.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"训练部分：{train_count:,} 条", flush=True)
        print(f"验证部分：{validation_count:,} 条", flush=True)
        print(
            f"训练比例：{train_count / total:.4%}",
            flush=True,
        )
        print(
            f"验证比例：{validation_count / total:.4%}",
            flush=True,
        )
        print("两部分 ID 交叉：0 条", flush=True)
        print(f"总耗时：{elapsed:.2f} 秒", flush=True)
        print("SPLIT_OK", flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()