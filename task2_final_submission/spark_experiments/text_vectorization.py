import argparse
import json
import math
import time
from datetime import datetime

from clean_data import PROJECT_DIR
from feature_processing import NUMERIC_FEATURES

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Normalizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_DIR = PROJECT_DIR / "output"

ENTITY_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in ["user", "product", "parent_product"]
    for suffix in ["review_count", "review_count_log", "unseen"]
]


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title-dim", type=int, default=4096)
    parser.add_argument("--comment-dim", type=int, default=16384)
    args = parser.parse_args()

    for value in [args.title_dim, args.comment_dim]:
        if value < 2 or value & (value - 1):
            parser.error("维度必须是大于等于 2 的 2 的幂")

    return args


def fit_model(train, prefix, dimension, model_dir):
    tokens = f"{prefix}_tokens"
    tf = f"_{prefix}_tf"
    weighted = f"_{prefix}_weighted"
    vector = f"{prefix}_tfidf"

    pipeline = Pipeline(stages=[
        HashingTF(
            inputCol=tokens,
            outputCol=tf,
            numFeatures=dimension,
            binary=False,
        ),
        IDF(
            inputCol=tf,
            outputCol=weighted,
            minDocFreq=2,
        ),
        Normalizer(
            inputCol=weighted,
            outputCol=vector,
            p=2.0,
        ),
    ])

    print(f"\n正在拟合 {prefix}：{dimension:,} 维", flush=True)
    started = time.perf_counter()

    # 只用训练部分的词语列，不使用验证数据或评分。
    model = pipeline.fit(train.select(tokens))

    elapsed = time.perf_counter() - started
    path = model_dir / f"{prefix}_tfidf"
    model.write().save(str(path))

    loaded = PipelineModel.load(str(path))
    if loaded.stages[1].idf.size != dimension:
        raise ValueError(f"{prefix} 处理器维度不正确")

    print(f"拟合耗时：{elapsed:.2f} 秒", flush=True)
    return loaded, elapsed


def attach_entities(spark, source, directory, expected):
    mapping = spark.read.parquet(str(directory)).select(
        "id",
        F.col("rating").alias("_old_rating"),
        *ENTITY_FEATURES,
    )

    # 防止重复 ID 导致关联后数据膨胀。
    stats = mapping.agg(
        F.count("*").alias("rows"),
        F.countDistinct("id").alias("ids"),
    ).first()

    if stats["rows"] != expected or stats["ids"] != expected:
        raise ValueError(f"旧实体特征数量或 ID 异常：{directory}")

    joined = source.join(mapping, on="id", how="left")

    invalid = (
        F.col("_old_rating").isNull()
        | (F.col("rating") != F.col("_old_rating"))
    )
    for column in ENTITY_FEATURES:
        invalid = invalid | F.col(column).isNull()

    if joined.filter(invalid).limit(1).count():
        raise ValueError("旧实体特征与当前数据的 ID 或评分不匹配")

    return joined.drop("_old_rating")


def save_features(
    spark, source, models, dimensions,
    output_path, expected, include_entities,
):
    started = time.perf_counter()

    result = models[0].transform(source)
    result = models[1].transform(result)

    numeric = list(NUMERIC_FEATURES)
    if include_entities:
        numeric += ENTITY_FEATURES

    result = result.select(
        "id", "rating", *numeric,
        "title_tfidf", "comment_tfidf",
    )

    # 新目录保存，不覆盖原来的特征。
    result.write.parquet(str(output_path))
    write_seconds = time.perf_counter() - started

    saved = spark.read.parquet(str(output_path))

    invalid = (
        F.col("title_tfidf").isNull()
        | F.col("comment_tfidf").isNull()
        | F.col("rating").isNull()
        | ~F.col("rating").isin(1, 2, 3, 4, 5)
    )
    for column in numeric:
        invalid = (
            invalid
            | F.col(column).isNull()
            | F.isnan(F.col(column).cast("double"))
        )

    stats = saved.agg(
        F.count("*").alias("rows"),
        F.countDistinct("id").alias("unique_ids"),
        F.sum(
            F.when(invalid, 1).otherwise(0)
        ).alias("invalid_rows"),
    ).first().asDict()

    if stats["rows"] != expected or stats["unique_ids"] != expected:
        raise ValueError(f"记录数或 ID 检查失败：{stats}")
    if stats["invalid_rows"] != 0:
        raise ValueError(f"存在异常特征：{stats}")

    print(f"\n保存记录数：{stats['rows']:,}", flush=True)

    # 仅检查三条样本，不把全量向量转换成数组。
    samples = saved.select(
        "id", "title_tfidf", "comment_tfidf"
    ).limit(3).collect()

    for row in samples:
        for column, dimension in zip(
            ["title_tfidf", "comment_tfidf"], dimensions
        ):
            vector = row[column]
            if vector.size != dimension:
                raise ValueError(f"{column} 维度不符")

            # 同时兼容 Spark 稀疏向量和稠密向量。
            values = (
                vector.values
                if hasattr(vector, "values")
                else vector
            )
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{column} 包含非有限值")

        print(
            f"id={row['id']}；"
            f"标题={row['title_tfidf'].size}维；"
            f"正文={row['comment_tfidf'].size}维",
            flush=True,
        )

    print(f"输出：{output_path}", flush=True)

    return {
        **stats,
        "output": str(output_path),
        "transform_and_write_seconds": round(write_seconds, 3),
    }


def main():
    args = get_arguments()

    base_names = ["train_features_base", "validation_features_base"]
    entity_names = [
        "train_features_with_entity",
        "validation_features_with_entity",
    ]
    expected_counts = [6_104_774, 1_526_196]

    for name in base_names:
        if not (OUTPUT_DIR / name).is_dir():
            raise FileNotFoundError(f"缺少：{OUTPUT_DIR / name}")

    available = [(OUTPUT_DIR / name).is_dir() for name in entity_names]
    if any(available) and not all(available):
        raise FileNotFoundError("旧版训练、验证实体特征必须同时存在")

    include_entities = all(available)

    name = (
        f"t{args.title_dim}_c{args.comment_dim}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir = OUTPUT_DIR / "text_dimension_experiments" / name
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = run_dir / "text_models"
    model_dir.mkdir()

    summary = {
        "status": "running",
        "title_dimension": args.title_dim,
        "comment_dimension": args.comment_dim,
        "entity_features_reused": include_entities,
        "idf_fit_source": "train_features_base only",
        "validation_used_for_fitting": False,
        "numeric_features_scaled": False,
        "numeric_features": list(NUMERIC_FEATURES) + (
            ENTITY_FEATURES if include_entities else []
        ),
        "text_features": ["title_tfidf", "comment_tfidf"],
        "min_doc_freq": 2,
        "normalization": "L2",
    }

    spark = None
    started = time.perf_counter()

    try:
        spark = (
            SparkSession.builder
            .appName("Task2LowerDimensionTFIDF")
            .master("local[4]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "64")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        print(f"本次实验目录：{run_dir}", flush=True)
        print(f"复用实体特征：{include_entities}", flush=True)

        columns = [
            "id", "rating", *NUMERIC_FEATURES,
            "title_tokens", "comment_tokens",
        ]

        sources = [
            spark.read.parquet(
                str(OUTPUT_DIR / name)
            ).select(*columns)
            for name in base_names
        ]

        for source, expected in zip(sources, expected_counts):
            if source.count() != expected:
                raise ValueError("基础特征条数与固定划分不一致")

        title_model, title_seconds = fit_model(
            sources[0], "title", args.title_dim, model_dir
        )
        comment_model, comment_seconds = fit_model(
            sources[0], "comment", args.comment_dim, model_dir
        )

        summary["idf_fit_seconds"] = {
            "title": title_seconds,
            "comment": comment_seconds,
        }
        summary["datasets"] = []

        for index, split in enumerate(["train", "validation"]):
            source = sources[index]

            if include_entities:
                source = attach_entities(
                    spark,
                    source,
                    OUTPUT_DIR / entity_names[index],
                    expected_counts[index],
                )

            suffix = "with_entity" if include_entities else "tfidf"
            output_path = run_dir / f"{split}_features_{suffix}"

            report = save_features(
                spark,
                source,
                [title_model, comment_model],
                [args.title_dim, args.comment_dim],
                output_path,
                expected_counts[index],
                include_entities,
            )
            summary["datasets"].append(report)

        summary["status"] = "complete"
        print("\nTEXT_DIMENSION_EXPERIMENT_OK", flush=True)
        print(f"新结果全部位于：{run_dir}", flush=True)

    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise

    finally:
        summary["total_seconds"] = round(
            time.perf_counter() - started, 3
        )

        try:
            with (run_dir / "text_vectorization_summary.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
        finally:
            if spark is not None:
                spark.stop()


if __name__ == "__main__":
    main()