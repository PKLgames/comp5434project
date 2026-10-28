import json
import time

# 复用环境、路径和数值特征清单，不执行之前的 main()。
from clean_data import PROJECT_DIR
from feature_processing import NUMERIC_FEATURES

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Normalizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "text_models"

# 第一版候选维度，后续可通过验证实验调整。
TITLE_DIMENSION = 2 ** 16       # 65,536
COMMENT_DIMENSION = 2 ** 18     # 262,144

# 至少出现在两个训练文档中的哈希位置才保留 IDF 权重。
MIN_DOC_FREQ = 2


def fit_text_pipeline(train, tokens_column, prefix, dimension):
    """仅使用训练部分的分词列拟合 IDF。"""

    tf_column = f"_{prefix}_tf"
    weighted_column = f"_{prefix}_weighted"
    output_column = f"{prefix}_tfidf"

    pipeline = Pipeline(stages=[
        HashingTF(
            inputCol=tokens_column,
            outputCol=tf_column,
            numFeatures=dimension,
            binary=False,
        ),
        IDF(
            inputCol=tf_column,
            outputCol=weighted_column,
            minDocFreq=MIN_DOC_FREQ,
        ),
        # 将每条文本向量做 L2 归一化。
        # 空向量仍然保持为空向量，不需要除以零。
        Normalizer(
            inputCol=weighted_column,
            outputCol=output_column,
            p=2.0,
        ),
    ])

    print(f"\n正在拟合 {prefix} 的 IDF……", flush=True)
    started = time.perf_counter()

    # 不传入评分列，也不使用验证数据。
    model = pipeline.fit(train.select(tokens_column))
    fit_seconds = time.perf_counter() - started

    model_path = MODEL_DIR / f"{prefix}_tfidf"
    model.write().overwrite().save(str(model_path))

    # 从磁盘重新加载，确认后续可以复用。
    loaded = PipelineModel.load(str(model_path))

    if loaded.stages[1].idf.size != dimension:
        raise ValueError(f"{prefix} 模型维度检查失败")

    print(f"IDF 拟合耗时：{fit_seconds:.2f} 秒", flush=True)
    print(f"处理器已保存：{model_path}", flush=True)

    return loaded, {
        "input_column": tokens_column,
        "output_column": output_column,
        "dimension": dimension,
        "min_doc_freq": MIN_DOC_FREQ,
        "normalization": "L2",
        "idf_fit_seconds": round(fit_seconds, 2),
        "model_path": str(model_path),
    }


def transform_and_save(
    spark,
    source,
    title_model,
    comment_model,
    output_name,
    expected_rows,
):
    """应用已拟合的同一套处理器，不在这里重新拟合。"""

    print(f"\n===== 生成 {output_name} =====", flush=True)
    started = time.perf_counter()

    transformed = title_model.transform(source)
    transformed = comment_model.transform(transformed)

    # 输出模型输入所需的列，避免重复保存正文和中间向量。
    # id 用于对齐，rating 是目标，都不属于输入特征。
    result = transformed.select(
        "id",
        "rating",
        *NUMERIC_FEATURES,
        "title_tfidf",
        "comment_tfidf",
    )

    output_path = OUTPUT_DIR / output_name
    result.write.mode("overwrite").parquet(str(output_path))

    processing_seconds = time.perf_counter() - started
    check_started = time.perf_counter()

    saved = spark.read.parquet(str(output_path))

    stats = saved.agg(
        F.count("*").alias("rows"),
        F.sum(
            F.when(
                F.col("title_tfidf").isNull()
                | F.col("comment_tfidf").isNull(),
                1,
            ).otherwise(0)
        ).alias("null_vector_rows"),
    ).first()

    if stats["rows"] != expected_rows:
        raise ValueError(
            f"记录数不一致：预期 {expected_rows}，"
            f"实际 {stats['rows']}"
        )

    if stats["null_vector_rows"] != 0:
        raise ValueError("保存结果存在空值向量")

    # 只取三条展示维度和非零元素数量。
    # 不把高维稀疏向量转换成稠密数组。
    examples = saved.select(
        "id", "title_tfidf", "comment_tfidf"
    ).limit(3).collect()

    print(f"保存后记录数：{stats['rows']:,}", flush=True)
    print(f"空值向量记录数：{stats['null_vector_rows']}", flush=True)
    print("\n向量样例：", flush=True)

    for row in examples:
        title = row["title_tfidf"]
        comment = row["comment_tfidf"]

        if title.size != TITLE_DIMENSION:
            raise ValueError("样本标题向量维度不符")

        if comment.size != COMMENT_DIMENSION:
            raise ValueError("样本正文向量维度不符")

        print(
            f"id={row['id']}；"
            f"标题维度={title.size}，"
            f"非零元素={title.numNonzeros()}；"
            f"正文维度={comment.size}，"
            f"非零元素={comment.numNonzeros()}",
            flush=True,
        )

    check_seconds = time.perf_counter() - check_started

    print(
        f"向量生成与保存耗时：{processing_seconds:.2f} 秒",
        flush=True,
    )
    print(f"输出目录：{output_path}", flush=True)

    return {
        "output": str(output_path),
        "rows": stats["rows"],
        "null_vector_rows": stats["null_vector_rows"],
        "transform_and_write_seconds": round(processing_seconds, 2),
        "validation_and_display_seconds": round(check_seconds, 2),
    }


def main():
    spark = (
        SparkSession.builder
        .appName("Task2TextVectorization")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    started = time.perf_counter()

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # 只读取本步骤需要的列。
        needed_columns = [
            "id",
            "rating",
            *NUMERIC_FEATURES,
            "title_tokens",
            "comment_tokens",
        ]

        train = spark.read.parquet(
            str(OUTPUT_DIR / "train_features_base")
        ).select(*needed_columns)

        validation = spark.read.parquet(
            str(OUTPUT_DIR / "validation_features_base")
        ).select(*needed_columns)

        # 两次 fit 均只使用 train。
        title_model, title_info = fit_text_pipeline(
            train,
            tokens_column="title_tokens",
            prefix="title",
            dimension=TITLE_DIMENSION,
        )

        comment_model, comment_info = fit_text_pipeline(
            train,
            tokens_column="comment_tokens",
            prefix="comment",
            dimension=COMMENT_DIMENSION,
        )

        train_info = transform_and_save(
            spark,
            train,
            title_model,
            comment_model,
            output_name="train_features_tfidf",
            expected_rows=6_104_774,
        )

        validation_info = transform_and_save(
            spark,
            validation,
            title_model,
            comment_model,
            output_name="validation_features_tfidf",
            expected_rows=1_526_196,
        )

        summary = {
            "method": "HashingTF + IDF + L2 normalization",
            "idf_fit_source": "train_features_base only",
            "validation_used_for_fitting": False,
            "numeric_features": NUMERIC_FEATURES,
            "text_features": ["title_tfidf", "comment_tfidf"],
            "identifier": "id",
            "target": "rating",
            "numeric_features_scaled": False,
            "models": [title_info, comment_info],
            "datasets": [train_info, validation_info],
            "spark_version": spark.version,
            "total_seconds_including_validation": round(
                time.perf_counter() - started, 2
            ),
        }

        summary_path = OUTPUT_DIR / "text_vectorization_summary.json"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(f"\n处理记录：{summary_path}", flush=True)
        print("TFIDF_OK", flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()