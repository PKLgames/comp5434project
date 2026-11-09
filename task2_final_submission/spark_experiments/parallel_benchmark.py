import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

# 复用已有的基础特征规则和环境配置。
from feature_processing import add_features, NUMERIC_FEATURES
from clean_data import PROJECT_DIR

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


INPUT_PATH = PROJECT_DIR / "output" / "train_80"
RESULT_DIR = PROJECT_DIR / "output" / "parallel_benchmark"
EXPECTED_ROWS = 6_104_774
REPEATS = 3


def experiment_signature():
    """检查不同线程实验是否使用相同的数据和代码。"""
    files = sorted(INPUT_PATH.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"没有找到训练数据：{INPUT_PATH}")

    information = {
        "input_path": str(INPUT_PATH.resolve()),
        "files": [
            [p.name, p.stat().st_size, p.stat().st_mtime_ns]
            for p in files
        ],
        "feature_code": hashlib.sha256(
            (PROJECT_DIR / "feature_processing.py").read_bytes()
        ).hexdigest(),
        "benchmark_code": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }

    return hashlib.sha256(
        json.dumps(information, sort_keys=True).encode("utf-8")
    ).hexdigest()


def workload(spark):
    """读取数据、计算基础特征、执行聚合以触发真实计算。"""
    source = spark.read.parquet(str(INPUT_PATH)).select(
        "title",
        "comment",
        "votes",
        "purchased",
        "time",
        "comment_missing",
        "votes_invalid",
    )

    featured = add_features(source)

    # 对每个数值特征求和，确保特征真的被计算。
    # 只做 count() 可能让 Spark 跳过不需要的特征计算。
    expressions = [F.count("*").alias("rows")]

    for column in NUMERIC_FEATURES:
        expressions.append(
            F.sum(
                F.col(column).cast("double")
            ).alias(f"sum_{column}")
        )

    return featured.agg(*expressions).first().asDict()


def verify(result, reference):
    if result["rows"] != EXPECTED_ROWS:
        raise ValueError(f"记录数不正确：{result['rows']}")

    for key in reference:
        if key == "rows":
            continue

        if result[key] is None or not math.isclose(
            result[key],
            reference[key],
            rel_tol=1e-9,
            abs_tol=1e-5,
        ):
            raise ValueError(f"计算结果不一致：{key}")


def run_benchmark(threads):
    signature = experiment_signature()

    spark = (
        SparkSession.builder
        .appName(f"Task2Benchmark{threads}Threads")
        .master(f"local[{threads}]")
        .config("spark.driver.memory", "4g")
        # 除工作线程数外，保持这些执行设置一致。
        .config("spark.default.parallelism", "64")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.files.minPartitionNum", "64")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        if spark.sparkContext.master != f"local[{threads}]":
            raise ValueError("实际线程配置与要求不同")

        print(f"\n使用 {threads} 个工作线程", flush=True)
        print("正在进行一次完整预热，不计入正式耗时……", flush=True)

        reference = workload(spark)

        if reference["rows"] != EXPECTED_ROWS:
            raise ValueError("预热读取的数据条数不正确")

        seconds = []

        for repeat in range(1, REPEATS + 1):
            # 不主动缓存训练数据。
            # 此操作也不会清除 Windows 的文件缓存。
            spark.catalog.clearCache()

            print(
                f"\n开始正式测量 {repeat}/{REPEATS}……",
                flush=True,
            )

            started = time.perf_counter()
            result = workload(spark)
            elapsed = time.perf_counter() - started

            # 校验放在计时结束之后。
            verify(result, reference)
            seconds.append(elapsed)

            print(f"本次耗时：{elapsed:.2f} 秒", flush=True)

        report = {
            "threads": threads,
            "spark_version": spark.version,
            "rows": EXPECTED_ROWS,
            "signature": signature,
            "warmup_runs": 1,
            "measured_runs": REPEATS,
            "seconds": seconds,
            "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds),
            "stdev_seconds": statistics.stdev(seconds),
            "checksum": reference,
            "scope": (
                "Input reading + basic feature computation + aggregation; "
                "excludes Spark startup, warmup and result-file writing"
            ),
        }

        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULT_DIR / f"threads_{threads}.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print(
            f"\n平均耗时：{report['mean_seconds']:.2f} 秒",
            flush=True,
        )
        print(
            f"中位耗时：{report['median_seconds']:.2f} 秒",
            flush=True,
        )
        print(f"结果：{report_path}", flush=True)
        print(f"BENCHMARK_{threads}_OK", flush=True)

    finally:
        spark.stop()


def summarize():
    reports = []

    for threads in [1, 2, 4]:
        path = RESULT_DIR / f"threads_{threads}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 {threads} 线程结果，请先完成该项实验"
            )

        with path.open(encoding="utf-8") as file:
            report = json.load(file)

        if report["threads"] != threads:
            raise ValueError("结果文件与线程数不一致")

        reports.append(report)

    if len({r["signature"] for r in reports}) != 1:
        raise ValueError(
            "三项实验的数据或代码不一致，需要使用同一版本重跑"
        )

    if len({r["spark_version"] for r in reports}) != 1:
        raise ValueError("三项实验使用的 Spark 版本不同")

    reference = reports[0]["checksum"]

    for report in reports:
        verify(report["checksum"], reference)

    baseline = reports[0]["median_seconds"]
    rows = []

    print("\n===== 线程效率实验汇总 =====")

    for report in reports:
        threads = report["threads"]
        speedup = baseline / report["median_seconds"]

        row = {
            "threads": threads,
            "mean_seconds": round(report["mean_seconds"], 3),
            "median_seconds": round(report["median_seconds"], 3),
            "stdev_seconds": round(report["stdev_seconds"], 3),
            "speedup_vs_1_thread": round(speedup, 3),
            "parallel_efficiency_percent": round(
                speedup / threads * 100, 2
            ),
        }
        rows.append(row)

        print(
            f"{threads} 线程："
            f"平均 {row['mean_seconds']} 秒，"
            f"中位 {row['median_seconds']} 秒，"
            f"加速比 {row['speedup_vs_1_thread']}"
        )

    output_path = RESULT_DIR / "summary.csv"

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n汇总表：{output_path}")
    print("BENCHMARK_SUMMARY_OK")


def main():
    parser = argparse.ArgumentParser()

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--threads", type=int, choices=[1, 2, 4])
    action.add_argument("--summary", action="store_true")

    args = parser.parse_args()

    if args.summary:
        summarize()
    else:
        run_benchmark(args.threads)


if __name__ == "__main__":
    main()