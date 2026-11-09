"""Check the complete submission format, ID order, range and optional reference bytes."""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--sample', type=Path)
    p.add_argument('--reference', type=Path, help='Require byte identity with the archived final submission')
    a = p.parse_args()
    with a.csv.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != ['id', 'rating'] or len(rows) != 10001:
        raise ValueError('Expected header id,rating and exactly 10,000 data rows')
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError('Each data row must have two columns')
    ids = [int(row[0]) for row in rows[1:]]
    ratings = [float(row[1]) for row in rows[1:]]
    if ids != list(range(10000)):
        raise ValueError('Expected unique test IDs 0..9999 in prediction order')
    if not all(math.isfinite(x) and 1 <= x <= 5 for x in ratings):
        raise ValueError('Ratings must be finite and inside [1,5]')
    if a.sample:
        with a.sample.open(newline='', encoding='utf-8-sig') as f:
            reference = list(csv.reader(f))
        if reference[0] != rows[0] or [int(row[0]) for row in reference[1:]] != ids:
            raise ValueError('Header or ID order differs from the supplied sample')
    if a.reference and a.csv.read_bytes() != a.reference.read_bytes():
        raise ValueError('CSV differs from the archived final submission')
    print('SUBMISSION_CHECK_PASS', 'rows', len(ids), 'min', min(ratings), 'max', max(ratings),
          'mean', statistics.mean(ratings), 'sha256', hashlib.sha256(a.csv.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
