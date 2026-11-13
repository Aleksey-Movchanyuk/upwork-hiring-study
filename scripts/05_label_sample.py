"""Step 5 (manual): write a sheet of random postings for hand-labelling.

You can usually tell AI-written from hand-written job posts. Label ~200 of them
(column `your_label`: ai / human / unsure) so step 6 can measure how well the
automatic "AI-style signals" proxy matches a human judgement.

The sheet contains raw job text, so it is PRIVATE: it is git-ignored and must
never be published or shared. Only aggregate results leave this folder.
"""
import csv
import hashlib
import sys

from dotenv import load_dotenv

from _common import DATA, ROOT

load_dotenv(ROOT / ".env")
from upwork_ml.db import dict_cursor, get_connection  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = DATA / "label_sample_PRIVATE.csv"

QUERY = """
select j.uid, j.title, j.description
from jobs j join job_outcomes o on o.uid = j.uid
order by md5(j.uid || 'article-label-seed')
limit %s
"""

conn = get_connection()
with dict_cursor(conn) as cur:
    cur.execute(QUERY, (N,))
    rows = cur.fetchall()
conn.close()

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["job_key", "title", "description_first_900_chars", "your_label"])
    for r in rows:
        key = hashlib.sha1(r["uid"].encode()).hexdigest()[:12]
        w.writerow([key, r["title"], (r["description"] or "")[:900].replace("\r", " "), ""])
print(f"wrote {len(rows)} postings to {OUT.relative_to(ROOT)}; fill `your_label` with ai / human / unsure")
