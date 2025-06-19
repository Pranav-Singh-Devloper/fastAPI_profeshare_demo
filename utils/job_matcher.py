import os, json, pickle
from BM_25 import preprocess_jobs, build_bm25_model, match_students_to_jobs

def run_bm25_match(student_data):
    base_dir = os.path.dirname(__file__) + "/.."
    job_files = ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]
    jobs = []

    for fname in job_files:
        path = os.path.join(base_dir, fname)
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
            jobs.extend(lines)

    job_texts, job_index = preprocess_jobs(jobs)
    bm25 = build_bm25_model(job_texts)
    matches = match_students_to_jobs(student_data, jobs, bm25, job_index)

    pkl_path = os.path.join(base_dir, "student_job_matches.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(matches, f)

    return matches, pkl_path
