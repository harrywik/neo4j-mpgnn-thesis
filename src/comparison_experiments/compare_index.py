import statistics
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connection details from environment
URI = os.getenv("URI", "neo4j://localhost:7687")
USER = os.getenv("USERNAME", "neo4j")
PASSWORD = os.getenv("PASSWORD", "password")
AUTH = (USER, PASSWORD)

DATABASES = ["arxiv2", "papers100M"]

# Increased range to get better signal-to-noise ratio
# We'll use a parameter for the range size to keep it flexible
QUERY = """
UNWIND range(0, $limit) AS targetId
MATCH (p:Paper {id: targetId}) 
RETURN elementId(p) AS internal_id;
"""

PROFILE_QUERY = "PROFILE " + QUERY

NUM_RUNS = 30
WARMUP_RUNS = 10
QUERY_LIMIT = 5000 # Increased from 127 to 5000

def get_db_hits(profile):
    """Recursively sum db_hits from a profile."""
    hits = profile.get('dbHits', 0)
    for child in profile.get('children', []):
        hits += get_db_hits(child)
    return hits

def verify_schema(session, database):
    print(f"\n--- Schema & Plan Verification for '{database}' ---")
    # Check for index
    result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties, state, type WHERE 'Paper' IN labelsOrTypes")
    indexes = list(result)
    for idx in indexes:
        print(f"Index: {idx['name']} | Props: {idx['properties']} | State: {idx['state']}")

    # Check plan and DB Hits once
    result = session.run(PROFILE_QUERY, limit=QUERY_LIMIT)
    records = list(result)
    summary = result.consume()
    total_hits = get_db_hits(summary.profile)

    print(f"Nodes Found  : {len(records)}")
    print(f"Total DB Hits: {total_hits}")
    print(f"Main Operator: {summary.profile.get('operatorType')}")

    return total_hits

def run_benchmark(database):
    times = []

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=database) as session:
            db_hits = verify_schema(session, database)

            print(f"Running {NUM_RUNS} iterations (no PROFILE)...")

            for i in range(NUM_RUNS):
                # We measure the total time for the server to process and for us to receive
                result = session.run(QUERY, limit=QUERY_LIMIT)
                list(result) # Fully consume results
                summary = result.consume()

                # total_time = available_after + consumed_after
                total_time_ms = summary.result_available_after + summary.result_consumed_after
                times.append(total_time_ms)

    # Isolate the warm runs
    warm_times = times[WARMUP_RUNS:]

    if not warm_times:
        return None

    mu = statistics.mean(warm_times)
    std = statistics.stdev(warm_times) if len(warm_times) > 1 else 0

    return {
        "database": database,
        "warm_times": warm_times,
        "mean": mu,
        "std": std,
        "db_hits": db_hits
    }

if __name__ == "__main__":
    all_results = []
    for db in DATABASES:
        try:
            res = run_benchmark(db)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"Error benchmarking database '{db}': {e}")

    print("\n" + "="*40)
    print("         BENCHMARK RESULTS (Limit: " + str(QUERY_LIMIT) + ")")
    print("="*40)

    for res in all_results:
        print(f"\nDatabase: {res['database']}")
        print(f"Total DB Hits       : {res['db_hits']}")
        print(f"Warm runs (ms)      : {res['warm_times']}")
        print(f"Mean (μ)            : {res['mean']:.3f} ms")
        print(f"Std Dev (σ)         : {res['std']:.3f} ms")
        print(f"Time per seek       : {res['mean']/QUERY_LIMIT*1000:.3f} μs")

    if len(all_results) == 2:
        r1, r2 = all_results
        diff = r2['mean'] - r1['mean']
        factor = r2['mean'] / r1['mean'] if r1['mean'] != 0 else float('inf')
        print("\n" + "-"*40)
        print(f"Comparison: {r2['database']} vs {r1['database']}")
        print(f"Difference: {diff:+.3f} ms")
        print(f"Factor    : {factor:.2f}x")
        print("-" * 40)