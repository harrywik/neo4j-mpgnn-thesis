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

# The isolated index-seek query
QUERY = """
PROFILE 
UNWIND range(0, 127) AS targetId
MATCH (p:Paper {id: targetId}) 
RETURN elementId(p) AS internal_id;
"""

NUM_RUNS = 20
WARMUP_RUNS = 5

def verify_schema(session, database):
    print(f"\n--- Schema Verification for '{database}' ---")
    # Check for index
    result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties, state, type WHERE 'Paper' IN labelsOrTypes")
    indexes = list(result)
    if not indexes:
        print("WARNING: No index found on :Paper")
    for idx in indexes:
        print(f"Index: {idx['name']} | Labels: {idx['labelsOrTypes']} | Props: {idx['properties']} | State: {idx['state']} | Type: {idx['type']}")
    
    # Check count of nodes
    result = session.run("MATCH (p:Paper) RETURN count(p) AS total")
    print(f"Total :Paper nodes: {result.single()['total']}")
    
    result = session.run("MATCH (p:Paper) WHERE p.id IS NOT NULL RETURN count(p) AS with_id")
    print(f"Nodes with 'id' property: {result.single()['with_id']}")

def run_benchmark(database):
    times = []
    record_counts = []
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session(database=database) as session:
            verify_schema(session, database)
            
            print(f"\nRunning {NUM_RUNS} iterations on '{database}'...")
            
            for i in range(NUM_RUNS):
                result = session.run(QUERY)
                records = list(result)
                summary = result.consume()
                
                record_counts.append(len(records))
                
                # server-side execution time in milliseconds
                db_time_ms = summary.result_available_after
                times.append(db_time_ms)
                
                if i == 0:
                    # Print simplified plan info
                    print(f"First run plan info: {summary.profile.get('operatorType') if summary.profile else 'No profile'}")
                    print(f"First run records found: {len(records)}")

    # Check if we actually found any records
    avg_records = sum(record_counts) / len(record_counts)
    if avg_records == 0:
        print(f"CRITICAL: No records found for query in '{database}'")

    # Isolate the warm runs
    warm_times = times[WARMUP_RUNS:]
    
    if not warm_times:
        return None
        
    mu = statistics.mean(warm_times)
    std = statistics.stdev(warm_times) if len(warm_times) > 1 else 0
    
    return {
        "database": database,
        "cold_times": times[:WARMUP_RUNS],
        "warm_times": warm_times,
        "mean": mu,
        "std": std,
        "avg_records": avg_records
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
    print("         BENCHMARK RESULTS")
    print("="*40)
    
    for res in all_results:
        print(f"\nDatabase: {res['database']}")
        print(f"Avg Records Found   : {res['avg_records']}")
        print(f"Cold runs (discarded): {res['cold_times']} ms")
        print(f"Warm runs evaluated : {res['warm_times']} ms")
        print(f"Mean (μ)            : {res['mean']:.3f} ms")
        print(f"Std Dev (σ)         : {res['std']:.3f} ms")
    
    if len(all_results) == 2:
        r1, r2 = all_results
        diff = r2['mean'] - r1['mean']
        factor = r2['mean'] / r1['mean'] if r1['mean'] != 0 else float('inf')
        print("\n" + "-"*40)
        print(f"Comparison: {r2['database']} vs {r1['database']}")
        print(f"Difference: {diff:+.3f} ms")
        print(f"Factor    : {factor:.2f}x")
        print("-"*40)