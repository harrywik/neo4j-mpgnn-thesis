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
RETURN elementId(p);
"""

NUM_RUNS = 20
WARMUP_RUNS = 5

def run_benchmark(database):
    times = []
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        # Check if we can connect and if database exists
        with driver.session(database=database) as session:
            print(f" Running {NUM_RUNS} iterations on '{database}'...")
            
            for i in range(NUM_RUNS):
                result = session.run(QUERY)
                summary = result.consume()  # Forces query execution to finish
                
                # server-side execution time in milliseconds
                db_time_ms = summary.result_available_after
                times.append(db_time_ms)
                
    # Isolate the warm runs
    warm_times = times[WARMUP_RUNS:]
    
    # Calculate stats
    if not warm_times:
        return None
        
    mu = statistics.mean(warm_times)
    std = statistics.stdev(warm_times) if len(warm_times) > 1 else 0
    
    return {
        "database": database,
        "cold_times": times[:WARMUP_RUNS],
        "warm_times": warm_times,
        "mean": mu,
        "std": std
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