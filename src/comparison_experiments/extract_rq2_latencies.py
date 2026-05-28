import json
import os
import csv

def get_latency(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            run = data.get('run', {})
            metrics = data.get('metrics', {})
            
            batches_seen = run.get('batches_seen', 0)
            epoch_time_dict = metrics.get('epoch_time_s', {})
            total_epoch_time = epoch_time_dict.get('total_s', 0)
            
            if batches_seen > 0:
                return (total_epoch_time / batches_seen) * 1000
            return 0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

datasets = ["arxiv", "papers100M", "products"]
implementations = {
    "Baseline": "baseline_neo4j",
    "Java UDP": "java_neo4j",
    "Pre-agg": "preagg"
}

base_path = "experiment_results/report_files/rq_bottlenecks"

output_file = f"{base_path}/rq2_total_latency_per_batch_comparison.csv"

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dataset", "Baseline", "Java UDP", "Pre-agg"])
    
    for ds in datasets:
        row = [ds]
        for impl_name, impl_dir in implementations.items():
            path = os.path.join(base_path, impl_dir, ds, "measurements.json")
            latency = get_latency(path)
            row.append(f"{latency:.2f}" if latency is not None else "N/A")
        writer.writerow(row)

print(f"Results written to {output_file}")
