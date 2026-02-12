# Thesis Plan: GNNs on Graph Databases

This thesis investigates the synergy between graph databases (Neo4j) and Machine Learning. Below is a three-step plan


## `Step 1:` System Architecture & Implementation
Step 1 is to implement the graph neural network tightly coupled with the database according to the architecture requirements below.

### `1.1`. high level architecture
![](arch.png)

### `1.2` Framework
* **Pytorch (Geometric):** The machine learning library to be used are pytorch and pytorch geometric. PyG has useful classes like GraphStore, featureStore (both implemented as a neo4j DB), BaseSampler (defines the sampling method, tightly coupled with the graphstore).



---

## `Step 2:` Scalability & Topology
Benchmark the system across datasets of three different graph sizes.

### `2.1` Scalability Tiers
* **Small:** `ogbn-arxiv` (~170k nodes) 
* **Medium:** `ogbn-products` (~2.4M nodes) 
* **Large:** `ogbn-papers100M` (~111M nodes)

### `2.2` Topology Factors
* **Homogeneity (Primary):** Use standard GraphSAGE to validate the pipeline.
* **Heterogeneity (Secondary):** If time permits, extend Cypher queries to handle **metapaths** (e.g., `Author-Writes-Paper-Cites-Paper`) for heterogeneous GNNs.

---

## `Step 3:` Task Generalization & In-Database Inference
Evaluate the system's flexibility beyond simple node classification if we have time.

### `3.1` Multi-Task Evaluation
* **Primary:** Node Classification (Accuracy, Throughput, Inference time).
* **Secondary:** Link Prediction and Regression to probe if sampling quality remains consistent across different loss functions.

### `3.2` In-Database Operations
* **Message Passing in Cypher:** Investigate if the **Aggregation** step (e.g., `MEAN` in GraphSAGE) can be performed within the Cypher query to further reduce data transfer volume.
* **Inference:** Develop a resource-efficient inference workflow where a "forward pass" for a new node is triggered via a single Cypher call, minimizing the need to recreate the graph structure in memory.

## Experimental setup
This sections answers how we plan to perform our training, and what we want to include in the final report regarding the training process
* outline the hardware used
* use node classification as the primary task, and use accuracy as the metric
* run until optimum (on accuracy on validation data) or stop after 24h
* record the continous increase of performance on training/validation data. know how much time is between the measurements
* training metrics; total training time, time to process one batch, batch size
* never alter more than one parameter in a single experiment
* compare results from graphs of different sizes
* duration of sampling phase vs training phase 
* measure throughput

## Performance evaluation
* node classification accuracy
* inference speed
* sampling time vs forward pass time (GPU utilization)
* speed-, GPU-ratio improvement to DGL, maybe others too
* when evaluating different sampling methods measure, nbr of sampled nodes, edges, and time (maybe compare model performance too if time permits)