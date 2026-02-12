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
* **Homogeneity (Primary):** (look at heterogenous graphs if time permits)

---

## `Step 3:` Task Generalization & In-Database Inference


### `3.1` ML task
* **Node Classification:**  we use classifiation as the task for performance evaluation,, and we measure accuracy, throughput, inference time. If time permits we may look at other ML tasks too like link prediction or regression


### `3.2` In-Database Operations
* **Message Passing in Cypher:** Investigate if the some implementation of the message passing function can be defined as a cypher query so that inference for new nodes can be done directly in cypher.

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