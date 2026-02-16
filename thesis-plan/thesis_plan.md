# Graph neural networks on graph databases

**Authors:** Victor Pekkari (`vi8011pe-s@student.lu.se`) & Harry Wik (`harry.wik.7273@student.lu.se`)  
**Date:** February 2026

---

## Background
Neo4j is a global leader in graph database technology, providing optimized storage and querying for highly interconnected data. Unlike tabular datasets where rows can be processed independently, graphs are defined by their relationships. This structure is leveraged by Graph Neural Networks (GNNs), which extend traditional neural models by incorporating the connections (edges) between data points (nodes).

In a GNN, the representation of a node may be computed using a "message-passing" mechanism that aggregates information from its neighbors. This allows the model to capture relational patterns that traditional neural networks miss. However, this dependency on neighboring nodes creates significant computational challenges. 

On large-scale graphs, a single worker node cannot typically hold all required neighborhoods in memory. Because large graphs resist simple partitioning, efficiently fetching these neighborhoods during training and inference becomes a non-trivial bottleneck. This project explores how graph databases, specifically Neo4j’s clustering and querying capabilities, can provide a unified interface to solve these scalability and data-access issues.



---

## Project scope
In this thesis, in collaboration with Neo4j, we will investigate the synergies between message-passing GNNs and graph databases. The project focuses on how GNNs can benefit from integration with a distributed graph database to overcome memory and partitioning limitations.

This includes researching the following primary areas:
* **Performance and Scalability:** Evaluating the performance gains that arise from a tight integration between GNNs and graph databases.
* **Neighborhood Fetching:** Designing and optimizing efficient strategies for fetching node neighborhoods from the database during the forward pass.
* **Distributed Training Throughput:** Describe the increased throughput of the system as we distribute the training over multiple workers.
* **Extending the system to transforming graphs:** What is the best way to react to modifications of the graph? 

---

## Method

### System architecture
We plan to implement a system where the GNN is trained on a machine separate from the graph that is stored on a graph database, parts of the graph will then be fetched iteratively in batches during the training process (the neighbors of a inference node will also be fetched from the graph DB during inference).

The Extract Transform Load (ETL) pipeline supplying the model with data is a multi-node experimental system wherein implementation differences of part such as:
* Data retrieval
* Structure of data transfer over the network
* Parallelism
* Cache layers

will achieve different outcomes in terms of throughput of useful information reaching the model. The aim of this paper is to measure throughput and iteratively propose and implement solutions to the main bottlenecks of the ETL-pipeline.

### Datasets
The datasets (consisting of homogenous graphs) to be used for training and evaluating the model are:

* **ogbn-arxiv** (170k nodes): A directed graph representing the citation network between all Computer Science (CS) arXiv papers indexed by MAG.
* **ogbn-products** (2.4M nodes): An undirected and unweighted graph representing an Amazon product co-purchasing network.
* **ogbn-papers100M** (111M nodes): A directed citation graph of 111 million papers indexed by MAG.

More information can be found at: [https://ogb.stanford.edu/docs/nodeprop/](https://ogb.stanford.edu/docs/nodeprop/)

---

## Experimental Setup
Experiments will follow a controlled variable approach, altering only one parameter per trial to ensure consistency.

* **Training Protocol:** Models will run until validation accuracy reaches an optimum or a $x$-hour limit is hit ($x$ is yet to be decided). 
* **Metrics:** We will log training/validation performance over time, throughput, total training time, batch size, and time-per-batch.

---

## Performance Evaluation
The implementation will be evaluated based on both predictive accuracy and computational efficiency.

* **Model Performance:** Node classification accuracy across the three different graphs.
* **System Efficiency:** Inference speed, sampling time vs. forward pass time (GPU utilization), and speed/GPU-ratio improvements compared to other popular GNN implementations. If time permits, we may also measure the speed-up gained by increasing the number of workers.
* **Sampling Analysis:** For different sampling methods, we will measure the number of sampled nodes/edges, sampling latency, and resulting model performance.

---

## Previous work
There are not that many previous research studies regarding synergies between GNNs and graph databases. That said, there has been one paper that showed that it is possible to use graph databases to train GNNs on machines that can't fit the whole graph in its memory: [https://arxiv.org/abs/2411.11375](https://arxiv.org/abs/2411.11375).

---

## Resources
The hiring contract with Neo4j will be from 2026-02-02 to 2026-06-30. The thesis project will be completed in time to present it on 2026-06-04. Neo4j will provide computers to work on.

---

## Involved parties

### Students
* **Victor Pekkari:** `vi8011pe-s@student.lu.se`
* **Harry Wik:** `harry.wik.7273@student.lu.se`

### Supervisors
* **Brian Shi**, Neo4j: `brian.shi@neo4j.com`
* **Alfred Clemedtson**, Neo4j: `alfred.clemedtson@neo4j.com`
* **Xuan-Son Vu**, Lund University: `xuan-son.vu@cs.lth.se`

### Examiner
* **Jacek Malec**, Lund University: `jacek.malec@cs.lth.se`