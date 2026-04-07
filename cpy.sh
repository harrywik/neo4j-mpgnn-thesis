#!/bin/bash

rsync -avz --exclude '.git' --exclude '.venv' -e ssh . harry.wik@neo4j:~/neo4j-mpgnn-thesis
