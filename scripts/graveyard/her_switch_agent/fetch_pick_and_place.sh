#!/bin/bash

mpirun -np 4 python -m odium.experiment.train_switch --env-name FetchPickAndPlace-v1
