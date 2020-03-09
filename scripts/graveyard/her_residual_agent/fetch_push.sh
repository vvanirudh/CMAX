#!/bin/bash

mpirun -np 4 python -m odium.experiment.train_residual --env-name ResidualFetchPush-v1
