#!/bin/bash

mpirun -np 4 python -m odium.experiment.train_switch_residual --env-name ResidualFetchPush-v1
