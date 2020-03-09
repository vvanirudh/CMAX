#!/bin/bash

mpirun -np 4 python -m odium.experiment.train --env-name FetchPushWithObstacle-v1
