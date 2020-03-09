#!/bin/bash

python -m odium.experiment.train_rts --env-name FetchPushAmongObstacles-v1 --env-id 4 --planning-env-id 4 --reward-type sparse --n-epochs 50 --n-cycles 10 --n-batches 10 --n-test-rollouts 20 --n-rollouts-per-cycle 5 --her --n-expansions 3 --n-offline-expansions 3 --n-rts-workers 3 --debug
