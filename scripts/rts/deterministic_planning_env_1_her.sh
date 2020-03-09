#!/bin/bash

python -m odium.experiment.train_rts --env-name FetchPushAmongObstacles-v1 --env-id 1 --planning-env-id 1 --reward-type sparse --n-epochs 50 --n-cycles 1 --n-batches 10 --n-test-rollouts 1 --n-rollouts-per-cycle 1 --her --n-expansions 3 --n-offline-expansions 3 --n-rts-workers 3 --deterministic --debug
