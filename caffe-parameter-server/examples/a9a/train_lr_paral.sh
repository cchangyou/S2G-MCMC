#!/usr/bin/env sh
dist_dir=distribute

mpirun --hostfile hostfile -np 2 ./$dist_dir/bin/caffe.bin train --solver=./examples/a9a/fnn_solver_paral.prototxt
