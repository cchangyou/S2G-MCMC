#!/usr/bin/env sh
dist_dir=distribute

mpirun --hostfile hostfile -np 3 ./$dist_dir/bin/caffe.bin train --solver=./examples/mnist/lenet_solver_paral_BDK.prototxt

