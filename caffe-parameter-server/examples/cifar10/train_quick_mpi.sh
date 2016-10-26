#!/usr/bin/env sh

dist_dir=distribute

mpirun --hostfile hostfile -np 3 ./$dist_dir/bin/caffe.bin train --solver=examples/cifar10/cifar10_quick_solver_mpi.prototxt

