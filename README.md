# go-scicomp
go-scicomp is a Go wrapper on TensorFlow intended to produce Go binaries with scientific
computation capabilities. It is also nice to just drop the binary on a target machine with
GPU and seamlessly take advantage of hardware acceleration.

## GPU setup instructions
[Setup Ubuntu 16.04 with NVIDIA GPU](https://gist.github.com/sdeoras/3e773f7e7402de0ef823c8d24d4b83f3) to run
TensorFlow jobs.

## GPU Acceleration
Below is a simple matrix inversion benchmark
![GPU Acceleration](/art/matrix-inversion-cpu-vs-gpu.png)

## Running on Raspberry Pi
Checkout tag `r1.0.0` of this repo to run on raspberry pi devices.
Also see the README in that tag for further instructions on setting up tensorflow for RPI
