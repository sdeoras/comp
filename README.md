# comp

## about
comp is a Go wrapper on TensorFlow intended to produce Go binaries with scientific
computation capabilities. It is also nice to just drop the binary on a target machine with
GPU and seamlessly take advantage of hardware acceleration.

## GPU setup instructions
[Setup Ubuntu 16.04 with NVIDIA GPU](https://gist.github.com/sdeoras/3e773f7e7402de0ef823c8d24d4b83f3) to run
TensorFlow jobs.

## GPU Acceleration
Below is a simple matrix inversion benchmark
![GPU Acceleration](/art/matrix-inversion-cpu-vs-gpu.png)

## RPI Build
You will need to have a forked copy of tensorflow when running on RPI devices.
```go
git clone https://github.com/sdeoras/tensorflow.git
cd tensorflow
git checkout r1.0.0
```
