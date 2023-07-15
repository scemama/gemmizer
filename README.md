# Gemmizer

This Python code sets up a ZeroMQ-based server that can perform matrix
multiplication tasks with given parameters. It is built to handle either
CPU-based or GPU-based computation using Numpy and Cupy libraries respectively.
The whole system is built for concurrent execution of these tasks, using Python
threading.


The worker threads run independently of each other and can process tasks
concurrently. The number of tasks they can process at the same time is limited
by the number of threads, which is set by Nstreams.

The ZeroMQ library allows this system to work across multiple processes or even
across networked machines, depending on the endpoints used for the sockets. In
this case, the clients connect to the server via TCP on port 5555, while the
workers connect via an in-process endpoint.

One important aspect of this code is its ability to handle numpy and cupy
computations. The choice between numpy and cupy is determined by the global
variable target. This is a powerful feature as cupy allows GPU-accelerated
computations which can be significantly faster for large-scale matrix
operations.

## Requirements

### Python (server and client)

- numpy
- pyzmq
- cupy, if GEMMs run on a GPU

### Fortran (client)

- libzmq
- f77zmq


## How to run

1. Run the GEMM server:

```bash
python ./gemmizer.py &
```

2. Run the tests

```
python ./test.py
./test_dgemm
./test_sgemm
```

3. Kill the GEMM server

```bash
fg
[Ctrl]-C
```
