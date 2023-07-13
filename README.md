# gemmizer
Python script that performs GEMM on GPUs taking matrices from a zeromq socket

## Requirements

- numpy
- cupy, if GEMMs run on a GPU


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
fg ; [Ctrl]-C
```
