default: test_dgemm test_sgemm

test_dgemm: test_dgemm.f90
	ifort test_dgemm.f90 -mkl=sequential -lf77zmq -lzmq -o test_dgemm

test_sgemm: test_sgemm.f90
	ifort test_sgemm.f90 -mkl=sequential -lf77zmq -lzmq -o test_sgemm

