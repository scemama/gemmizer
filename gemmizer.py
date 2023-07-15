#!/usr/bin/env python

import zmq
import sys
import threading

# Number of worker threads that will be started to handle incoming tasks.
Nstreams=16

# 'CPU' or 'GPU'
target="CPU"

url_client="tcp://*:5555"


# Global variables

nt = { 'N': 'F', 'T': 'C', 'n': 'F', 't': 'C' }
url_worker = "inproc://workers"

def server():
    """
This function sets up the ZeroMQ context and two types of sockets: a
ROUTER socket for communicating with clients, and a DEALER socket for
communicating with worker threads. After setting up these sockets and binding
them to appropriate endpoints, it starts the worker threads and then starts a
built-in ZeroMQ device that bridges the clients and workers sockets,
facilitating the passing of messages between them.
   """

    # Prepare our context and sockets
    context = zmq.Context(1)

    # Socket to talk to clients
    clients = context.socket(zmq.ROUTER)
    clients.bind(url_client)

    # Socket to talk to workers
    workers = context.socket(zmq.DEALER)
    workers.bind(url_worker)

    # Launch pool of worker threads
    for i in range(Nstreams):
        thread = threading.Thread(target=worker_thread, args=(context, ))
        thread.start()

    zmq.device(zmq.QUEUE, clients, workers)

    # We never get here but clean up anyhow
    clients.close()
    workers.close()
    context.term()



def worker_thread(context):
    """
Each worker thread runs this function. It sets up a REP socket connected to the
DEALER socket in the main thread. It then enters a loop where it waits for a
request from a client. These requests are expected to be commands to perform a
matrix multiplication operation with certain parameters.

The worker supports two types of multiplication, 'dgemm' and 'sgemm', for
double-precision and single-precision matrices, respectively. It also responds
to an 'exit' command, which will make it stop processing requests and
terminate.

For a multiplication request, the worker reads matrix dimensions and other
parameters, loads matrices from received byte strings, performs the requested
matrix multiplication operation, then sends the resulting matrix back to the
client as a byte string.
    """

    import numpy as np
    if target == 'GPU':
        import cupy as cp

    # Socket to talk to dispatcher
    socket = context.socket(zmq.REP)

    socket.connect(url_worker)

    while True:
        # Read the request
        message_parts = socket.recv_multipart()
        command = message_parts[0].decode('utf-8')
        print(command)
        sys.stdout.flush()

        typ = -1
        if command == 'dgemm':
            typ = np.float64
        elif command == 'sgemm':
            typ = np.float32

        elif command == 'exit':
            socket.send_string('OK')
            # Close the socket
            socket.close()
            # Terminate the context
            context.term()
            break

        else:
            socket.send_string(f'Unknown command: {command}')

        # Parse matrix dimensions
        nt1, nt2, m, n, k, alpha, beta = message_parts[1].decode('utf-8').split(',')
        nt1 = nt1.strip()
        nt2 = nt2.strip()
        m = int(m)
        n = int(n)
        k = int(k)
        alpha = float(alpha)
        beta  = float(beta)
        print(nt1,nt2,m,n,k,alpha,beta)

        # Load matrices in Fortran order (column major)
        A = np.frombuffer(message_parts[2], dtype=typ).reshape((m, k), order=nt[nt1])
        if beta != 0.:
          C = np.frombuffer(message_parts[3], dtype=typ).reshape((m, n), order='F')
          B = np.frombuffer(message_parts[4], dtype=typ).reshape((k, n), order=nt[nt2])
        else:
          B = np.frombuffer(message_parts[3], dtype=typ).reshape((k, n), order=nt[nt2])

        # Perform matrix multiplication
        if target == 'CPU':
            if beta != 0.:
              if alpha == 1.:
                  C = np.matmul(A, B) + beta * C
              else:
                  C = alpha * np.matmul(A, B) + beta * C
            else:
              if alpha == 1.:
                  C = np.matmul(A, B)
              else:
                  C = alpha * np.matmul(A, B)

        elif target == 'GPU':
            A = cp.asarray(A)
            B = cp.asarray(B)
            if beta != 0.:
              C = cp.asarray(C)
              if alpha == 1.:
                  C = cp.matmul(A, B) + beta * C
              else:
                  C = alpha * cp.matmul(A, B) + beta * C
            else:
              if alpha == 1.:
                  C = cp.matmul(A, B)
              else:
                  C = alpha * cp.matmul(A, B)
            C = cp.asnumpy(C)

        # Send result back as binary
        socket.send(C.tobytes(order='F'))
        print('ok')




if __name__ == "__main__":
    server()
