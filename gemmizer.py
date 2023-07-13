#!/usr/bin/env python

import zmq
import sys


def server(target):
    import numpy as np
    if target == 'GPU':
        import cupy as cp

    # Prepare our context and sockets
    context = zmq.Context()

    #  Socket to reply to client requests
    socket = context.socket(zmq.REP)
#   socket.bind("ipc:///tmp/gemmizer_socket") # replace this with your socket address
    socket.bind("tcp://*:5555")

    # Handle transposes by setting Fortran or C format
    nt = { 'N': 'F', 'T': 'C', 'n': 'F', 't': 'C' }

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
    server('CPU')
