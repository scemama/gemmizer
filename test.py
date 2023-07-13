#!/usr/bin/env python

import unittest
import zmq
import numpy as np

class TestDgemmServer(unittest.TestCase):
    def setUp(self):
        # Prepare our context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def tearDown(self):
        # Send 'exit' command to server
#        self.socket.send_multipart([b'exit'])
#        reply = self.socket.recv()
#        self.assertEqual(reply.decode('utf-8'), 'OK')

        # Close the socket and terminate the context
        self.socket.close()
        self.context.term()

    def test_dgemmNN(self):
        # Create some non-square test matrices
        A = np.random.rand(5, 7)
        B = np.random.rand(7, 3)

        # Calculate the expected result
        expected = np.matmul(A, B)

        # Send 'dgemm' command with matrices to server
        self.socket.send_multipart([
            b'dgemm',
            f'N,N,5,3,7,1.0,0.0'.encode('utf-8'),
            A.tobytes(order='F'),
            B.tobytes(order='F')
        ])

        # Get result from server
        result_bytes = self.socket.recv()

        # Convert result to numpy array
        result = np.frombuffer(result_bytes, dtype=np.float64).reshape(expected.shape, order='F')

        # Check if the result from the server matches the expected result
        self.assertTrue(np.allclose(result, expected))

    def test_dgemmTN(self):
        # Create some non-square test matrices
        A = np.random.rand(7, 5)
        B = np.random.rand(7, 3)

        # Calculate the expected result
        expected = np.matmul(A.T, B)

        # Send 'dgemm' command with matrices to server
        self.socket.send_multipart([
            b'dgemm',
            f'T,N,5,3,7,1.0,0.0'.encode('utf-8'),
            A.tobytes(order='F'),
            B.tobytes(order='F')
        ])

        # Get result from server
        result_bytes = self.socket.recv()

        # Convert result to numpy array
        result = np.frombuffer(result_bytes, dtype=np.float64).reshape(expected.shape, order='F')

        # Check if the result from the server matches the expected result
        self.assertTrue(np.allclose(result, expected))

    def test_dgemmTT(self):
        # Create some non-square test matrices
        A = np.random.rand(7, 5)
        B = np.random.rand(3, 7)

        # Calculate the expected result
        expected = np.matmul(A.T, B.T)

        # Send 'dgemm' command with matrices to server
        self.socket.send_multipart([
            b'dgemm',
            f'T,T,5,3,7,1.0,0.0'.encode('utf-8'),
            A.tobytes(order='F'),
            B.tobytes(order='F')
        ])

        # Get result from server
        result_bytes = self.socket.recv()

        # Convert result to numpy array
        result = np.frombuffer(result_bytes, dtype=np.float64).reshape(expected.shape, order='F')

        # Check if the result from the server matches the expected result
        self.assertTrue(np.allclose(result, expected))

    def test_dgemmNT(self):
        # Create some non-square test matrices
        A = np.random.rand(5, 7)
        B = np.random.rand(3, 7)

        # Calculate the expected result
        expected = np.matmul(A, B.T)

        # Send 'dgemm' command with matrices to server
        self.socket.send_multipart([
            b'dgemm',
            f'N,T,5,3,7,1.0,0.0'.encode('utf-8'),
            A.tobytes(order='F'),
            B.tobytes(order='F')
        ])

        # Get result from server
        result_bytes = self.socket.recv()

        # Convert result to numpy array
        result = np.frombuffer(result_bytes, dtype=np.float64).reshape(expected.shape, order='F')

        # Check if the result from the server matches the expected result
        self.assertTrue(np.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()


