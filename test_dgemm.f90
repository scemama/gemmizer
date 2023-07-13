program test
  implicit none
  include 'f77_zmq_free.h'

  double precision, allocatable, dimension (:,:) :: A, B, C, Cref

  integer :: rc
  integer(ZMQ_PTR) :: context
  integer(ZMQ_PTR) :: socket
  character*(128) :: args

  allocate( A(5,7), B(3,7), C(5,3), Cref(5,3) )

  A = reshape( (/ 0.25411728, 0.33372948, 0.79007076, 0.45648902, 0.51268183, &
       0.86442056, 0.218108, 0.19854651, 0.31927935, 0.29092046, 0.18344948, &
       0.61151984, 0.21323907, 0.08928674, 0.51866581, 0.48986397, 0.99813841, &
       0.25994163, 0.24655499, 0.2326665 , 0.92772686, 0.03259491, 0.19786076, &
       0.63578155, 0.74667576, 0.30435633, 0.5901392 , 0.83344062, 0.66099918, &
       0.65824617, 0.88155817, 0.86330696, 0.1874176 , 0.90796444, 0.75899183 /), &
       (/ 5,7 /) )

  B = reshape( (/ 0.52355632, 0.2751289 , 0.21737057, 0.32139644, 0.02008003, &
       0.77215043, 0.77854062, 0.86229312, 0.94616623, 0.83739632, 0.55079313, &
       0.30212382, 0.69834079, 0.09453764, 0.52286556, 0.90709716, 0.62643434, &
       0.18490888, 0.58756551, 0.20808955, 0.77641335 /), (/ 3,7 /) )

  call dgemm('N','T', 5, 3, 7, 1.d0, A, 5, B, 3, 0.d0, Cref, 5)

  context = f77_zmq_ctx_new()
  socket  = f77_zmq_socket(context, ZMQ_REQ)
  rc      = f77_zmq_connect(socket, 'tcp://localhost:5555')

  rc = f77_zmq_send(socket, 'dgemm', 5, ZMQ_SNDMORE)
  args = 'N,T,5,3,7,1.0,0.0'
  rc = f77_zmq_send(socket, args, len(trim(args)),  ZMQ_SNDMORE)
  rc = f77_zmq_send8(socket, A, 8_8*size(A,1)*size(A,2), ZMQ_SNDMORE)
  rc = f77_zmq_send8(socket, B, 8_8*size(B,1)*size(B,2), 0)
  rc = f77_zmq_recv8(socket, C, 8_8*size(C,1)*size(C,2), 0)

  print *, C
  print *, ''
  print *, Cref
  print *, ''
  print *, C-Cref

  rc = f77_zmq_close(socket)
  rc = f77_zmq_ctx_destroy(context)


end
