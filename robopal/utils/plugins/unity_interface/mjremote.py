import socket
import struct


class mjremote:
    nqpos = 0
    nmocap = 0
    ncamera = 0
    width = 0
    height = 0
    _s = None

    def _recvall(self, buffer):
        view = memoryview(buffer)
        while len(view):
            nrecv = self._s.recv_into(view)
            view = view[nrecv:]

    # result = (nqpos, nmocap, ncamra, width, height)
    def connect(self, address='127.0.0.1', port=1050):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((address, port))
        data = bytearray(20)
        self._recvall(data)
        result = struct.unpack('iiiii', data)
        self.nqpos, self.nmocap, self.ncamera, self.width, self.height = result
        return result

    def close(self):
        if self._s:
            self._s.close()
            self._s = None

    # result = (key, active, select, refpos[3], refquat[4])
    def getinput(self):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 1))
        data = bytearray(40)
        self._recvall(data)
        result = struct.unpack('iiifffffff', data)
        return result

    # buffer = bytearray(3*width*height)
    def getimage(self, buffer):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 2))
        self._recvall(buffer)

    def savesnapshot(self):
        if not self._s:
            return 'Not connected'
        self._s.send(struct.pack("i", 3))

    def savevideoframe(self):
        if not self._s:
            return 'Not connected'
        self._s.send(struct.pack("i", 4))

    def setcamera(self, index):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 5))
        self._s.sendall(struct.pack("i", index))

    # qpos = numpy.ndarray(nqpos)
    def setqpos(self, qpos):
        if not self._s:
            return 'Not connected'
        if len(qpos) != self.nqpos:
            return 'qpos has wrong size'
        fqpos = qpos.astype('float32')
        self._s.sendall(struct.pack("i", 6))
        self._s.sendall(fqpos.tobytes())

    # pos = numpy.ndarray(3*nmocap), quat = numpy.ndarray(4*nmocap)
    def setmocap(self, pos, quat):
        if not self._s:
            return 'Not connected'
        if len(pos) != 3 * self.nmocap:
            return 'pos has wrong size'
        if len(quat) != 4 * self.nmocap:
            return 'quat has wrong size'
        fpos = pos.astype('float32')
        fquat = quat.astype('float32')
        self._s.sendall(struct.pack("i", 7))
        self._s.sendall(fpos.tobytes())
        self._s.sendall(fquat.tobytes())

    def sendForce(self, force):
        if not self._s:
            return 'Not connected'
        if len(force) != 3:
            return 'qpos has wrong size'
        fforce = force.astype('float32')
        self._s.sendall(struct.pack("i", 8))
        self._s.sendall(fforce.tobytes())

