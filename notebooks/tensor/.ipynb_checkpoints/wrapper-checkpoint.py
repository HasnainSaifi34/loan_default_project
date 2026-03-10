import ctypes
import os
import numpy as np

# -------- Load Shared Library --------

LIB_PATH = "/lib/tensor/libtensor.so"

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Shared library not found at {LIB_PATH}")

_lib = ctypes.CDLL(LIB_PATH)


# -------- Define C Struct --------

class _CTensor(ctypes.Structure):
    _fields_ = [
        ("ndim", ctypes.c_int),
        ("shapes", ctypes.POINTER(ctypes.c_size_t)),
        ("strides", ctypes.POINTER(ctypes.c_size_t)),
        ("data", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
        ("using_mmap", ctypes.c_int),
        ("is_view", ctypes.c_int),
    ]


# -------- Configure Function Signatures --------

_lib.createEmptyTensor.restype = ctypes.POINTER(_CTensor)
_lib.createEmptyTensor.argtypes = [
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int
]

_lib.Transpose.restype = ctypes.POINTER(_CTensor)
_lib.Transpose.argtypes = [ctypes.POINTER(_CTensor)]

_lib.TensorAdd.restype = ctypes.POINTER(_CTensor)
_lib.TensorAdd.argtypes = [
    ctypes.POINTER(_CTensor),
    ctypes.POINTER(_CTensor)
]

_lib.freeTensor.argtypes = [ctypes.POINTER(_CTensor)]
_lib.printTensor.argtypes = [ctypes.POINTER(_CTensor)]
_lib.printTensorInfo.argtypes = [ctypes.POINTER(_CTensor)]


# -------- Python Wrapper Class --------

class Tensor:

    def __init__(self, shape):
        shape_arr = (ctypes.c_size_t * len(shape))(*shape)
        self._ptr = _lib.createEmptyTensor(shape_arr, len(shape))

        if not self._ptr:
            raise MemoryError("Failed to create tensor")

    @classmethod
    def _from_ptr(cls, ptr):
        obj = cls.__new__(cls)
        obj._ptr = ptr
        return obj

    @property
    def shape(self):
        return tuple(self._ptr.contents.shapes[i]
                     for i in range(self._ptr.contents.ndim))

    def numpy(self):
        """Zero-copy NumPy view"""
        data = np.ctypeslib.as_array(
            self._ptr.contents.data,
            shape=(self._ptr.contents.size,)
        )
        return data.reshape(self.shape)

    def print(self):
        _lib.printTensor(self._ptr)

    def info(self):
        _lib.printTensorInfo(self._ptr)

    def T(self):
        return Tensor._from_ptr(
            _lib.Transpose(self._ptr)
        )

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Addition requires Tensor")

        return Tensor._from_ptr(
            _lib.TensorAdd(self._ptr, other._ptr)
        )

    def free(self):
        if self._ptr:
            _lib.freeTensor(self._ptr)
            self._ptr = None

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.freeTensor(self._ptr)
            self._ptr = None
            
   