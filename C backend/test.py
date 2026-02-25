import ctypes

# load dll
lib = ctypes.CDLL("./matrix.dll")


# -------- define function types --------

lib.newEmptyMatrix.argtypes = [
    ctypes.c_int,
    ctypes.c_int
]

lib.newEmptyMatrix.restype = ctypes.c_void_p


lib.Insert.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double
]


lib.Get.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int
]

lib.Get.restype = ctypes.c_double


lib.PrintM.argtypes = [
    ctypes.c_void_p
]

lib.Transpose.argtypes=[ctypes.c_void_p]
lib.Transpose.restype=ctypes.c_void_p

lib.arr_to_matrix.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # double *
    ctypes.c_int,                     # n
    ctypes.c_int,                     # rows
    ctypes.c_int                      # cols
]

lib.arr_to_matrix.restype = ctypes.c_void_p
# -------- TEST --------

# M = lib.newEmptyMatrix(3,4)

data = [0,3,12,18,1,4,14,20,2,7,16,21]

arr_type = ctypes.c_double * len(data)
arr = arr_type(*data)
M = lib.arr_to_matrix(arr, len(data), 3, 4);


lib.PrintM(M)

print("\nPython Get:",lib.Get(M,1,2))

Mt = lib.Transpose(M)

print("\nTranspose")
lib.PrintM(Mt)

