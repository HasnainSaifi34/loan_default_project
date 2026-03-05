# Tensor Engine in C — Reference Documentation

> A hand-written tensor library in C that supports creation, memory management, views, and arithmetic operations on multi-dimensional arrays.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Structure — `Tensor`](#data-structure--tensor)
3. [Key Concepts](#key-concepts)
   - [Strides](#strides)
   - [Contiguous vs Non-Contiguous Memory](#contiguous-vs-non-contiguous-memory)
   - [Views (Zero-Copy Operations)](#views-zero-copy-operations)
   - [Reference Counting](#reference-counting)
   - [Memory Allocation Strategy (calloc vs mmap)](#memory-allocation-strategy-calloc-vs-mmap)
4. [API Reference](#api-reference)
   - [createEmptyTensor](#createemptytensor)
   - [freeTensor](#freetensor)
   - [printTensorInfo](#printtensorinfo)
   - [printTensor](#printtensor)
   - [isNull](#isnull)
   - [isContiguous](#iscontiguous)
   - [getOffset](#getoffset)
   - [flat_to_multi](#flat_to_multi)
   - [Transpose](#transpose)
   - [reshape](#reshape)
   - [TensorAdd](#tensoradd)
   - [TensorSub](#tensorsub)
   - [TensorMul](#tensormul)
   - [matMul](#matmul)
5. [Memory Layout Diagrams](#memory-layout-diagrams)
6. [Error Handling Philosophy](#error-handling-philosophy)
7. [Known Limitations](#known-limitations)

---

## Overview

This library implements a basic **tensor engine** in pure C. A tensor is a generalisation of scalars, vectors, and matrices to N dimensions. This engine:

- Allocates and manages multi-dimensional array data in flat (1D) memory
- Uses **stride-based indexing** to interpret that flat memory as N-dimensional shapes
- Supports **zero-copy views** via `Transpose` and `reshape` — operations that reinterpret memory without copying it
- Tracks shared memory ownership using **reference counting**
- Chooses between `calloc` and `mmap` for data allocation based on tensor size

---

## Data Structure — `Tensor`

```c
typedef struct Tensor {
    int ndim;          // Number of dimensions (e.g., 2 for a matrix)
    size_t *shapes;    // Array of length ndim: size of each dimension
    size_t *strides;   // Array of length ndim: step size in flat memory for each dimension
    double *data;      // Pointer to the flat array of doubles (the actual values)
    size_t size;       // Total number of elements (product of all shapes)
    int using_mmap;    // 1 if data was allocated with mmap, 0 if calloc
    int *ref_count;    // Heap-allocated integer: how many Tensor structs share this data
} Tensor;
```

### Field-by-field explanation

| Field | Type | Purpose |
|---|---|---|
| `ndim` | `int` | How many dimensions this tensor has. 1 = vector, 2 = matrix, etc. |
| `shapes` | `size_t *` | The size of each dimension. A (3×4) matrix has `shapes = {3, 4}`. |
| `strides` | `size_t *` | How many elements to skip in `data` to move one step along each dimension. |
| `data` | `double *` | The raw flat memory holding all values as 64-bit floats. |
| `size` | `size_t` | Total element count = product of all `shapes`. |
| `using_mmap` | `int` | Tracks which allocator was used so `freeTensor` can use the right deallocator. |
| `ref_count` | `int *` | Shared pointer among original tensor and all its views; frees data only when it reaches 0. |

---

## Key Concepts

### Strides

Strides are the heart of how this engine works. Instead of using nested arrays, all tensor data lives in a **single flat `double` array**. Strides tell you how many elements to skip in that flat array to move one step in a given dimension.

**How strides are calculated** (row-major / C-order):

```
strides[N-1] = 1
strides[i]   = strides[i+1] * shapes[i+1]
```

**Example — a (3 × 4) matrix:**

```
shapes  = {3, 4}
strides = {4, 1}
```

To access element `[r][c]`:
```
offset = r * strides[0] + c * strides[1]
       = r * 4          + c * 1
```

This maps the 2D logical index into a single position in the flat array.

---

### Contiguous vs Non-Contiguous Memory

A tensor is **contiguous** when its elements are laid out in memory in the exact order you'd expect from its shape — no gaps, no reversals.

The `isContiguous` function verifies this by walking backwards through dimensions and checking that each stride equals the expected step:

```
expected = 1
for i from ndim-1 down to 0:
    if strides[i] != expected → NOT contiguous
    expected *= shapes[i]
```

**Why does this matter?**  
After a `Transpose`, the strides are reversed. The data is still in original order in memory, but the logical layout is different. Such a tensor is **non-contiguous**. Arithmetic operations handle both cases.

---

### Views (Zero-Copy Operations)

`Transpose` and `reshape` both return a **view** — a new `Tensor` struct that points to the **same `data` array** as the original. No data is copied.

This is efficient but means:
- Modifying data through a view modifies the original too.
- The `ref_count` is incremented so the data isn't freed while any view is alive.
- `shapes` and `strides` are freshly allocated per view (they are NOT shared).

---

### Reference Counting

`ref_count` is a **heap-allocated integer shared among an original tensor and all its views**.

```
Original tensor:  ref_count → [1]
After Transpose:  ref_count → [2]   (both point to the same int)
After freeing original: ref_count → [1]   (data stays alive)
After freeing transpose: ref_count → [0]  (data is freed now)
```

`freeTensor` decrements `*ref_count`. It only frees the `data` (and the `ref_count` allocation itself) when the count reaches zero. It always frees `shapes`, `strides`, and the `Tensor` struct itself, because those are unique per tensor.

---

### Memory Allocation Strategy (calloc vs mmap)

When creating a tensor, the engine checks whether the required memory exceeds the system page size (typically 4 KB):

```c
size_t page_size = sysconf(_SC_PAGESIZE);

if (memsize < page_size) {
    data = calloc(size, sizeof(double));   // small tensor → heap
} else {
    data = mmap(..., MAP_PRIVATE | MAP_ANONYMOUS, ...);  // large tensor → virtual memory
}
```

| Strategy | When used | Freed with |
|---|---|---|
| `calloc` | `memsize < page_size` | `free(data)` |
| `mmap` | `memsize >= page_size` | `munmap(data, memsize)` |

`calloc` initialises memory to zero automatically.  
`mmap` with `MAP_ANONYMOUS` also provides zero-initialised pages from the OS.  
`using_mmap` flag ensures `freeTensor` calls the correct deallocator.

---

## API Reference

### `createEmptyTensor`

```c
Tensor *createEmptyTensor(size_t *shape, int N);
```

**Purpose:** Allocates and initialises a new tensor with all elements set to zero.

**Parameters:**
- `shape` — array of `N` dimension sizes (e.g., `{3, 4}` for a 3×4 matrix)
- `N` — number of dimensions

**Returns:** Pointer to a new `Tensor`, or `NULL` on any allocation failure.

**What it does internally:**
1. Validates `N > 0`
2. Allocates `shapes` and `strides` arrays
3. Computes strides in row-major order
4. Computes total `size` (product of shapes)
5. Chooses `calloc` or `mmap` based on `memsize` vs page size
6. Allocates `ref_count` and sets it to `1`
7. Assembles and returns the `Tensor` struct

**Cleanup on failure:** Every allocation failure path frees all previously allocated memory before returning `NULL` — no leaks.

---

### `freeTensor`

```c
void freeTensor(Tensor *T);
```

**Purpose:** Releases all memory owned by a tensor, respecting reference counting for shared data.

**Behaviour:**
- Decrements `*ref_count`
- If `ref_count` hits `0`: frees `data` (via `free` or `munmap`) and `ref_count` itself
- Always frees `shapes`, `strides`, and the `Tensor` struct

**Safe to call with `NULL`** — exits immediately if `T` is `NULL`.

---

### `printTensorInfo`

```c
void printTensorInfo(Tensor *T);
```

**Purpose:** Prints metadata about the tensor to stdout. Useful for debugging.

**Output includes:**
- `ndim`, `size`, `using_mmap`, `ref_count`
- Full `shapes` array
- Full `strides` array

---

### `printTensor`

```c
void printTensor(Tensor *T);
```

**Purpose:** Prints the actual values of the tensor.

**Supports:**
- **1D tensors** — prints as `[ v0 v1 v2 ... ]`
- **2D tensors** — prints as a grid of rows and columns

Uses stride-based offset calculation so it works correctly on **non-contiguous tensors** (e.g., after a transpose).

**Limitation:** Prints an error message for tensors with `ndim > 2`.

---

### `isNull`

```c
int isNull(Tensor *T);
```

**Returns:** `1` if `T` is `NULL`, `0` otherwise. Used as a guard in arithmetic functions.

---

### `isContiguous`

```c
int isContiguous(Tensor *T);
```

**Returns:** `1` if the tensor's strides match row-major contiguous layout, `0` otherwise.

**Algorithm:**
```
expected = 1
for i = ndim-1 down to 0:
    if strides[i] != expected → return 0
    expected *= shapes[i]
return 1
```

---

### `getOffset`

```c
size_t getOffset(size_t *strides, size_t *indices, int ndim);
```

**Purpose:** Converts a multi-dimensional index into a flat memory offset.

```
offset = indices[0]*strides[0] + indices[1]*strides[1] + ... + indices[ndim-1]*strides[ndim-1]
```

**Example:** For a (3×4) tensor, `indices = {1, 2}` → `offset = 1*4 + 2*1 = 6`

---

### `flat_to_multi`

```c
void flat_to_multi(size_t flat_index, size_t *shape, int ndim, size_t *index);
```

**Purpose:** Converts a flat (linear) index back into a multi-dimensional index array. This is the inverse of `getOffset` for contiguous layout.

**Algorithm** (right-to-left modulo decomposition):
```
remaining = flat_index
for k = ndim-1 down to 0:
    index[k] = remaining % shape[k]
    remaining /= shape[k]
```

**Example:** flat index `6` in shape `{3, 4}` → `index = {1, 2}`

**Used by:** `TensorAdd`, `TensorSub`, `TensorMul` when operating on non-contiguous tensors.

---

### `Transpose`

```c
Tensor *Transpose(Tensor *A);
```

**Purpose:** Returns a **zero-copy view** of `A` with dimensions reversed (i.e., a transposed matrix).

**Limitation:** Only works for tensors with `ndim <= 2`. Returns `NULL` for higher dimensions.

**How it works:**  
Creates a new `Tensor` struct with `shapes` and `strides` reversed compared to the original. The `data` pointer is shared (not copied), and `ref_count` is incremented.

```
Original (3×4):  shapes={3,4}  strides={4,1}
Transposed (4×3): shapes={4,3}  strides={1,4}
```

No memory is copied. Accessing `T[i][j]` on the transpose computes the correct offset into the original data via the reversed strides.

---

### `reshape`

```c
Tensor *reshape(Tensor *T, size_t *new_shapes, int new_ndim);
```

**Purpose:** Returns a **zero-copy view** of `T` with a new shape, as long as the total element count is unchanged.

**Preconditions:**
- `T` must not be `NULL`
- `T` must be **contiguous** (non-contiguous tensors cannot be reshaped as views)
- `product(new_shapes) == T->size`

**Returns:** `NULL` if any precondition fails, otherwise a new `Tensor` view sharing the same data.

**How it works:**  
Allocates new `shapes` and `strides` arrays (computed fresh in row-major order), sets `data = T->data`, increments `ref_count`.

---

### `TensorAdd`

```c
Tensor *TensorAdd(Tensor *T1, Tensor *T2);
```

**Purpose:** Element-wise addition of two tensors. Returns a new tensor.

**Preconditions:** Both tensors must have equal `ndim`, `size`, and matching `shapes` in every dimension.

**Fast path (both contiguous):**
```c
res->data[i] = T1->data[i] + T2->data[i];
```

**General path (either non-contiguous):**  
Uses `flat_to_multi` + `getOffset` to resolve correct physical offsets before adding.

---

### `TensorSub`

```c
Tensor *TensorSub(Tensor *T1, Tensor *T2);
```

**Purpose:** Element-wise subtraction (`T1 - T2`). Same logic and preconditions as `TensorAdd`.

---

### `TensorMul`

```c
Tensor *TensorMul(Tensor *T1, Tensor *T2);
```

**Purpose:** Element-wise multiplication (Hadamard product, not matrix multiply). Same logic and preconditions as `TensorAdd`.

> **Note:** This is NOT matrix multiplication. Use `matMul` for that.

---

### `matMul`

```c
Tensor *matMul(Tensor *A, Tensor *B);
```

**Purpose:** Standard matrix multiplication of two 2D tensors. Returns a new tensor of shape `(A.rows × B.cols)`.

**Preconditions:**
- Both `A` and `B` must have `ndim == 2`
- `A->shapes[1]` must equal `B->shapes[0]` (inner dimensions must match)

**Algorithm** (naive triple-loop):
```
for r in 0..Row_A:
    for c in 0..Col_B:
        sum = 0
        for k in 0..Inner:
            sum += A[r,k] * B[k,c]
        C[r,c] = sum
```

Uses stride-based offsets directly, so it works correctly even when `A` or `B` are transposed views.

```c
A_offset = A->strides[0]*r + A->strides[1]*k;
B_offset = B->strides[0]*k + B->strides[1]*c;
```

---

## Memory Layout Diagrams

### A (2 × 3) tensor in flat memory

```
shapes  = {2, 3}
strides = {3, 1}

Flat data: [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ]
              ^                   ^
            [0,0]               [1,0]

Logical view:
  [0,0]=1  [0,1]=2  [0,2]=3
  [1,0]=4  [1,1]=5  [1,2]=6

offset[0,1] = 0*3 + 1*1 = 1  → data[1] = 2.0 ✓
offset[1,2] = 1*3 + 2*1 = 5  → data[5] = 6.0 ✓
```

### After `Transpose` — same data, reversed strides

```
Transposed shapes  = {3, 2}
Transposed strides = {1, 3}

Flat data (unchanged): [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ]

Logical view of transpose:
  [0,0]=1  [0,1]=4
  [1,0]=2  [1,1]=5
  [2,0]=3  [2,1]=6

offset[1,1] = 1*1 + 1*3 = 4  → data[4] = 5.0 ✓
```

### Reference count during view lifetime

```
createEmptyTensor(...)   →  data=0xABCD, ref_count=[1]
Tensor *T = ...              T->ref_count → [1]

Tensor *Tv = Transpose(T)    T->ref_count → [2]
                             Tv->ref_count → [2]  (same pointer)

freeTensor(T)                ref_count → [1]  (data NOT freed)
freeTensor(Tv)               ref_count → [0]  (data IS freed now)
```

---

## Error Handling Philosophy

Every function that allocates memory follows a strict **cleanup-on-failure** pattern:

1. Attempt allocation
2. If it fails, free everything allocated so far
3. Return `NULL`

This ensures there are no memory leaks on partial construction. Example from `createEmptyTensor`:

```c
Tensor *T = malloc(sizeof(Tensor));
if (!T) {
    free(strides);   // free earlier allocations
    free(shapes);
    return NULL;
}
```

All public functions accept `NULL` inputs gracefully — they either return `NULL` or return immediately.

---

## Known Limitations

| Limitation | Detail |
|---|---|
| `Transpose` only supports `ndim <= 2` | Higher-dimensional transposes (axis permutation) are not yet implemented |
| `printTensor` only supports 1D and 2D | No recursive printing for N-dim tensors |
| `reshape` requires contiguous input | Cannot reshape a transposed view directly — must materialise it first |
| `matMul` only supports 2D tensors | Batched matrix multiplication (3D+) is not supported |
| Element-wise ops require identical shapes | No broadcasting (e.g., adding a vector to each row of a matrix) |
| `flat_to_multi` uses a VLA (`size_t index[T1->ndim]`) | Variable-length arrays are stack-allocated; very large `ndim` could cause stack overflow |
| Single file, no header | There is no `.h` file; all types and functions are defined in `tensor.c` |
