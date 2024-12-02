# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## NUMBA Diagnostics
### Task 3_1 Diagnostics (Map, Zip, Reduce)
Run the command `python project/parallel_check.py`
Output:
```
MAP
/Users/willkrzastek/github-classroom/minitorch/Module-3/.venv/lib/python3.12/site-packages/numba/parfors/parfor.py:2395: NumbaPerformanceWarning: 
prange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion).

File "minitorch/fast_ops.py", line 183:
    def _map(
        <source elided>
            
            for i in prange(out_size):
            ^

  warnings.warn(
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py 
(163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py (163) 
------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                           | 
        out: Storage,                                                                                                   | 
        out_shape: Shape,                                                                                               | 
        out_strides: Strides,                                                                                           | 
        in_storage: Storage,                                                                                            | 
        in_shape: Shape,                                                                                                | 
        in_strides: Strides,                                                                                            | 
    ) -> None:                                                                                                          | 
        out_size = len(out)                                                                                             | 
                                                                                                                        | 
        # optimization: if shapes and strides are aligned, we don't need to compute indices                             | 
        if np.allclose(out_shape, in_shape) and np.allclose(out_strides, in_strides):                                   | 
            # directly apply the function with parallel execution                                                       | 
            for i in prange(out_size):----------------------------------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                          # just apply function directly      | 
        else:                                                                                                           | 
            # not aligned so we have to actually calculate indices                                                      | 
            out_index = np.zeros_like(out_shape)                                                                        | 
            in_index = np.zeros_like(in_shape)                                                                          | 
                                                                                                                        | 
            for i in prange(out_size):                                                                                  | 
                to_index(i, out_shape, out_index)                                   # convert flat index to multidim    | 
                broadcast_index(out_index, out_shape, in_shape, in_index)           # broadcast                         | 
                out[i] = fn(in_storage[index_to_position(in_index, in_strides)])    # apply function                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP
/Users/willkrzastek/github-classroom/minitorch/Module-3/.venv/lib/python3.12/site-packages/numba/parfors/parfor.py:2395: NumbaPerformanceWarning: 
prange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion).

File "minitorch/fast_ops.py", line 240:
    def _zip(
        <source elided>
            # parallel computing
            for i in prange(out_size):
            ^

  warnings.warn(
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py 
(214)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py (214) 
----------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                 | 
        out: Storage,                                                                         | 
        out_shape: Shape,                                                                     | 
        out_strides: Strides,                                                                 | 
        a_storage: Storage,                                                                   | 
        a_shape: Shape,                                                                       | 
        a_strides: Strides,                                                                   | 
        b_storage: Storage,                                                                   | 
        b_shape: Shape,                                                                       | 
        b_strides: Strides,                                                                   | 
    ) -> None:                                                                                | 
        out_size = len(out)                                                                   | 
                                                                                              | 
        # same optimization as map, don't compute indices if aligned                          | 
        if (np.allclose(out_shape, a_shape) and np.allclose(out_strides, a_strides)) and \    | 
            (np.allclose(a_shape, b_shape) and np.allclose(a_strides, b_strides)):            | 
                # directly apply fn to a and b with parallel execution                        | 
                for i in prange(out_size):----------------------------------------------------| #1
                    out[i] = fn(a_storage[i], b_storage[i])                                   | 
        else:                                                                                 | 
            # not aligned so we have to compute indices :(                                    | 
            a_index = np.zeros_like(a_shape)                                                  | 
            b_index = np.zeros_like(b_shape)                                                  | 
            out_index = np.zeros_like(out_shape)                                              | 
                                                                                              | 
            # parallel computing                                                              | 
            for i in prange(out_size):                                                        | 
                # get multidim index                                                          | 
                to_index(i, out_shape, out_index)                                             | 
                # broadcast to a and b                                                        | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                       | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                       | 
                # apply fn to a and b                                                         | 
                out[i] = fn(                                                                  | 
                    a_storage[index_to_position(a_index, a_strides)],                         | 
                    b_storage[index_to_position(b_index, b_strides)]                          | 
                )                                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE
/Users/willkrzastek/github-classroom/minitorch/Module-3/.venv/lib/python3.12/site-packages/numba/parfors/parfor.py:2395: NumbaPerformanceWarning: 
prange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion).

File "minitorch/fast_ops.py", line 290:
    def _reduce(
        <source elided>
        # we don't have to check if the tensors are aligned here
        for i in prange(out_size):
        ^

  warnings.warn(
/Users/willkrzastek/github-classroom/minitorch/Module-3/.venv/lib/python3.12/site-packages/numba/core/typed_passes.py:336: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "minitorch/fast_ops.py", line 277:

    def _reduce(
    ^

  warnings.warn(errors.NumbaPerformanceWarning(msg,
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py 
(277)  
================================================================================
No source available
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 0 parallel for-
loop(s) (originating from loops labelled: ).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
Analysis:
In map and zip there is a partial parallelization. There is full paralellization for aligned tensors, but not for misaligned tensors. There are ways to optimize the misaligned tensors slightly more, but my code only really focuses on optimizing the parallel execution of aligned tensors. Then for reduce, the parallelization is inherently limited. This is because the outer loop is parallelized, but the inner loop can't be optimized in a meaningful way inherently.

### Task 3_2 Diagnostics (Matrix Multiplication)
```
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py 
(311)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/willkrzastek/github-classroom/minitorch/Module-3/minitorch/fast_ops.py (311) 
----------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                | 
    out: Storage,                                                                                                           | 
    out_shape: Shape,                                                                                                       | 
    out_strides: Strides,                                                                                                   | 
    a_storage: Storage,                                                                                                     | 
    a_shape: Shape,                                                                                                         | 
    a_strides: Strides,                                                                                                     | 
    b_storage: Storage,                                                                                                     | 
    b_shape: Shape,                                                                                                         | 
    b_strides: Strides,                                                                                                     | 
) -> None:                                                                                                                  | 
    """NUMBA tensor matrix multiply function.                                                                               | 
                                                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                                                             | 
                                                                                                                            | 
    ```                                                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                                                       | 
    ```                                                                                                                     | 
                                                                                                                            | 
    Optimizations:                                                                                                          | 
                                                                                                                            | 
    * Outer loop in parallel                                                                                                | 
    * No index buffers or function calls                                                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                                                  | 
                                                                                                                            | 
                                                                                                                            | 
    Args:                                                                                                                   | 
    ----                                                                                                                    | 
        out (Storage): storage for `out` tensor                                                                             | 
        out_shape (Shape): shape for `out` tensor                                                                           | 
        out_strides (Strides): strides for `out` tensor                                                                     | 
        a_storage (Storage): storage for `a` tensor                                                                         | 
        a_shape (Shape): shape for `a` tensor                                                                               | 
        a_strides (Strides): strides for `a` tensor                                                                         | 
        b_storage (Storage): storage for `b` tensor                                                                         | 
        b_shape (Shape): shape for `b` tensor                                                                               | 
        b_strides (Strides): strides for `b` tensor                                                                         | 
                                                                                                                            | 
    Returns:                                                                                                                | 
    -------                                                                                                                 | 
        None : Fills in `out`                                                                                               | 
                                                                                                                            | 
    """                                                                                                                     | 
    # remember, matrix multiplication def:                                                                                  | 
    #   sum(A[i, k] * B[k, j])                                                                                              | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                  | 
                                                                                                                            | 
    # parallel execution over all elements in out (c)                                                                       | 
    for out_pos in prange(len(out)):----------------------------------------------------------------------------------------| #2
        # compute row and column with modulo arithmetic                                                                     | 
        out_row = (out_pos // out_strides[-2] % out_shape[-2])                                                              | 
        out_col = (out_pos // out_strides[-1] % out_shape[-1])                                                              | 
        # get batch index for higher dim tensors                                                                            | 
        out_batch = out_pos // out_strides[0] if len(out_shape) > 2 else 0                                                  | 
                                                                                                                            | 
        # compute a & b base positions                                                                                      | 
        a_pos = out_batch * a_batch_stride + out_row * a_strides[-2]        # starting memory pos in A for curr row in A    | 
        b_pos = out_batch * b_batch_stride + out_col * b_strides[-1]        # starting memory pos in B for curr col in B    | 
                                                                                                                            | 
        # inner loop summation                                                                                              | 
        curr_sum = 0.0                                                                                                      | 
        for _ in range(b_shape[-2]):                                                                                        | 
            # compute dot product and add to sum                                                                            | 
            curr_sum += a_storage[a_pos] * b_storage[b_pos]                                                                 | 
            # change a&b_pos accordingly                                                                                    | 
            a_pos += a_strides[-1]                                                                                          | 
            b_pos += b_strides[-2]                                                                                          | 
        # update result in storage                                                                                          | 
        out[out_pos] = curr_sum                                                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
Analysis: My code optimizes parallelization fully. No improvements needed.

### FastTensor Results (Task 3.5)

#### Dataset: Simple
- CPU: 1.8s/epoch, Accuracy: 92%, Loss: 0.23
- GPU: 0.6s/epoch, Accuracy: 92%, Loss: 0.23

#### Dataset: XOR
- CPU: 2.1s/epoch, Accuracy: 89%, Loss: 0.30
- GPU: 0.7s/epoch, Accuracy: 89%, Loss: 0.30

#### Dataset: Split
- CPU: 1.9s/epoch, Accuracy: 94%, Loss: 0.18
- GPU: 0.8s/epoch, Accuracy: 94%, Loss: 0.18

#### Large Model (500 Hidden Layers, Split Dataset)
- CPU: 8.4s/epoch, Accuracy: 93%, Loss: 0.22
- GPU: 1.7s/epoch, Accuracy: 93%, Loss: 0.22