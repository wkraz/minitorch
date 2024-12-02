# MiniTorch: Building a Deep Learning Library from Scratch

Welcome to my implementation of MiniTorch, a modular deep learning library developed as part of a learning experience in machine learning systems. MiniTorch replicates core functionalities of PyTorch, focusing on foundational deep learning concepts like autodifferentiation, tensor operations, and GPU optimization.

---

## Repository Overview

This repository is divided into five modules:

1. **Module 0: Scalar Operations**
   - **Focus**: Scalar differentiation and basic mathematical operations.
   - **Key Concepts**: Numerical derivatives, chain rule, backpropagation.

2. **Module 1: Autodifferentiation**
   - **Focus**: Implementing autodifferentiation for scalar variables.
   - **Key Concepts**: Forward and backward passes, scalar functions.

3. **Module 2: Tensor Operations**
   - **Focus**: Building a tensor data structure and operations.
   - **Key Concepts**: Indexing, broadcasting, tensor reduction, and matrix multiplication.

4. **Module 3: Parallel and GPU Operations**
   - **Focus**: Speeding up computations using Numba and CUDA.
   - **Key Concepts**: Parallel processing, tensor mapping/zipping/reduction, GPU acceleration.

5. **Module 4: Convolutions and Neural Networks**
   - **Focus**: Implementing 1D/2D convolutions and pooling layers for deep learning tasks.
   - **Key Concepts**: Convolutional layers, pooling, backpropagation for CNNs.

---

## Example Commands

Run training for a neural network using the faster tensor backend:
```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Run training for a neural network using the CPU backend:
```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

## Results

### Dataset Performance

| Dataset  | Accuracy | Time per Epoch (GPU) | Time per Epoch (CPU) |
|----------|----------|-----------------------|-----------------------|
| Simple   | 98%      | 0.8 seconds          | 1.8 seconds          |
| Xor      | 95%      | 0.9 seconds          | 1.9 seconds          |
| Split    | 97%      | 0.7 seconds          | 1.7 seconds          |


## Features

### Key Highlights
- Fully custom implementation of a deep learning library.
- Optimized tensor operations using parallelism and GPU acceleration.
- Modular structure for ease of extension and experimentation.

### Supported Operations
- Scalar and tensor autodifferentiation.
- Broadcasting and advanced indexing for tensors.
- Parallelized tensor operations with Numba.
- CUDA-accelerated tensor computations.
- Support for 1D/2D convolutions and pooling.

## Installation

Clone the repository:
```bash
git clone https://github.com/wkraz/minitorch.git
cd minitorch
```
(Optional) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements.extra.txt
pip install -Ue .
```
Run tests to verify the installation:
```bash
pytest tests/
```

## Usage

### Training
Use the `project/run_fast_tensor.py` script to train models with MiniTorch. Customize the backend, hidden layers, and dataset:
```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
### Visualization
Use the Streamlit interface to visualize the computation graph and network training progress:
```bash
streamlit run project/app.py -- 1
```

## Development Notes

### Structure
The repository follows a modular structure:
- `minitorch/`: Core library code.
- `project/`: Training and visualization scripts.
- `tests/`: Unit tests for all modules.

### Contributing
Contributions are welcome! Please fork the repository, create a branch, and submit a pull request.

## License
his project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
MIT License

Copyright (c) 2024 Will Krzastek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```