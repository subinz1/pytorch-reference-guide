"""
Sparse Tensors, Complex Numbers, and FFT
=========================================

Demonstrates:
1. COO sparse tensors: creation, operations, conversion
2. CSR sparse tensors: efficient row-based operations
3. BSR sparse tensors: block-structured sparsity
4. Complex number tensors: creation, operations
5. FFT (Fast Fourier Transform): forward, inverse, frequency analysis
"""

import torch


# ===========================================================================
# 1. COO (Coordinate) Sparse Tensors
# ===========================================================================

def demo_coo_sparse():
    """COO format: store (row, col, value) triples."""
    print("=" * 60)
    print("COO SPARSE TENSORS")
    print("=" * 60)

    # Create a 5x5 sparse matrix with 4 non-zero entries
    indices = torch.tensor([
        [0, 1, 2, 4],   # row indices
        [1, 3, 0, 4],   # col indices
    ])
    values = torch.tensor([3.0, 7.0, -1.0, 5.0])

    sparse_coo = torch.sparse_coo_tensor(indices, values, size=(5, 5))
    print(f"  COO tensor:")
    print(f"    Shape: {sparse_coo.shape}")
    print(f"    nnz:   {sparse_coo._nnz()}")
    print(f"    Dense:\n{sparse_coo.to_dense()}")

    # Arithmetic with sparse tensors
    sparse2 = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 0]]),
        torch.tensor([1.0, 2.0]),
        size=(5, 5),
    )

    # Addition
    result = sparse_coo + sparse2
    print(f"\n  Addition (COO + COO):")
    print(f"    nnz: {result._nnz()}")
    print(f"    Dense:\n{result.to_dense()}")

    # Scalar multiplication
    scaled = sparse_coo * 2.0
    print(f"\n  Scalar multiplication (COO * 2):")
    print(f"    Dense:\n{scaled.to_dense()}")

    # Sparse matrix-vector multiply
    vec = torch.randn(5)
    result = torch.mv(sparse_coo.to_dense(), vec)
    print(f"\n  Matrix-vector product: {result.shape}")

    # Convert sparse -> dense -> sparse
    dense = sparse_coo.to_dense()
    back_to_sparse = dense.to_sparse()
    print(f"\n  Round-trip (sparse -> dense -> sparse): "
          f"match = {torch.equal(sparse_coo.to_dense(), back_to_sparse.to_dense())}")


# ===========================================================================
# 2. CSR (Compressed Sparse Row) Sparse Tensors
# ===========================================================================

def demo_csr_sparse():
    """CSR format: efficient for row-based access and matrix operations."""
    print("\n" + "=" * 60)
    print("CSR SPARSE TENSORS")
    print("=" * 60)

    # CSR uses crow_indices (row pointers) and col_indices
    # For a 4x4 matrix with entries at:
    # (0,1)=3, (1,0)=7, (1,2)=5, (2,2)=1, (3,1)=4, (3,3)=2
    crow_indices = torch.tensor([0, 1, 3, 4, 6])  # cumulative nnz per row
    col_indices = torch.tensor([1, 0, 2, 2, 1, 3])
    values = torch.tensor([3.0, 7.0, 5.0, 1.0, 4.0, 2.0])

    sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(4, 4))
    print(f"  CSR tensor:")
    print(f"    Shape: {sparse_csr.shape}")
    print(f"    crow_indices: {crow_indices.tolist()}")
    print(f"    col_indices:  {col_indices.tolist()}")
    print(f"    values:       {values.tolist()}")
    print(f"    Dense:\n{sparse_csr.to_dense()}")

    # CSR matrix-matrix multiplication (sparse @ dense)
    dense_mat = torch.randn(4, 3)
    result = torch.sparse.mm(sparse_csr, dense_mat)
    print(f"\n  Sparse CSR @ Dense: {list(result.shape)}")

    # Verify against dense multiplication
    expected = sparse_csr.to_dense() @ dense_mat
    match = torch.allclose(result, expected, atol=1e-6)
    print(f"  Matches dense matmul: {match}")

    # Memory comparison
    dense_size = 4 * 4 * 4  # 4x4 float32
    sparse_size = (len(crow_indices) + len(col_indices)) * 4 + len(values) * 4
    print(f"\n  Memory comparison (4x4 matrix, 6 nonzeros):")
    print(f"    Dense:  {dense_size} bytes")
    print(f"    CSR:    {sparse_size} bytes")
    print(f"    Savings become significant for large sparse matrices")


# ===========================================================================
# 3. BSR (Block Sparse Row) Sparse Tensors
# ===========================================================================

def demo_bsr_sparse():
    """BSR format: block-structured sparsity."""
    print("\n" + "=" * 60)
    print("BSR SPARSE TENSORS")
    print("=" * 60)

    # BSR: each nonzero element is a dense block
    # 6x6 matrix with 2x2 blocks: 3x3 block grid
    # Non-zero blocks at positions (0,0) and (1,2) in the block grid
    crow_indices = torch.tensor([0, 1, 2, 2])  # 3 block-rows
    col_indices = torch.tensor([0, 2])          # block-column indices
    values = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],   # block at (0,0)
        [[5.0, 6.0], [7.0, 8.0]],   # block at (1,2)
    ])

    sparse_bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(6, 6))
    print(f"  BSR tensor:")
    print(f"    Shape:      {sparse_bsr.shape}")
    print(f"    Block size: 2x2")
    print(f"    Dense:\n{sparse_bsr.to_dense()}")

    # BSR is useful for structured pruning where entire blocks are zero
    print(f"\n  Use cases for BSR:")
    print(f"    - Structured pruning (prune entire blocks)")
    print(f"    - Block-diagonal matrices")
    print(f"    - Efficient GPU kernels (blocks align with CUDA tiles)")


# ===========================================================================
# 4. Sparsity Patterns and Density
# ===========================================================================

def demo_sparsity_patterns():
    """Create sparse tensors from common patterns."""
    print("\n" + "=" * 60)
    print("SPARSITY PATTERNS")
    print("=" * 60)

    # Identity matrix (very sparse for large n)
    n = 100
    eye_sparse = torch.eye(n).to_sparse()
    density = eye_sparse._nnz() / (n * n)
    print(f"  Identity ({n}x{n}): nnz={eye_sparse._nnz()}, density={density:.4f}")

    # Random sparse matrix
    dense = torch.randn(100, 100)
    mask = torch.rand(100, 100) < 0.05
    sparse_random = (dense * mask).to_sparse()
    density = sparse_random._nnz() / (100 * 100)
    print(f"  Random 5% ({100}x{100}): nnz={sparse_random._nnz()}, density={density:.4f}")

    # Diagonal matrix
    diag_vals = torch.randn(50)
    diag_sparse = torch.diag(diag_vals).to_sparse()
    print(f"  Diagonal ({50}x{50}): nnz={diag_sparse._nnz()}")

    # Banded matrix (tridiagonal)
    n = 10
    main = torch.randn(n)
    upper = torch.randn(n - 1)
    lower = torch.randn(n - 1)
    banded = (torch.diag(main) + torch.diag(upper, 1) + torch.diag(lower, -1))
    banded_sparse = banded.to_sparse()
    print(f"  Tridiagonal ({n}x{n}): nnz={banded_sparse._nnz()}")


# ===========================================================================
# 5. Complex Number Tensors
# ===========================================================================

def demo_complex_numbers():
    """Creating and operating on complex tensors."""
    print("\n" + "=" * 60)
    print("COMPLEX NUMBER TENSORS")
    print("=" * 60)

    # Method 1: from real and imaginary parts
    real = torch.tensor([1.0, 2.0, 3.0])
    imag = torch.tensor([4.0, 5.0, 6.0])
    z1 = torch.complex(real, imag)
    print(f"  z1 = {z1}")

    # Method 2: from Python complex literals
    z2 = torch.tensor([1+2j, 3+4j, 5+6j])
    print(f"  z2 = {z2}")

    # Properties
    print(f"\n  Properties of z1:")
    print(f"    Real part:      {z1.real}")
    print(f"    Imag part:      {z1.imag}")
    print(f"    Magnitude:      {z1.abs()}")
    print(f"    Phase (angle):  {z1.angle()}")
    print(f"    Conjugate:      {z1.conj()}")

    # Arithmetic
    print(f"\n  Arithmetic:")
    print(f"    z1 + z2 = {z1 + z2}")
    print(f"    z1 * z2 = {z1 * z2}")
    print(f"    z1 / z2 = {z1 / z2}")

    # Complex matrix operations
    A = torch.randn(3, 3, dtype=torch.cfloat)
    print(f"\n  Complex matrix A: shape={A.shape}, dtype={A.dtype}")
    print(f"    Matrix multiply: {(A @ A).shape}")

    # Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta)
    theta = torch.tensor([0.0, 3.14159/4, 3.14159/2, 3.14159])
    euler = torch.exp(1j * theta)
    print(f"\n  Euler's formula e^(i*theta):")
    for i in range(len(theta)):
        print(f"    theta={theta[i]:.3f}: "
              f"cos={euler[i].real:.3f}, sin={euler[i].imag:.3f}")


# ===========================================================================
# 6. Fast Fourier Transform (FFT)
# ===========================================================================

def demo_fft():
    """FFT operations: forward, inverse, frequency analysis."""
    print("\n" + "=" * 60)
    print("FAST FOURIER TRANSFORM (FFT)")
    print("=" * 60)

    # Create a simple signal: sum of two sinusoids
    sample_rate = 1000  # Hz
    duration = 1.0       # seconds
    n_samples = int(sample_rate * duration)
    t = torch.linspace(0, duration, n_samples)

    freq1, freq2 = 50.0, 120.0   # Hz
    signal = torch.sin(2 * 3.14159 * freq1 * t) + 0.5 * torch.sin(2 * 3.14159 * freq2 * t)
    print(f"  Signal: {n_samples} samples, {duration}s at {sample_rate}Hz")
    print(f"  Components: {freq1}Hz (amplitude 1.0) + {freq2}Hz (amplitude 0.5)")

    # Forward FFT
    spectrum = torch.fft.fft(signal)
    freqs = torch.fft.fftfreq(n_samples, d=1.0 / sample_rate)

    # Only positive frequencies (symmetric for real signals)
    pos_mask = freqs >= 0
    magnitudes = spectrum.abs()[pos_mask] / n_samples * 2  # normalize
    pos_freqs = freqs[pos_mask]

    # Find peak frequencies
    top_k = 5
    top_indices = magnitudes.topk(top_k).indices
    print(f"\n  Top {top_k} frequency components:")
    for idx in top_indices:
        f = pos_freqs[idx].item()
        m = magnitudes[idx].item()
        if m > 0.01:
            print(f"    {f:6.1f} Hz: magnitude = {m:.4f}")

    # Inverse FFT: reconstruct the signal
    reconstructed = torch.fft.ifft(spectrum).real
    error = (signal - reconstructed).abs().max().item()
    print(f"\n  Inverse FFT reconstruction error: {error:.2e}")

    # 2D FFT (for images)
    image = torch.randn(64, 64)
    spectrum_2d = torch.fft.fft2(image)
    reconstructed_2d = torch.fft.ifft2(spectrum_2d).real
    error_2d = (image - reconstructed_2d).abs().max().item()
    print(f"\n  2D FFT on 64x64 image:")
    print(f"    Spectrum shape: {list(spectrum_2d.shape)}")
    print(f"    Reconstruction error: {error_2d:.2e}")

    # rfft: optimized FFT for real-valued signals
    spectrum_r = torch.fft.rfft(signal)
    print(f"\n  Real FFT (rfft) — for real signals:")
    print(f"    Input:    {signal.shape[0]} samples")
    print(f"    Spectrum: {spectrum_r.shape[0]} complex values")
    print(f"    (rfft returns only positive frequencies, saving ~50% memory)")

    # Frequency filtering: simple low-pass filter
    spectrum_filtered = spectrum.clone()
    cutoff_freq = 80  # Hz: keep only frequencies below 80 Hz
    mask = freqs.abs() > cutoff_freq
    spectrum_filtered[mask] = 0
    filtered_signal = torch.fft.ifft(spectrum_filtered).real
    print(f"\n  Low-pass filter (cutoff={cutoff_freq}Hz):")
    print(f"    Original signal energy:  {(signal ** 2).sum().item():.2f}")
    print(f"    Filtered signal energy:  {(filtered_signal ** 2).sum().item():.2f}")
    print(f"    (120Hz component removed, 50Hz component preserved)")


if __name__ == "__main__":
    demo_coo_sparse()
    demo_csr_sparse()
    demo_bsr_sparse()
    demo_sparsity_patterns()
    demo_complex_numbers()
    demo_fft()
    print("\n" + "=" * 60)
    print("All sparse/complex/FFT demos completed successfully!")
    print("=" * 60)
