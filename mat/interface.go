package mat

import "io"

type Mat struct {
	Buf  []float64
	Size []int64
}

// Operator is the matrix operator that works on a session which
// needs to be closed.
type Operator interface {
	io.Closer

	// New matrix from buffer.
	New(buf []float64, size ...int) (*Mat, error)
	// Zeros is a new matrix with all zeros.
	Zeros(size ...int) (*Mat, error)
	// Ones is a new matrix with all ones.
	Ones(size ...int) (*Mat, error)
	// Rand is a new matrix with random numbers with uniform distribution.
	Rand(size ...int) (*Mat, error)
	// Randn is a new matrix with random numbers with normal distribution.
	Randn(size ...int) (*Mat, error)

	// Reshape reshapes a given matrix to new size.
	Reshape(mat *Mat, size ...int) (*Mat, error)
	// Slice can extract a slice from a matrix.
	Slice(mat *Mat, begin []int, size ...int) (*Mat, error)
	// Repmat repeats a matrix as many times along dimensions as requested.
	Repmat(mat *Mat, dim ...int) (*Mat, error)

	// Mul multiples two matrices.
	Mul(x, y *Mat) (*Mat, error)
	// Inv takes inverse of a matrix.
	Inv(mat *Mat) (*Mat, error)

	// Print will pretty print input matrix.
	Print(w io.Writer, mat *Mat) error
}
