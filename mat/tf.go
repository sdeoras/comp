package mat

import (
	"fmt"

	"github.com/sdeoras/comp/common"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Op is the operator for TF interface.
type Op struct {
	// It embeds common op
	common.Op
}

// NewOperator provides an instance of new operator for
// cloud file I/O
func NewOperator(options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitUsingB64Graph(op, modelPB, options); err != nil {
		return nil, err
	}
	return op, nil
}

// New creates a new matrix from a given slice.
func (op *Op) New(buf []float64, size ...int) (*Mat, error) {
	mat := new(Mat)
	mat.Buf = buf
	mat.Size = make([]int64, len(size))
	numel := int64(1)
	for i, v := range size {
		mat.Size[i] = int64(v)
		numel *= mat.Size[i]
	}

	if numel != int64(len(buf)) {
		return nil, fmt.Errorf("buffer len and size are incompatible")
	}

	return mat, nil
}

// Slice inverts a matrix assuming it is invertible.
func (op *Op) Slice(mat *Mat, begin []int, size ...int) (*Mat, error) {
	bufT, err := tf.NewTensor(mat.Buf)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor(mat.Size)
	if err != nil {
		return nil, err
	}

	b := make([]int64, len(begin))
	for i, v := range begin {
		b[i] = int64(v)
	}
	beginT, err := tf.NewTensor(b)
	if err != nil {
		return nil, err
	}

	s := make([]int64, len(size))
	for i, v := range size {
		s[i] = int64(v)
	}
	size2T, err := tf.NewTensor(s)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT
	feeds[op.GetGraph().Operation(inputShapeB).Output(0)] = beginT
	feeds[op.GetGraph().Operation(inputShapeS).Output(0)] = size2T

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(sliceOp).Output(0),
			op.GetGraph().Operation(sliceShapeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid output length. expected 2, got: %d", len(out))
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	outputSize, ok := out[1].Value().([]int64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []int64, got:%T", out[1].Value())
	}

	slicedMat := new(Mat)
	slicedMat.Buf = output
	slicedMat.Size = outputSize

	return slicedMat, nil
}

// Reshape reshapes the input matrix with new size.
func (op *Op) Reshape(mat *Mat, size ...int) (*Mat, error) {
	p := int64(1)
	for _, v := range size {
		p *= int64(v)
	}

	if mat.Numel() != p {
		return nil, fmt.Errorf("reshape should not change the number of elemements")
	}

	bufT, err := tf.NewTensor(mat.Buf)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor(mat.Size)
	if err != nil {
		return nil, err
	}

	s := make([]int64, len(size))
	for i, v := range size {
		s[i] = int64(v)
	}
	size2T, err := tf.NewTensor(s)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT
	feeds[op.GetGraph().Operation(inputShape2).Output(0)] = size2T

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(reshapeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 1 {
		return nil, fmt.Errorf("invalid output length. expected 1, got: %d", len(out))
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	reshapedMat := new(Mat)
	reshapedMat.Buf = output
	reshapedMat.Size = s

	return reshapedMat, nil
}

// Repmat repeats input matrix multiple times as specified in dim array
func (op *Op) Repmat(mat *Mat, dim ...int) (*Mat, error) {
	bufT, err := tf.NewTensor(mat.Buf)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor(mat.Size)
	if err != nil {
		return nil, err
	}

	s := make([]int64, len(dim))
	for i, v := range dim {
		s[i] = int64(v)
	}
	size2T, err := tf.NewTensor(s)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT
	feeds[op.GetGraph().Operation(inputShape2).Output(0)] = size2T

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(tileOp).Output(0),
			op.GetGraph().Operation(tileShapeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid output length. expected 2, got: %d", len(out))
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	shape, ok := out[1].Value().([]int64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []int64, got:%T", out[1].Value())
	}

	tiledMat := new(Mat)
	tiledMat.Buf = output
	tiledMat.Size = shape

	return tiledMat, nil
}

// Mul multiples two matrices.
func (op *Op) Mul(x, y *Mat) (*Mat, error) {
	bufT, err := tf.NewTensor(x.Buf)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor(x.Size)
	if err != nil {
		return nil, err
	}

	bufT2, err := tf.NewTensor(y.Buf)
	if err != nil {
		return nil, err
	}

	sizeT2, err := tf.NewTensor(y.Size)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT
	feeds[op.GetGraph().Operation(inputBuffer2).Output(0)] = bufT2
	feeds[op.GetGraph().Operation(inputShape2).Output(0)] = sizeT2

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(mulOp).Output(0),
			op.GetGraph().Operation(mulShapeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid output length, expected 2, got %d", len(out))
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	shape, ok := out[1].Value().([]int64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []int64, got:%T", out[1].Value())
	}

	invMat := new(Mat)
	invMat.Buf = output
	invMat.Size = shape

	return invMat, nil
}

// Inv inverts a matrix assuming it is invertible.
func (op *Op) Inv(mat *Mat) (*Mat, error) {
	bufT, err := tf.NewTensor(mat.Buf)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor(mat.Size)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(invOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	invMat := new(Mat)
	invMat.Buf = output
	invMat.Size = mat.Size

	return invMat, nil
}

// Zeros is a matrix with all zeros.
func (op *Op) Zeros(size ...int) (*Mat, error) {
	return op.matInit(zerosOp, size...)
}

// Ones is a matrix with all ones.
func (op *Op) Ones(size ...int) (*Mat, error) {
	return op.matInit(onesOp, size...)
}

// Rand is a matrix with random numbers that have uniform distribution.
func (op *Op) Rand(size ...int) (*Mat, error) {
	return op.matInit(randOp, size...)
}

// Randn is a matrix with random numbers that have normal distribution.
func (op *Op) Randn(size ...int) (*Mat, error) {
	return op.matInit(randnOp, size...)
}

func (op *Op) matInit(opType string, size ...int) (*Mat, error) {
	s := make([]int64, len(size))
	for i, v := range size {
		s[i] = int64(v)
	}

	sizeT, err := tf.NewTensor(s)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(opType).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	mat := new(Mat)
	mat.Buf = output
	mat.Size = make([]int64, len(size))
	for i, v := range size {
		mat.Size[i] = int64(v)
	}

	return mat, nil
}
