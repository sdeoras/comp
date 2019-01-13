package mat

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Transpose takes transpose of a matrix. Defined only for a 2D matrix.
func (op *Op) Transpose(mat *Mat) (*Mat, error) {
	if len(mat.Size) != 2 {
		return nil, fmt.Errorf("transpose only defined for a 2D matrix")
	}

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
			op.GetGraph().Operation(transposeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid outMat length, expected 1, got: %d", len(out))
	}

	outMat, ok := out[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	transposedMat := new(Mat)
	transposedMat.Buf = outMat
	transposedMat.Size = []int64{mat.Size[1], mat.Size[0]}

	return transposedMat, nil
}
