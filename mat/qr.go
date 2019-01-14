package mat

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Qr takes QR decomposition of a matrix.
func (op *Op) Qr(mat *Mat) (*Mat, *Mat, error) {
	bufT, err := tf.NewTensor(mat.Buf)
	if err != nil {
		return nil, nil, err
	}

	sizeT, err := tf.NewTensor(mat.Size)
	if err != nil {
		return nil, nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer1).Output(0)] = bufT
	feeds[op.GetGraph().Operation(inputShape1).Output(0)] = sizeT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(qrOpQ).Output(0),
			op.GetGraph().Operation(qrOpR).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, nil, err
	}

	if len(out) < 2 {
		return nil, nil, fmt.Errorf("invalid outQ length, expected 2, got: %d", len(out))
	}

	outQ, ok := out[0].Value().([]float64)
	if !ok {
		return nil, nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[0].Value())
	}

	qMat := new(Mat)
	qMat.Buf = outQ
	qMat.Size = mat.Size

	outR, ok := out[1].Value().([]float64)
	if !ok {
		return nil, nil, fmt.Errorf("type assertion error, expected []float64, got:%T", out[1].Value())
	}

	rMat := new(Mat)
	rMat.Buf = outR
	rMat.Size = []int64{mat.Size[1], mat.Size[1]}

	return qMat, rMat, nil
}
