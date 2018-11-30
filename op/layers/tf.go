package layers

import (
	"fmt"

	"github.com/sdeoras/go-scicomp/common"

	"github.com/sdeoras/go-scicomp/image"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Op is the operator for TF interface.
type Op struct {
	// It embeds common op
	common.Op
}

// NewOperator provides an instance of new operator for interfacing
// with TF.
func NewOperator(options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitOperator(op, modelPB, options); err != nil {
		return nil, err
	}
	return op, nil
}

func (op *Op) FlattenImage(image image.Image) ([]byte, error) {
	data := make([][][][]uint8, 1)
	data[0] = image

	inputT, err := tf.NewTensor(data)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(myInput).Output(0)] = inputT

	if op.GetGraph().Operation(flattenOp) == nil {
		return nil, fmt.Errorf("op is nil")
	}

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(flattenOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][]uint8)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	return output[0], nil
}

func (op *Op) FlattenSlice(imageSlice image.Slice) ([]byte, error) {
	data := make([][][]uint8, 1)
	data[0] = imageSlice

	inputT, err := tf.NewTensor(data)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(myInputSlice).Output(0)] = inputT

	if op.GetGraph().Operation(flattenSliceOp) == nil {
		return nil, fmt.Errorf("op is nil")
	}

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(flattenSliceOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][]uint8)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	return output[0], nil
}
