package math

import (
	"fmt"
	"math"

	"github.com/sdeoras/go-scicomp/common"

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

func (op *Op) reduce(input []float64, operator string) (float64, error) {
	inputT, err := tf.NewTensor(input)
	if err != nil {
		return 0, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(f64).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(operator).Output(0),
		},
		nil,
	)

	if err != nil {
		return 0, err
	}

	if len(out) == 0 {
		return 0, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().(float64)
	if !ok {
		return 0, fmt.Errorf("could not get valid output due to type assertion errors")
	}

	return output, nil
}

func (op *Op) Mean(input []float64) (float64, error) {
	return op.reduce(input, meanOp)
}

func (op *Op) Max(input []float64) (float64, error) {
	return op.reduce(input, maxOp)
}

func (op *Op) Min(input []float64) (float64, error) {
	return op.reduce(input, minOp)
}

func (op *Op) Prod(input []float64) (float64, error) {
	return op.reduce(input, prodOp)
}

func (op *Op) Sum(input []float64) (float64, error) {
	return op.reduce(input, sumOp)
}

func (op *Op) Std(input []float64) (float64, error) {
	inputT, err := tf.NewTensor(input)
	if err != nil {
		return 0, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(f64).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(varOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return 0, err
	}

	if len(out) == 0 {
		return 0, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().(float64)
	if !ok {
		return 0, fmt.Errorf("could not get valid output due to type assertion errors")
	}

	if output < 0 {
		return 0, fmt.Errorf("invalid negative output")
	}

	output = math.Sqrt(output)

	return output, nil
}
