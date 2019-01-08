package vec

import (
	"fmt"
	"math/cmplx"

	"github.com/sdeoras/comp/common"
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
	if err := common.InitUsingB64Graph(op, modelPB, options); err != nil {
		return nil, err
	}
	return op, nil
}

func (op *Op) Linspace(start, stop float64, num int) ([]float64, error) {
	startTensor, err := tf.NewTensor(start)
	if err != nil {
		return nil, err
	}

	stopTensor, err := tf.NewTensor(stop)
	if err != nil {
		return nil, err
	}

	numTensor, err := tf.NewTensor(int64(num))
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(startT).Output(0)] = startTensor
	feeds[op.GetGraph().Operation(stopT).Output(0)] = stopTensor
	feeds[op.GetGraph().Operation(numT).Output(0)] = numTensor

	outTensor, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(linspaceOp).Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	if len(outTensor) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	l, ok := outTensor[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("could not generate linspace slice")
	}

	return l, nil
}

func (op *Op) CumSum(input []float64) ([]float64, error) {
	myInput, err := tf.NewTensor(input)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(f64).Output(0)] = myInput

	outTensor, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(cumsumOp).Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	if len(outTensor) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	l, ok := outTensor[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("could not generate cumsum slice")
	}

	return l, nil
}

func (op *Op) CumProd(input []float64) ([]float64, error) {
	myInput, err := tf.NewTensor(input)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(f64).Output(0)] = myInput

	outTensor, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(cumprodOp).Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	if len(outTensor) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	l, ok := outTensor[0].Value().([]float64)
	if !ok {
		return nil, fmt.Errorf("could not generate cumprod slice")
	}

	return l, nil
}

func (op *Op) FFT(input []float64) ([]float64, error) {
	cinput := make([]complex128, len(input))
	for i := range input {
		cinput[i] = complex(input[i], 0)
	}

	myInput, err := tf.NewTensor(cinput)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(fftInput).Output(0)] = myInput

	outTensor, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(fftOp).Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	if len(outTensor) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	l, ok := outTensor[0].Value().([]complex128)
	if !ok {
		return nil, fmt.Errorf("type assertion error:%T", outTensor[0].Value())
	}

	out := make([]float64, len(l))
	for i := range l {
		out[i] = cmplx.Abs(l[i])
	}

	return out, nil
}
