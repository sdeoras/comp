package norm

import (
	"fmt"

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

func (m *Op) Softmax(input []float64) ([]float64, error) {
	inputT, err := tf.NewTensor(input)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[m.GetGraph().Operation(f64).Output(0)] = inputT

	out, err := m.GetSession().Run(
		feeds,
		[]tf.Output{
			m.GetGraph().Operation(softmaxOp).Output(0),
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
		return nil, fmt.Errorf("could not get valid output due to type assertion errors")
	}

	return output, nil
}
