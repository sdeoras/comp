package find

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

func (op *Op) Unique(input []float64) ([]float64, []int, error) {
	inputT, err := tf.NewTensor(input)
	if err != nil {
		return nil, nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(f64).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(uniqueOp_f64).Output(0),
			op.GetGraph().Operation(uniqueOp_f64).Output(2),
		},
		nil,
	)

	if err != nil {
		return nil, nil, err
	}

	if len(out) == 0 {
		return nil, nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok1 := out[0].Value().([]float64)
	count, ok2 := out[1].Value().([]int32)
	if !ok1 || !ok2 {
		return nil, nil, fmt.Errorf("could not get valid output due to type assertion errors")
	}

	countInt := make([]int, len(count))
	for i := range count {
		countInt[i] = int(count[i])
	}

	return output, countInt, nil
}
