package cloud

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

// NewOperator provides an instance of new operator for
// cloud file I/O
func NewOperator(options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitOperator(op, modelPB, options); err != nil {
		return nil, err
	}
	return op, nil
}

func (op *Op) Enumerate(path string) ([]string, error) {
	inputDir, err := tf.NewTensor(path)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputPath).Output(0)] = inputDir

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(matchingFiles).Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	files, ok := out[0].Value().([]string)
	if !ok {
		return nil, fmt.Errorf("could not get list of matching files")
	}

	return files, nil
}

func (op *Op) Read(file string) ([]byte, error) {
	inputFile, err := tf.NewTensor(file)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputFileName).Output(0)] = inputFile

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(readFile).Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	buf, ok := out[0].Value().(string)
	if !ok {
		return nil, fmt.Errorf("could not get file contents")
	}

	return []byte(buf), nil
}

func (op *Op) Write(file string, buf []byte) error {
	outpuFile, err := tf.NewTensor(file)
	if err != nil {
		return err
	}

	outFileBuffer, err := tf.NewTensor(string(buf))
	if err != nil {
		return err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(outputFileName).Output(0)] = outpuFile
	feeds[op.GetGraph().Operation(outputFileBuffer).Output(0)] = outFileBuffer

	_, err = op.GetSession().Run(
		feeds,
		nil,
		[]*tf.Operation{
			op.GetGraph().Operation(writeFile),
		},
	)

	return err
}
