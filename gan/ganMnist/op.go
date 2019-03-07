package ganMnist

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/sdeoras/api"
	"github.com/sdeoras/comp/common"
	"github.com/sdeoras/comp/image"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

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

func (op *Op) Load(checkpoint *api.Checkpoint) error {
	feeds := make(map[tf.Output]*tf.Tensor)
	for k, v := range checkpoint.Weights {
		switch k {
		case genHidden1, genOut, discHidden1, discOut:
			localFeeds := make(map[tf.Output]*tf.Tensor)
			dataT, err := tf.NewTensor(v.Data)
			if err != nil {
				return err
			}

			sizeT, err := tf.NewTensor(v.Size)
			if err != nil {
				return err
			}

			localFeeds[op.GetGraph().Operation(buff).Output(0)] = dataT
			localFeeds[op.GetGraph().Operation(shape).Output(0)] = sizeT

			out, err := op.GetSession().Run(
				localFeeds,
				[]tf.Output{
					op.GetGraph().Operation(reshapeOp).Output(0),
				},
				nil,
			)

			if err != nil {
				return err
			}

			if len(out) != 1 {
				return fmt.Errorf("expected output from reshape to have len 1, got:%d", len(out))
			}

			output, ok := out[0].Value().([][]float32)
			if !ok {
				return fmt.Errorf("type assertion error, expected [][]float32, got:%T", out[0].Value())
			}

			outputT, err := tf.NewTensor(output)
			if err != nil {
				return err
			}

			feeds[op.GetGraph().Operation(fmt.Sprintf("%s_w_ph", k)).Output(0)] = outputT
		default:
			return fmt.Errorf("invalid key in checkpoint for this model:%s", k)
		}
	}

	for k, v := range checkpoint.Biases {
		switch k {
		case genHidden1, genOut, discHidden1, discOut:
			localFeeds := make(map[tf.Output]*tf.Tensor)
			dataT, err := tf.NewTensor(v.Data)
			if err != nil {
				return err
			}

			sizeT, err := tf.NewTensor(v.Size)
			if err != nil {
				return err
			}

			localFeeds[op.GetGraph().Operation(buff).Output(0)] = dataT
			localFeeds[op.GetGraph().Operation(shape).Output(0)] = sizeT

			out, err := op.GetSession().Run(
				localFeeds,
				[]tf.Output{
					op.GetGraph().Operation(reshapeOp).Output(0),
				},
				nil,
			)

			if err != nil {
				return err
			}

			if len(out) != 1 {
				return fmt.Errorf("expected output from reshape to have len 1, got:%d", len(out))
			}

			output, ok := out[0].Value().([]float32)
			if !ok {
				return fmt.Errorf("type assertion error, expected [][]float32, got:%T", out[0].Value())
			}

			outputT, err := tf.NewTensor(output)
			if err != nil {
				return err
			}

			feeds[op.GetGraph().Operation(fmt.Sprintf("%s_b_ph", k)).Output(0)] = outputT
		default:
			return fmt.Errorf("invalid key in checkpoint for this model:%s", k)
		}
	}

	// call init for initializing variables by feeding placeholders since
	// variables are initialized using placeholders
	_, err := op.GetSession().Run(
		feeds,
		nil,
		[]*tf.Operation{
			op.GetGraph().Operation(initOp),
		},
	)

	return err
}

func (op *Op) Generate(count int) ([][]byte, error) {
	rand.Seed(time.Now().UnixNano())
	data := make([][]float32, count)
	for i := range data {
		data[i] = make([]float32, 100)
		for j := range data[i] {
			data[i][j] = rand.Float32()*2.0 - 1.0
		}
	}

	dataT, err := tf.NewTensor(data)
	if err != nil {
		return nil, err
	}

	localFeeds := make(map[tf.Output]*tf.Tensor)
	localFeeds[op.GetGraph().Operation(inputNoise).Output(0)] = dataT
	out, err := op.GetSession().Run(
		localFeeds,
		[]tf.Output{
			op.GetGraph().Operation(generatorOutLayer).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 1 {
		return nil, fmt.Errorf("expected len out to be 1, got:%d", len(out))
	}

	output, ok := out[0].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("expected output to be of type [][]float32, got:%T", out[0].Value())
	}

	imOp, err := image.NewOperator(nil)
	if err != nil {
		return nil, err
	}
	defer imOp.Close()

	outImages := make([][]byte, count)
	for p := 0; p < count; p++ {
		q := 0
		im := make([][][]uint8, 28)
		for i := range im {
			im[i] = make([][]uint8, 28)
			for j := range im[i] {
				im[i][j] = make([]uint8, 3)
				for k := range im[i][j] {
					im[i][j][k] = uint8(-1 * (output[p][q] - 1) * 255)
				}
				q++
			}
		}

		b, err := imOp.Encode(image.Image(im))
		if err != nil {
			return nil, err
		}

		outImages[p] = b

		if err := imOp.Write(image.Image(im), fmt.Sprintf("/tmp/gan_%d.jpg", p)); err != nil {
			return nil, err
		}
	}

	return outImages, nil
}
