package nnl4

import (
	"fmt"
	"math/rand"

	"github.com/sdeoras/comp/nn"

	"github.com/helinwang/tfsum"
	"github.com/sdeoras/comp/common"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Op struct {
	// It embeds common op
	common.Op
	learningRate        float32
	logdir              string
	name                string
	trainingStepCounter int64
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

// New provides a new trainer operator that can be used to feed taining data
// and step through training process.
func New(dim []int, learningRate float64, name, logdir string, options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitUsingB64Graph(op, modelPB, options); err != nil {
		return nil, err
	}

	if len(dim) != numLayers+1 {
		return nil, fmt.Errorf("you must provide 5 values for dim")
	}

	op.learningRate = float32(learningRate)
	op.logdir = logdir
	op.name = name

	feeds := make(map[tf.Output]*tf.Tensor)

	for k := 0; k < len(dim)-1; k++ {
		weights := make([][]float32, dim[k])
		for i := range weights {
			weights[i] = make([]float32, dim[k+1])
			for j := range weights[i] {
				weights[i][j] = rand.Float32()
			}
		}

		biases := make([]float32, dim[k+1])
		for i := range biases {
			biases[i] = rand.Float32()
		}

		weightsT, err := tf.NewTensor(weights)
		if err != nil {
			return nil, err
		}

		biasesT, err := tf.NewTensor(biases)
		if err != nil {
			return nil, err
		}

		feeds[op.GetGraph().Operation(
			fmt.Sprintf("%s%s%d", weightsInput, "Layer", k+1),
		).Output(0)] = weightsT
		feeds[op.GetGraph().Operation(
			fmt.Sprintf("%s%s%d", biasesInput, "Layer", k+1),
		).Output(0)] = biasesT
	}

	if _, err := op.GetSession().Run(
		feeds,
		nil,
		[]*tf.Operation{
			op.GetGraph().Operation(initOp),
		},
	); err != nil {
		return nil, err
	}

	return op, nil
}

// Load loads a given checkpoitn into current training session of receiver.
func (op *Op) Load(checkPoint *nn.CheckPoint) error {
	if checkPoint == nil || checkPoint.Weights == nil || checkPoint.Biases == nil {
		return fmt.Errorf("input is nil in checkpoint, cannot load")
	}

	feeds := make(map[tf.Output]*tf.Tensor)

	for k := 0; k < numLayers; k++ {
		if len(checkPoint.Weights[k][0]) != len(checkPoint.Biases[k]) {
			return fmt.Errorf("weights and biases lengths are not compatible for %d-th pair", k)
		}

		weightsT, err := tf.NewTensor(checkPoint.Weights[k])
		if err != nil {
			return err
		}

		biasesT, err := tf.NewTensor(checkPoint.Biases[k])
		if err != nil {
			return err
		}

		feeds[op.GetGraph().Operation(
			fmt.Sprintf("%s%s%d", weightsInput, "Layer", k+1),
		).Output(0)] = weightsT
		feeds[op.GetGraph().Operation(
			fmt.Sprintf("%s%s%d", biasesInput, "Layer", k+1),
		).Output(0)] = biasesT
	}

	_, err := op.GetSession().Run(
		feeds,
		nil,
		[]*tf.Operation{
			op.GetGraph().Operation(initOp),
		},
	)

	return err
}

// Save saves model checkpoint.
func (op *Op) Save() (*nn.CheckPoint, error) {
	out, err := op.GetSession().Run(
		nil,
		[]tf.Output{
			op.GetGraph().Operation(weightsOp + "Layer1").Output(0),
			op.GetGraph().Operation(biasesOp + "Layer1").Output(0),
			op.GetGraph().Operation(weightsOp + "Layer2").Output(0),
			op.GetGraph().Operation(biasesOp + "Layer2").Output(0),
			op.GetGraph().Operation(weightsOp + "Layer3").Output(0),
			op.GetGraph().Operation(biasesOp + "Layer3").Output(0),
			op.GetGraph().Operation(weightsOp + "Layer4").Output(0),
			op.GetGraph().Operation(biasesOp + "Layer4").Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 8 {
		return nil, fmt.Errorf("expected length == 8, got:%d", len(out))
	}

	weights := make([][][]float32, numLayers)
	biases := make([][]float32, numLayers)
	var ok bool

	weights[0], ok = out[0].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid weights due to type assertion errors: %T", out[0].Value())
	}

	biases[0], ok = out[1].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid biases due to type assertion errors: %T", out[1].Value())
	}

	weights[1], ok = out[2].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid weights due to type assertion errors: %T", out[2].Value())
	}
	biases[1], ok = out[3].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid biases due to type assertion errors: %T", out[3].Value())
	}

	weights[2], ok = out[4].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid weights due to type assertion errors: %T", out[4].Value())
	}
	biases[2], ok = out[5].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid biases due to type assertion errors: %T", out[5].Value())
	}

	weights[3], ok = out[6].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid weights due to type assertion errors: %T", out[6].Value())
	}
	biases[3], ok = out[7].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid biases due to type assertion errors: %T", out[7].Value())
	}

	return &nn.CheckPoint{
		Weights: weights,
		Biases:  biases,
	}, nil
}

// Predict does inference on given data.
func (op *Op) Predict(data *nn.Data) (*nn.PredictionOutput, error) {
	dataT, err := tf.NewTensor(data.X)
	if err != nil {
		return nil, err
	}

	labelsT, err := tf.NewTensor(data.Y)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(batchInput).Output(0)] = dataT
	feeds[op.GetGraph().Operation(labelsInput).Output(0)] = labelsT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(accuracyOp).Output(0),
			op.GetGraph().Operation(predictionOp).Output(0),
			op.GetGraph().Operation(truthOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 3 {
		return nil, fmt.Errorf("invalid trainingAccuracy length of zero")
	}

	trainingAccuracy, ok := out[0].Value().(float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid trainingAccuracy due to type assertion errors: %T", out[0].Value())
	}

	prediction, ok := out[1].Value().([]int64)
	if !ok {
		return nil, fmt.Errorf("could not get valid prediction due to type assertion errors: %T", out[1].Value())
	}

	truth, ok := out[2].Value().([]int64)
	if !ok {
		return nil, fmt.Errorf("could not get valid truth due to type assertion errors: %T", out[2].Value())
	}

	return &nn.PredictionOutput{
		Accuracy:         trainingAccuracy,
		PredictionArgmax: prediction,
		TruthArgmax:      truth,
	}, nil
}

// Step steps through iteration process.
func (op *Op) Step(trainingData, validationData *nn.Data) (*nn.TrainingOutput, error) {
	// increment counter
	defer func() {
		op.trainingStepCounter++
	}()

	trainingDataT, err := tf.NewTensor(trainingData.X)
	if err != nil {
		return nil, err
	}

	trainingLabelsT, err := tf.NewTensor(trainingData.Y)
	if err != nil {
		return nil, err
	}

	validationDataT, err := tf.NewTensor(validationData.X)
	if err != nil {
		return nil, err
	}

	validationLabelsT, err := tf.NewTensor(validationData.Y)
	if err != nil {
		return nil, err
	}

	learningRateT, err := tf.NewTensor(op.learningRate)
	if err != nil {
		return nil, err
	}

	w := &tfsum.Writer{Dir: op.logdir, Name: op.name}
	defer w.Close()

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(batchInput).Output(0)] = trainingDataT
	feeds[op.GetGraph().Operation(labelsInput).Output(0)] = trainingLabelsT
	feeds[op.GetGraph().Operation(learningRateInput).Output(0)] = learningRateT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(accuracyOp).Output(0),
			op.GetGraph().Operation(crossEntropyOp).Output(0),
			op.GetGraph().Operation(summaryTrainingOp).Output(0),
		},
		[]*tf.Operation{
			op.GetGraph().Operation(trainStepOp),
		},
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 3 {
		return nil, fmt.Errorf("invalid output length, expected 3, got:%d", len(out))
	}

	trainingAccuracy, ok := out[0].Value().(float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid training accuracy due to type assertion errors")
	}

	crossEntropy, ok := out[1].Value().(float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid training cross entropy due to type assertion errors")
	}

	summary, ok := out[2].Value().(string)
	if !ok {
		return nil, fmt.Errorf("could not get valid training summary due to type assertion errors")
	}

	if err := w.AddEvent(summary, op.trainingStepCounter); err != nil {
		return nil, err
	}

	// now work on validation set
	feeds = make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(batchInput).Output(0)] = validationDataT
	feeds[op.GetGraph().Operation(labelsInput).Output(0)] = validationLabelsT
	feeds[op.GetGraph().Operation(learningRateInput).Output(0)] = learningRateT

	out, err = op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(accuracyOp).Output(0),
			op.GetGraph().Operation(summaryValidationOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid output length, expected 2, got %d", len(out))
	}

	validationAccuracy, ok := out[0].Value().(float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid validation accuracy due to type assertion errors")
	}

	summary, ok = out[1].Value().(string)
	if !ok {
		return nil, fmt.Errorf("could not get valid validation summary due to type assertion errors")
	}

	if err := w.AddEvent(summary, op.trainingStepCounter); err != nil {
		return nil, err
	}

	return &nn.TrainingOutput{
		TrainingAccuracy:   trainingAccuracy,
		ValidationAccuracy: validationAccuracy,
		CrossEntropy:       crossEntropy,
	}, nil
}
