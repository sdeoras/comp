package train

import (
	"fmt"
	"math/rand"

	"github.com/helinwang/tfsum"
	"github.com/sdeoras/go-scicomp/common"
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
func New(numFeatures, numClasses int, learningRate float64, name, logdir string, options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitUsingB64Graph(op, modelPB, options); err != nil {
		return nil, err
	}

	op.learningRate = float32(learningRate)
	op.logdir = logdir
	op.name = name

	weights := make([][]float32, numFeatures)
	for i := range weights {
		weights[i] = make([]float32, numClasses)
		for j := range weights[i] {
			weights[i][j] = rand.Float32()
		}
	}
	weightsT, err := tf.NewTensor(weights)
	if err != nil {
		return nil, err
	}

	biases := make([]float32, numClasses)
	for i := range biases {
		biases[i] = rand.Float32()
	}

	biasesT, err := tf.NewTensor(biases)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(weightsInput).Output(0)] = weightsT
	feeds[op.GetGraph().Operation(biasesInput).Output(0)] = biasesT

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
func (op *Op) Load(checkPoint *CheckPoint) error {
	if checkPoint == nil || checkPoint.Weights == nil || checkPoint.Biases == nil {
		return fmt.Errorf("input is nil in checkpoint, cannot load")
	}

	if len(checkPoint.Weights[0]) != len(checkPoint.Biases) {
		return fmt.Errorf("weights and biases lengths are not compatible")
	}

	weightsT, err := tf.NewTensor(checkPoint.Weights)
	if err != nil {
		return err
	}

	biasesT, err := tf.NewTensor(checkPoint.Biases)
	if err != nil {
		return err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(weightsInput).Output(0)] = weightsT
	feeds[op.GetGraph().Operation(biasesInput).Output(0)] = biasesT

	_, err = op.GetSession().Run(
		feeds,
		nil,
		[]*tf.Operation{
			op.GetGraph().Operation(initOp),
		},
	)

	return err
}

// Save saves model checkpoint.
func (op *Op) Save() (*CheckPoint, error) {
	out, err := op.GetSession().Run(
		nil,
		[]tf.Output{
			op.GetGraph().Operation(weightsOp).Output(0),
			op.GetGraph().Operation(biasesOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid trainingAccuracy length of zero")
	}

	weights, ok := out[0].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid weights due to type assertion errors: %T", out[0].Value())
	}

	biases, ok := out[1].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("could not get valid biases due to type assertion errors: %T", out[1].Value())
	}

	return &CheckPoint{
		Weights: weights,
		Biases:  biases,
	}, nil
}

// Predict does inference on given data.
func (op *Op) Predict(data Data) (*PredictionOutput, error) {
	dataT, err := tf.NewTensor(data.Data)
	if err != nil {
		return nil, err
	}

	labelsT, err := tf.NewTensor(data.Labels)
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

	return &PredictionOutput{
		Accuracy:         trainingAccuracy,
		PredictionArgmax: prediction,
		TruthArgmax:      truth,
	}, nil
}

// Step steps through iteration process.
func (op *Op) Step(trainingData, validationData Data) (*TrainingOutput, error) {
	// increment counter
	defer func() {
		op.trainingStepCounter++
	}()

	trainingDataT, err := tf.NewTensor(trainingData.Data)
	if err != nil {
		return nil, err
	}

	trainingLabelsT, err := tf.NewTensor(trainingData.Labels)
	if err != nil {
		return nil, err
	}

	validationDataT, err := tf.NewTensor(validationData.Data)
	if err != nil {
		return nil, err
	}

	validationLabelsT, err := tf.NewTensor(validationData.Labels)
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

	return &TrainingOutput{
		TrainingAccuracy:   trainingAccuracy,
		ValidationAccuracy: validationAccuracy,
		CrossEntropy:       crossEntropy,
	}, nil
}
