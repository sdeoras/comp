package nn4l

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// layer count
	numLayers = 4

	// inputs
	weightsInput      = "weightsInit"
	biasesInput       = "biasesInit"
	batchInput        = "x"
	labelsInput       = "labels"
	learningRateInput = "learningRate"
	// ops
	initOp         = "init"
	crossEntropyOp = "crossEntropy"
	accuracyOp     = "accuracy"
	trainStepOp    = "trainStep"
	predictionOp   = "prediction"
	truthOp        = "truth"
	weightsOp      = "weights"
	biasesOp       = "biases"

	// tensorboard summary
	summaryTrainingOp   = "summaryTraining"
	summaryValidationOp = "summaryValidation"
)
