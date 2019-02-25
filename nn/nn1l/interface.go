package nn1l

// Data is a data to be fed to the trainer in the form of a 2D matrix
// and associated labels.
// Data.Data has observations, one per row (first index) and features as columns.
// Labels has classes probabilities as columns. Class with max probability is compared
// against prediction.
type Data struct {
	Data   [][]float32
	Labels [][]float32
}

// TrainingOutput is the output from each training step.
type TrainingOutput struct {
	CrossEntropy       float32
	TrainingAccuracy   float32
	ValidationAccuracy float32
}

// PredictionOutput is the output from prediction.
type PredictionOutput struct {
	Accuracy         float32
	TruthArgmax      []int64
	PredictionArgmax []int64
}

// CheckPoint is the model checkpoint in the form of weights and biases.
type CheckPoint struct {
	Weights [][]float32
	Biases  []float32
}

// Operator defines methods to train, predict and checkpoint a model
type Operator interface {
	// Step through iterations of training process.
	Step(trainingData, validationData Data) (*TrainingOutput, error)

	// Predict the model outcome.
	Predict(data Data) (*PredictionOutput, error)

	// Save obtains model checkpoint.
	Save() (*CheckPoint, error)

	// Load re-initializes trainer with a checkpoint.
	Load(checkPoint *CheckPoint) error
}
