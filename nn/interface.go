package nn

// Operator defines methods to train, predict and checkpoint a model
type Operator interface {
	// Step through iterations of training process.
	Step(trainingData, validationData *Data) (*TrainingOutput, error)

	// Predict the model outcome.
	Predict(data *Data) (*PredictionOutput, error)

	// Save obtains model checkpoint.
	Save() (*CheckPoint, error)

	// Load re-initializes trainer with a checkpoint.
	Load(checkPoint *CheckPoint) error
}
