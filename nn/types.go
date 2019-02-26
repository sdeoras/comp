package nn

// Data is a data to be fed to the trainer in the form of a 2D matrix
// and associated labels.
// Data.Data has observations, one per row (first index) and features as columns.
// Y has classes probabilities as columns. Class with max probability is compared
// against prediction.
type Data struct {
	X [][]float32
	Y [][]float32
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
	Weights [][][]float32
	Biases  [][]float32
}
