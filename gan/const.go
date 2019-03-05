package gan

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// init op to initialize variables
	initOp = "init"

	// keys to pull fron checkpoint proto buffer
	genHidden1  = "gen_hidden1"
	genOut      = "gen_out"
	discHidden1 = "disc_hidden1"
	discOut     = "disc_out"

	// to reshape
	buff      = "buff"
	shape     = "shape"
	reshapeOp = "reshapeOp"

	// inference and training inputs
	inputNoise                = "input_noise"
	discInput                 = "disc_input"
	generatorOutLayer         = "generatorOutLayer"
	realDiscriminatorOutLayer = "realDiscriminatorOutLayer"
	fakeDiscriminatorOutLayer = "fakeDiscriminatorOutLayer"
)
