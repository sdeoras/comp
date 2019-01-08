package vec

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// linspace operator constants
	startT     = "start"
	stopT      = "stop"
	numT       = "num"
	linspaceOp = "linspace"

	// inputs
	f64      = "myInput"
	fftInput = "myFFTInput"

	cumsumOp  = "cumsum"
	cumprodOp = "cumprod"
	fftOp     = "fft"
)
