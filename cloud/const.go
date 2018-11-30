package cloud

//go:generate ./model.sh

// these constants need to match those defined in model.py
const (
	inputPath        = "inputPath"
	matchingFiles    = "matchingFiles"
	inputFileName    = "inputFileName"
	readFile         = "readFile"
	outputFileName   = "outputFileName"
	outputFileBuffer = "outputFileBuffer"
	writeFile        = "writeFile"
)
