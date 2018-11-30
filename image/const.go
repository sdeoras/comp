package image

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	fileNameInput  = "inputFileName"
	inputBuffer    = "inputBuffer"
	inputImage     = "inputImage"
	inputRaw       = "inputRaw"
	inputRawImages = "inputRawImages"
	inputImages    = "inputImages"
	inputSize      = "inputSize"

	decodeFileOp    = "decodeFile"
	decodeOp        = "decode"
	encodeJpgOp     = "encodeJpg"
	rgb2GrayscaleOp = "rgbToGrayscale"
	sobelOp         = "sobel"
	resizeOp        = "resize"
	meanOp          = "mean"
	varianceOp      = "var"

	// this is a complex op, taking in a list of images
	// converting them to grayscale, then resizing them,
	// then convering to black and white, then picking
	// first layer from 3 colors, then flattening and
	// finally normalizing
	batchOp = "batch"
)
