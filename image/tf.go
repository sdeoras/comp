package image

import (
	"fmt"
	"io"
	"io/ioutil"

	"github.com/sdeoras/go-scicomp/common"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Op is the operator for TF interface.
type Op struct {
	// It embeds common op
	common.Op
}

// NewOperator provides an operator that implements Operator interface
// for jpg images.
func NewOperator(options *tf.SessionOptions) (*Op, error) {
	op := new(Op)
	if err := common.InitOperator(op, modelPB, options); err != nil {
		return nil, err
	}
	return op, nil
}

// Read reads an Image from a file.
func (op *Op) Read(fileName string) (Image, error) {
	inputT, err := tf.NewTensor(fileName)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(fileNameInput).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(decodeFileOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][][]uint8)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	return output, nil
}

// Write writes an image to a file.
func (op *Op) Write(image Image, fileName string) error {
	b, err := op.Encode(image)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(fileName, b, 0644)
}

// Encode encodes input image into byte buffer suitable for writing file.
func (op *Op) Encode(image Image) ([]byte, error) {
	inputT, err := tf.NewTensor(image)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputImage).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(encodeJpgOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().(string)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	return []byte(output), nil
}

// Decode decodes bytes from input reader into an Image.
func (op *Op) Decode(r io.Reader) (Image, error) {
	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	inputT, err := tf.NewTensor(string(buf))
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputBuffer).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(decodeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][][]uint8)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	return output, nil
}

// RGB2GrayScale reduces color image into a grayscale.
func (op *Op) RGB2GrayScale(images ...Image) ([]Image, error) {
	inputT, err := tf.NewTensor(images)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputImages).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(rgb2GrayscaleOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][][][]uint8)
	if !ok {
		return nil, fmt.Errorf("could not get valid output due to type assertion errors: %T", out[0].Value())
	}

	outImages := make([]Image, len(images))
	for i := range outImages {
		outImages[i] = output[i]
	}

	return outImages, nil
}

// Sobel returns the Y edges, X edges and raw sobel data as output.
func (op *Op) Sobel(images ...ImageRaw) ([]SobelRaw, error) {
	inputT, err := tf.NewTensor(images)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputRaw).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(sobelOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][][][][]float32)
	if !ok {
		return nil, fmt.Errorf("invalid output, could not perform type assertion")
	}

	rawSobel := make([]SobelRaw, len(images))
	for i := range rawSobel {
		rawSobel[i] = output[i]
	}

	return rawSobel, nil
}

// Resize resized input image using bilinear algorithm.
func (op *Op) Resize(height, width int, images ...Image) ([]ImageRaw, error) {
	if len(images) == 0 {
		return nil, nil
	}

	h0, w0, _ := images[0].Size()

	if height <= 0 && width <= 0 {
		height = h0
		width = w0
	}

	if height <= 0 && width > 0 {
		height = h0 * width / w0
	}

	if height > 0 && width <= 0 {
		width = w0 * height / h0
	}

	data := make([]Image, len(images))

	for i := range data {
		data[i] = images[i]
	}

	inputT, err := tf.NewTensor(data)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor([]int32{int32(height), int32(width)})
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputImage).Output(0)] = inputT
	feeds[op.GetGraph().Operation(inputSize).Output(0)] = sizeT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(resizeOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][][][]float32)
	if !ok {
		return nil, fmt.Errorf("invalid output, could not perform type assertion:%T", out[0].Value())
	}

	rawImages := make([]ImageRaw, len(images))

	for i := range rawImages {
		rawImages[i] = output[i]
	}

	return rawImages, nil
}

// GetStats gets image stats for normalization.
func (op *Op) GetStats(images ...ImageRaw) ([]Stats, error) {
	if len(images) == 0 {
		return nil, nil
	}

	inputT, err := tf.NewTensor(images)
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputRawImages).Output(0)] = inputT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(meanOp).Output(0),
			op.GetGraph().Operation(varianceOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) != 2 {
		return nil, fmt.Errorf("invalid output length. Expected 2")
	}

	output1, ok := out[0].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("invalid output, could not perform type assertion:%T", out[0].Value())
	}

	output2, ok := out[1].Value().([]float32)
	if !ok {
		return nil, fmt.Errorf("invalid output, could not perform type assertion:%T", out[0].Value())
	}

	if len(output1) != len(output2) {
		return nil, fmt.Errorf("mean and var lists are not of equal lengths")
	}

	stats := make([]Stats, len(output1))

	for i := range output1 {
		stats[i] = Stats{
			Mean:     output1[i],
			Variance: output2[i],
		}
	}

	return stats, nil
}

// GetBatch performs batch grayscale, resize and normalization, producing
// a 2D matrix of float32.
func (op *Op) MakeBatch(height, width int, images ...Image) (Batch, error) {
	if len(images) == 0 {
		return nil, nil
	}

	h0, w0, _ := images[0].Size()

	if height <= 0 && width <= 0 {
		height = h0
		width = w0
	}

	if height <= 0 && width > 0 {
		height = h0 * width / w0
	}

	if height > 0 && width <= 0 {
		width = w0 * height / h0
	}

	inputT, err := tf.NewTensor(images)
	if err != nil {
		return nil, err
	}

	sizeT, err := tf.NewTensor([]int32{int32(height), int32(width)})
	if err != nil {
		return nil, err
	}

	feeds := make(map[tf.Output]*tf.Tensor)
	feeds[op.GetGraph().Operation(inputImages).Output(0)] = inputT
	feeds[op.GetGraph().Operation(inputSize).Output(0)] = sizeT

	out, err := op.GetSession().Run(
		feeds,
		[]tf.Output{
			op.GetGraph().Operation(batchOp).Output(0),
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("invalid output length of zero")
	}

	output, ok := out[0].Value().([][]float32)
	if !ok {
		return nil, fmt.Errorf("invalid output, could not perform type assertion:%T", out[0].Value())
	}

	return output, nil
}
