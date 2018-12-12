package image

import "io"

// Color is one of three color values
type Color int

const (
	Red Color = iota
	Green
	Blue
)

// EdgeType defines the edge detection direction.
type EdgeType int

const (
	Y EdgeType = iota
	X
)

type Stats struct {
	Mean     float32
	Variance float32
}

// Image defines format of data returned by TensorFlow after image decoding
type Image [][][]uint8
type ImageRaw [][][]float32

// Slice is a color slice of an Image
type Slice [][]uint8
type SliceRaw [][]float32

// Mask for an image.
type Mask [][]bool

// SobelRaw is the raw output from sobel filter
type SobelRaw [][][][]float32

// A batch represents a batch of images, one per line, flattened.
type Batch [][]float32

const (
	depth = 3
)

// ImageProcessor works on an image producing various transformations
type ImageProcessor interface {
	Size() (int, int, int)
	Slice(c Color) (Slice, error)
	Flatten(image Image, flattenFunc func(r, g, b uint8) uint8) Slice
}

// Operator defines operations for working with images
type Operator interface {
	// Read decodes input jpg file into an Image
	Read(fileName string) (Image, error)

	// Write writes image back to disk as a jpg file.
	Write(image Image, fileName string) error

	// Decode decodes byte buffer from reader into an Image
	Decode(r io.Reader) (Image, error)

	// Encode encodes image into byte buffer suitable for writing jpg file.
	Encode(image Image) ([]byte, error)

	// RGB2GrayScale converts input image into gray scale
	RGB2GrayScale(images ...Image) ([]Image, error)

	// Resize uses bilinear algorithm with alignCorners = true to resize an image.
	// height and width, when non positive, autoscale will decide the values.
	Resize(height, width int, images ...Image) ([]ImageRaw, error)

	// ResizeNormalize will resize and normalize images. It subtracts mean and then
	// divides by std.
	ResizeNormalize(height, width int, mean, std float32, images ...Image) ([]ImageRaw, error)

	// Sobel returns raw sobel output from sobel filter.
	Sobel(images ...ImageRaw) ([]SobelRaw, error)

	// Stats returns mean and variance for a raw image
	GetStats(images ...ImageRaw) ([]Stats, error)

	// Batch does following to each image:
	// * Convert to grayscale
	// * Resize
	// * Extract slide
	// * Normalize
	MakeBatch(height, width int, images ...Image) (Batch, error)
}
