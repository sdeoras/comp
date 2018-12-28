package layers

import "github.com/sdeoras/comp/image"

type Operator interface {
	FlattenImage(image image.Image) ([]byte, error)
	FlattenSlice(imageSlice image.Slice) ([]byte, error)
	//FlattenImages(images ...image.Image) ([][]float32, error)
	//FlattenSlices(imageSlices ...image.Slice) ([][]float32, error)
}
