package image

import (
	"fmt"
	"math"
)

func (raw ImageRaw) Size() (int, int, int) {
	h := len(raw)
	if h == 0 {
		return 0, 0, 0
	}

	w := len(raw[0])
	if w == 0 {
		return h, 0, 0
	}

	return h, w, len(raw[0][0])
}

func (raw ImageRaw) ToImageScaled() Image {
	return raw.toImageScaled(float32(math.MaxUint8))
}

func (raw ImageRaw) ToImageUnscaled() Image {
	return raw.toImageScaled(1)
}

func (raw ImageRaw) toImageScaled(scaleFactor float32) Image {
	h, w, _ := raw.Size()

	image := NewBlack(h, w)

	for i := range raw {
		for j := range raw[i] {
			for k := range raw[i][j] {
				image[i][j][k] = uint8(raw[i][j][k] * scaleFactor)
			}
		}
	}

	return image
}

func (raw ImageRaw) Normalize(mean, variance float32) error {
	if variance == 0 {
		return fmt.Errorf("cannot normalize, variance is zero")
	}

	for i := range raw {
		for j := range raw[i] {
			for k := range raw[i][j] {
				raw[i][j][k] -= mean
				raw[i][j][k] /= float32(math.Sqrt(float64(variance)))
			}
		}
	}

	return nil
}
