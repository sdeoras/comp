package image

import (
	"fmt"
)

// NewMask returns a new mask with all values set to false.
func NewMask(height, width int) Mask {
	m := make([][]bool, height)
	for i := range m {
		m[i] = make([]bool, width)
	}
	return m
}

// Size returns size of the mask.
func (mask Mask) Size() (int, int) {
	height := len(mask)
	if height == 0 {
		return 0, 0
	}

	return height, len(mask[0])
}

// Invert inverts a mask modying the original mask and
// returning the same
func (mask Mask) Invert() Mask {
	for i := range mask {
		for j := range mask[i] {
			mask[i][j] = !mask[i][j]
		}
	}

	return mask
}

// InheritEdges updates mask with new edge data extracted from
// raw sobel data and a threshold.
func (mask Mask) InheritEdges(sobelRaw SobelRaw,
	edgeType EdgeType, f func(float32) bool, colors ...Color) error {
	h0, w0, d := sobelRaw.Size()
	h1, w1 := mask.Size()

	if h0 != h1 || w0 != w1 {
		return fmt.Errorf("mask and sobel raw sizes do not match")
	}

	for i := range mask {
		for j := range mask[i] {
			for k := range colors {
				if k < 0 || k >= d {
					return fmt.Errorf("index out of bounds for input colors")
				}
				if f(sobelRaw[i][j][k][edgeType]) {
					mask[i][j] = true
				}
			}

		}
	}

	return nil
}

// ToSliceRaw converts a mask into a floating point equivalent image frame.
func (mask Mask) ToSliceRaw() SliceRaw {
	h, w := mask.Size()

	im := make([][]float32, h)
	for i := range im {
		im[i] = make([]float32, w)
		for j := range im[i] {
			if mask[i][j] {
				im[i][j] = 1
			}
		}
	}

	return im
}
