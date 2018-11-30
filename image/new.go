package image

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// NewBlack creates a new image with zeros in all channels.
func NewBlack(height, width int) Image {
	im := make([][][]uint8, height)
	for i := range im {
		im[i] = make([][]uint8, width)
		for j := range im[i] {
			im[i][j] = make([]uint8, depth)
		}
	}
	return im
}

// NewWhite creates a new image with all values set to 255.
func NewWhite(height, width int) Image {
	im := make([][][]uint8, height)
	for i := range im {
		im[i] = make([][]uint8, width)
		for j := range im[i] {
			im[i][j] = make([]uint8, depth)
			for k := range im[i][j] {
				im[i][j][k] = math.MaxUint8
			}
		}
	}
	return im
}

// NewRand creates a new image with random noise.
func NewRand(height, width int) Image {
	rand.Seed(time.Now().UnixNano())
	im := make([][][]uint8, height)
	for i := range im {
		im[i] = make([][]uint8, width)
		for j := range im[i] {
			im[i][j] = make([]uint8, depth)
			for k := range im[i][j] {
				im[i][j][k] = uint8(rand.Intn(math.MaxUint8))
			}
		}
	}
	return im
}

// NewConst creates a image with different channels offset with constants.
func NewConst(height, width int, r, g, b uint8) Image {
	c := make([]uint8, depth)
	c[0], c[1], c[2] = r, g, b

	im := make([][][]uint8, height)
	for i := range im {
		im[i] = make([][]uint8, width)
		for j := range im[i] {
			im[i][j] = make([]uint8, depth)
			for k := range im[i][j] {
				im[i][j][k] = c[k]
			}
		}
	}
	return im
}

// Copy copies an image.
func Copy(im Image) Image {
	height := len(im)
	if height == 0 {
		return nil
	}

	width := len(im[0])
	if width == 0 {
		return nil
	}

	depth := len(im[0][0])
	if depth != depth {
		return nil
	}

	im2 := NewBlack(height, width)

	for i := range im {
		for j := range im[i] {
			for k := range im[i][j] {
				im2[i][j][k] = im[i][j][k]
			}
		}
	}

	return im2
}

// Add adds two images and creates a new one.
func Add(im1, im2 Image) (Image, error) {
	if !IsSameSize(im1, im2) {
		return nil, fmt.Errorf("not same sizes")
	}

	h, w, d := im1.Size()

	if d != depth {
		return nil, fmt.Errorf("depth not equal to 3")
	}

	im := NewWhite(h, w)

	for i := range im1 {
		for j := range im1[i] {
			for k := range im1[i][j] {
				im[i][j][k] = im1[i][j][k] + im2[i][j][k]
			}
		}
	}

	return im, nil
}

// IsSameSize returns of two input images are of the same size.
func IsSameSize(im1, im2 Image) bool {
	h1, w1, d1 := im1.Size()
	h2, w2, d2 := im2.Size()

	if h1 != h2 || w1 != w2 || d1 != d2 {
		return false
	}

	return true
}
