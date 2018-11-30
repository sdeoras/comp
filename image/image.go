package image

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Size returns height, width and number of color channels.
func (image Image) Size() (int, int, int) {
	height := len(image)
	if height == 0 {
		return 0, 0, 0
	}

	width := len(image[0])
	if width == 0 {
		return height, 0, 0
	}

	return height, width, len(image[0][0])
}

func (image Image) Slice(c Color) (Slice, error) {
	if c < 0 || c >= 3 {
		return nil, fmt.Errorf("incorrect color value")
	}

	height, width, _ := image.Size()
	imageSclice := make([][]uint8, height)
	for i := range imageSclice {
		imageSclice[i] = make([]uint8, width)
		for j := range imageSclice[i] {
			imageSclice[i][j] = image[i][j][c]
		}
	}

	return imageSclice, nil
}

// scribe is internal function. Length of x and y should be 2
// else it will panic
func (image Image) scribe(height, width int, x, y []int) Image {
	if len(x) != 2 || len(y) != 2 {
		panic("input x and y must have length 2 each")
	}

	slope := float64(y[1]-y[0]) / float64(x[1]-x[0])
	c := float64(y[0]) - slope*float64(x[0])
	start := x[0]
	n := x[1] - x[0]
	if n < 0 {
		n = x[0] - x[1]
		start = x[1]
	}

	for i := 0; i < n; i++ {
		px := start + i
		py := int(slope*float64(px) + c)

		if px >= 0 && py >= 0 && px < height && py < width {
			for k := 0; k < depth; k++ {
				image[px][py][k] = 0
			}
		}
	}

	return image
}

// RandScribe draws a random scribe (line) on receiver image
// and modifies it before returning it.
func (image Image) RandScribe() Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	x := make([]int, 2)
	y := make([]int, 2)

	// pick two rand points
	for y[0] == y[1] || x[0] == x[1] {
		for i := range x {
			x[i] = rand.Intn(height)
			y[i] = rand.Intn(width)
		}
	}

	image.scribe(height, width, x, y)

	return image
}

// RandTriangle draws a random triangle on receiver image
// and modifies it before returning it.
func (image Image) RandTriangle() Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	x := make([]int, 3)
	y := make([]int, 3)

	// pick rand points
	for y[0] == y[1] && x[0] == x[1] ||
		y[1] == y[2] && x[1] == x[2] ||
		y[0] == y[2] && x[0] == x[2] {
		for i := range x {
			x[i] = rand.Intn(height)
			y[i] = rand.Intn(width)
		}
	}
	x = append(x, x[0])
	y = append(y, y[0])

	image.scribe(height, width, x[0:2], y[0:2]).
		scribe(height, width, x[1:3], y[1:3]).
		scribe(height, width, x[2:4], y[2:4])

	return image
}

// RandCircle draws random partially filled circle.
func (image Image) RandCircle() Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	xc := rand.Intn(height)
	yc := rand.Intn(width)
	r := 0
	for r == 0 {
		r = height
		if r > width {
			r = width
		}
		r = rand.Intn(r) / 3
	}

	for y := yc - r; y <= yc+r; y++ {
		for x := xc - r; x < xc+r; x++ {
			if x >= 0 && x < height && y >= 0 && y < width &&
				(x-xc)*(x-xc)+(y-yc)*(y-yc) <= r*r {
				if rand.Intn(2) == 1 {
					for k := 0; k < depth; k++ {
						image[x][y][k] = 255
					}
				}
			}
		}
	}

	return image
}

// RandBox draws random partially filled box.
func (image Image) RandBox() Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	xc := rand.Intn(height)
	yc := rand.Intn(width)
	r := 0
	for r == 0 {
		r = height
		if r > width {
			r = width
		}
		r = rand.Intn(r) / 3
	}

	for y := yc - r; y <= yc+r; y++ {
		for x := xc - r; x < xc+r; x++ {
			if x >= 0 && x < width && y >= 0 && y < height {
				if rand.Intn(2) == 1 {
					for k := 0; k < depth; k++ {
						image[x][y][k] = 255
					}
				}
			}
		}
	}

	return image
}

// Circle draws partially filled circle.
func (image Image) Circle(x, y, r int) Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	xc := x
	yc := y

	for y := yc - r; y <= yc+r; y++ {
		for x := xc - r; x < xc+r; x++ {
			if x >= 0 && x < width && y >= 0 && y < height &&
				(x-xc)*(x-xc)+(y-yc)*(y-yc) <= r*r {
				if rand.Intn(2) == 1 {
					for k := 0; k < depth; k++ {
						image[x][y][k] = 255
					}
				}
			}
		}
	}

	return image
}

// Box draws partially filled box.
func (image Image) Box(x, y, halfWidth int) Image {
	rand.Seed(time.Now().UnixNano())

	height := len(image)
	if height == 0 {
		return image
	}

	width := len(image[0])
	if width == 0 {
		return image
	}

	if len(image[0][0]) != 3 {
		return image
	}

	xc := x
	yc := y
	r := halfWidth

	for y := yc - r; y <= yc+r; y++ {
		for x := xc - r; x < xc+r; x++ {
			if x >= 0 && x < width && y >= 0 && y < height {
				if rand.Intn(2) == 1 {
					for k := 0; k < depth; k++ {
						image[x][y][k] = 255
					}
				}
			}
		}
	}

	return image
}

// AdaptiveToneShift shifts the exposure by n counts either towards highlights
// of towards shadows (shiftLeft = true), but maintains the relative dynamic
// range by squeezing the histogram.
func (image Image) AdaptiveToneShift(n uint8, shiftLeft bool) Image {
	latitude := math.MaxUint8 - n

	for i := range image {
		for j := range image[i] {
			for k := range image[i][j] {
				v := float64(image[i][j][k]) / float64(math.MaxUint8) * float64(latitude)
				if shiftLeft {
					image[i][j][k] = uint8(v)
				} else {
					image[i][j][k] = uint8(float64(n) + v)
				}
			}
		}
	}

	return image
}

// RawScaled rescales byte based input image into a float32 based
// data and rescales it in the range [0, 1]
func (image Image) RawScaled() ImageRaw {
	return image.rawScaled(float32(math.MaxUint8))
}

// RawUnscaled rescales byte based input image into a float32 based
// data
func (image Image) RawUnscaled() ImageRaw {
	return image.rawScaled(1)
}

// RawScaled rescales byte based input image into a float32 based
// data and rescales it in the range [0, 1]
func (image Image) rawScaled(scaleFactor float32) ImageRaw {
	h, w, d := image.Size()
	data := make([][][]float32, h)
	for i := range data {
		data[i] = make([][]float32, w)
		for j := range data[i] {
			data[i][j] = make([]float32, d)
			for k := range data[i][j] {
				data[i][j][k] = float32(image[i][j][k]) / scaleFactor
			}
		}
	}
	return data
}

// ApplyMask creates a new image copying only those pixels from input image that
// have a corresponding mask value of true.
func (image Image) ApplyMask(mask Mask) (Image, error) {
	hI, wI, _ := image.Size()
	hM, wM := mask.Size()

	if hI != hM || wI != wM {
		return nil, fmt.Errorf("image and mask are not the same size")
	}

	outImage := NewBlack(hI, wI)

	for i := range image {
		for j := range image[i] {
			for k := range image[i][j] {
				if mask[i][j] {
					outImage[i][j][k] = image[i][j][k]
				}
			}
		}
	}

	return outImage, nil
}

// Copy copies an image.
func (image Image) Copy() Image {
	height := len(image)
	if height == 0 {
		return nil
	}

	width := len(image[0])
	if width == 0 {
		return nil
	}

	depth := len(image[0][0])
	if depth != depth {
		return nil
	}

	im2 := NewBlack(height, width)

	for i := range image {
		for j := range image[i] {
			for k := range image[i][j] {
				im2[i][j][k] = image[i][j][k]
			}
		}
	}

	return im2
}
