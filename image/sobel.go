package image

func (sobelRaw SobelRaw) Size() (int, int, int) {
	h := len(sobelRaw)
	if h == 0 {
		return 0, 0, 0
	}

	w := len(sobelRaw[0])
	if w == 0 {
		return h, 0, 0
	}

	return h, w, len(sobelRaw[0][0])
}
