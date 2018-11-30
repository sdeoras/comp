package mat

// Matrix is an interface defining following methods.
type Matrix interface {
	// Numel computes number of elements in matrix.
	Numel() int64
	// GetSize returns the size of matrix.
	GetSize() []int
	// GetRaw returns raw buffer of the matrix
	GetRaw() []float64
	// IsEqual compares two matrices for their content and size and only returns true
	// if both checks are true.
	IsEqual(x *Mat) bool
	// IsSameSize compares two matrices for their size.
	IsSameSize(x *Mat) bool
}

// Numel computes number of elements in matrix.
func (m *Mat) Numel() int64 {
	p := int64(1)
	for _, v := range m.Size {
		p *= v
	}

	return p
}

// GetSize returns the size of matrix.
func (m *Mat) GetSize() []int {
	s := make([]int, len(m.Size))
	for i, v := range m.Size {
		s[i] = int(v)
	}
	return s
}

// GetRaw returns raw buffer of the matrix
func (m *Mat) GetRaw() []float64 {
	return m.Buf
}

// IsEqual compares two matrices for their content and size and only returns true
// if both checks are true.
func (m *Mat) IsEqual(x *Mat) bool {
	if m == nil && x == nil {
		return true
	}

	if m != nil && x != nil && len(m.Buf) == len(x.Buf) && len(m.Size) == len(x.Size) {
		for i := range m.Buf {
			if m.Buf[i] != x.Buf[i] {
				return false
			}
		}

		for i := range m.Size {
			if m.Size[i] != x.Size[i] {
				return false
			}
		}

		return true
	}

	return false
}

// IsSameSize compares two matrices for their size.
func (m *Mat) IsSameSize(x *Mat) bool {
	if m == nil && x == nil {
		return true
	}

	if m != nil && x != nil && len(m.Size) == len(x.Size) {
		for i := range m.Size {
			if m.Size[i] != x.Size[i] {
				return false
			}
		}

		return true
	}

	return false
}
