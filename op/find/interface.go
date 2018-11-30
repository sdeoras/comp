package find

// Operator defines several operations on float64 data type.
type Operator interface {
	// Unique finds unique elements in x and returns a sorted slice along
	// with the counts of each of those elements.
	Unique(x []float64) ([]float64, []int, error)
}
