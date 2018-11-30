package layers

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// unique operator constants
	myInput      = "myInput"
	myInputSlice = "myInputSlice"

	flattenOp      = "flattenImage"
	flattenSliceOp = "flattenSlice"
)
