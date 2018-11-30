package find

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// unique operator constants
	f64          = "myInput"
	uniqueOp_f64 = "unique"
)
