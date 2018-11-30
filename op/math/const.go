package math

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// inputs
	f64 = "myInput"

	// operators
	meanOp = "mean"
	maxOp  = "max"
	minOp  = "min"
	prodOp = "prod"
	sumOp  = "sum"
	varOp  = "var"
)
