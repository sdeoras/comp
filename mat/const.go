package mat

//go:generate ./model.sh

// these constants refer to keys from TF namespace and are defined in
// model.py
const (
	// inputs
	inputBuffer1 = "buff1"
	inputShape1  = "shape1"
	inputBuffer2 = "buff2"
	inputShape2  = "shape2"
	inputShapeB  = "shapeBegin"
	inputShapeS  = "shapeSize"

	// ops
	invOp        = "inv"
	qrOpQ        = "qrdecomp_q"
	qrOpR        = "qrdecomp_r"
	transposeOp  = "transposeOp"
	zerosOp      = "zeros"
	onesOp       = "ones"
	randOp       = "rand"
	randnOp      = "randn"
	mulOp        = "mul"
	mulShapeOp   = "mulShape"
	sliceOp      = "sliceOp"
	sliceShapeOp = "sliceShapeOp"
	reshapeOp    = "reshapeOp"
	tileShapeOp  = "tileShapeOp"
	tileOp       = "tileOp"
)
