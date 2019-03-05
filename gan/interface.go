package gan

import "github.com/sdeoras/comp/proto"

type Operator interface {
	Load(checkpoint *proto.Checkpoint) error
	Infer(count int) error
}
