package common

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/tensorflow/tensorflow/tensorflow/go"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

const (
	Version = "version"
)

// Op is the operator for TF interface.
type Op struct {
	graph *tensorflow.Graph
	sess  *tensorflow.Session
}

func (op *Op) SetGraph(graph *tensorflow.Graph) {
	op.graph = graph
}

func (op *Op) SetSession(sess *tensorflow.Session) {
	op.sess = sess
}

func (op *Op) GetGraph() *tensorflow.Graph {
	return op.graph
}

func (op *Op) GetSession() *tensorflow.Session {
	return op.sess
}

// Operator defines a generic operator to interface with TF.
type Operator interface {
	// Version returns version info.
	Version() (string, error)
	// GetGraph retrieves graph.
	GetGraph() *tensorflow.Graph
	// SetGraph assigns graph to the operator.
	SetGraph(*tensorflow.Graph)
	// GetSession gets the session.
	GetSession() *tensorflow.Session
	// SetSession assigns session to the operator.
	SetSession(*tensorflow.Session)
}

// InitOperator initializes an operator with a graph and a session.
func InitOperator(op Operator, b64SerializedGraph string, options *tf.SessionOptions) error {
	def, err := base64.StdEncoding.DecodeString(b64SerializedGraph)
	if err != nil {
		return err
	}

	graph := tf.NewGraph()
	if err := graph.Import(def, ""); err != nil {
		return err
	}

	sess, err := tf.NewSession(graph, options)
	if err != nil {
		return err
	}

	op.SetGraph(graph)
	op.SetSession(sess)

	return nil
}

func (op *Op) Version() (string, error) {
	ver, err := op.GetSession().Run(
		nil,
		[]tf.Output{
			op.GetGraph().Operation(Version).Output(0),
		},
		nil)
	if err != nil {
		return "", err
	}

	if len(ver) == 0 {
		return "", fmt.Errorf("invalid output length of zero")
	}

	modelVersion, ok := ver[0].Value().(string)
	if !ok {
		return "", fmt.Errorf("could not get version info as string")
	}

	tfVersion := tf.Version()

	jb, err := json.Marshal(struct {
		TF    string
		Model string
	}{TF: tfVersion, Model: modelVersion})
	return string(jb), err
}

func (op *Op) Close() error {
	return op.GetSession().Close()
}
