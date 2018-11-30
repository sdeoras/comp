package common

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"

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

// InitUsingB64Graph initializes an operator with a graph and a session.
func InitUsingB64Graph(op Operator, b64SerializedGraph string, options *tf.SessionOptions) error {
	graphDef, err := base64.StdEncoding.DecodeString(b64SerializedGraph)
	if err != nil {
		return err
	}

	return InitOperator(op, graphDef, options)
}

// InitUsingGraphReader reads graph using an io reader and initializes the operator.
func InitUsingGraphReader(op Operator, graphReader io.Reader, options *tf.SessionOptions) error {
	graphDef, err := ioutil.ReadAll(graphReader)
	if err != nil {
		return err
	}

	return InitOperator(op, graphDef, options)
}

// InitUsingGraphFile reads graph from a file and initializes the operator.
func InitUsingGraphFile(op Operator, graphFile string, options *tf.SessionOptions) error {
	graphDef, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return err
	}

	return InitOperator(op, graphDef, options)
}

// InitOperator is initializes operator using graphDef passes as a byte buffer.
func InitOperator(op Operator, graphDef []byte, options *tf.SessionOptions) error {
	graph := tf.NewGraph()
	if err := graph.Import(graphDef, ""); err != nil {
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

// Version prints version info of the graph def and the tensorflow library being used.
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

// Close closes the tensorflow session.
func (op *Op) Close() error {
	return op.GetSession().Close()
}
