package cloud

import "io"

// FileManager manages file io to cloud
type FileManager interface {
	io.Closer
	// Enumerate lists files that match path.
	// Example input: gs://my-bucket/path/to/files/*.jpg
	Enumerate(path string) ([]string, error)
	// Read reads file contents and returns buffer
	// Example input: gs://my-bucket/path/to/files/earth.jpg
	Read(file string) ([]byte, error)
	// Write writes buffer to file path.
	Write(file string, buffer []byte) error
}

// ReadWriter reads and write files to cloud storage paths.
type ReadWriter interface {
	io.Closer
	// Read reads file contents and returns buffer
	// Example input: gs://my-bucket/path/to/files/earth.jpg
	Read(file string) ([]byte, error)
	// Write writes buffer to file path.
	Write(file string, buffer []byte) error
}

// FileEnumerator lists matching files from cloud path.
type Enumerator interface {
	io.Closer
	// Enumerate lists files that match path.
	// Example input: gs://my-bucket/path/to/files/*.jpg
	Enumerate(path string) ([]string, error)
}

// Reader manages file reading from cloud path.
type Reader interface {
	io.Closer
	// Read reads file contents and returns buffer
	// Example input: gs://my-bucket/path/to/files/earth.jpg
	Read(file string) ([]byte, error)
}

// Writer manages writing byte buffer to cloud path.
type Writer interface {
	io.Closer
	// Write writes buffer to file path.
	Write(file string, buffer []byte) error
}
