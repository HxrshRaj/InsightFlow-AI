// Package spam serves the MDL1 TF-IDF + logistic regression model using only the standard library.
package spam

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

const (
	magic   = "MDL1"
	marker  = 0x01020304
	version = 1
)

// Model is a TF-IDF + logistic regression text classifier.
type Model struct {
	Terms []string
	Index map[string]int
	IDF   []float64
	Coef  []float64
	Bias  float64
	// Order is the byte order the file was written in (detected, not assumed).
	Order binary.ByteOrder
}

// Load reads an MDL1 file from disk.
func Load(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Parse decodes MDL1 bytes. The byte order is detected from the marker word, so the same
// code is correct on little-endian (x86, ARM) and big-endian (s390x) hosts and files.
func Parse(b []byte) (*Model, error) {
	if len(b) < 24 || string(b[:4]) != magic {
		return nil, errors.New("not an MDL1 file")
	}
	var order binary.ByteOrder
	switch binary.LittleEndian.Uint32(b[4:8]) {
	case marker:
		order = binary.LittleEndian
	case 0x04030201:
		order = binary.BigEndian
	default:
		return nil, errors.New("corrupt byte-order marker")
	}
	if v := order.Uint32(b[8:12]); v != version {
		return nil, fmt.Errorf("unsupported version %d", v)
	}
	n := int(order.Uint32(b[12:16]))
	m := &Model{Order: order, Bias: math.Float64frombits(order.Uint64(b[16:24]))}
	m.Terms = make([]string, n)
	m.Index = make(map[string]int, n)
	pos := 24
	for i := 0; i < n; i++ {
		if pos+2 > len(b) {
			return nil, errors.New("truncated term table")
		}
		l := int(order.Uint16(b[pos:]))
		pos += 2
		if pos+l > len(b) {
			return nil, errors.New("truncated term")
		}
		m.Terms[i] = string(b[pos : pos+l])
		m.Index[m.Terms[i]] = i
		pos += l
	}
	if pos+16*n != len(b) {
		return nil, errors.New("trailing or missing bytes")
	}
	m.IDF = make([]float64, n)
	m.Coef = make([]float64, n)
	for i := 0; i < n; i++ {
		m.IDF[i] = math.Float64frombits(order.Uint64(b[pos+8*i:]))
	}
	pos += 8 * n
	for i := 0; i < n; i++ {
		m.Coef[i] = math.Float64frombits(order.Uint64(b[pos+8*i:]))
	}
	return m, nil
}

// Score returns P(spam) for text, mirroring python/baseline.py:score.
func (m *Model) Score(text string) float64 {
	counts := map[int]int{}
	for _, t := range Analyze(text) {
		if i, ok := m.Index[t]; ok {
			counts[i]++
		}
	}
	var sq float64
	w := make(map[int]float64, len(counts))
	for i, c := range counts {
		v := float64(c) * m.IDF[i]
		w[i] = v
		sq += v * v
	}
	z := m.Bias
	if norm := math.Sqrt(sq); norm > 0 {
		for i, v := range w {
			z += m.Coef[i] * v / norm
		}
	}
	if z >= 0 {
		return 1 / (1 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1 + e)
}
