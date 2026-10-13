package main

import (
	"flag"
	"log"
	"net/http"
	"time"

	spam "github.com/HxrshRaj/insightflow-ai/nlp_serving/go"
)

func main() {
	modelPath := flag.String("model", "testdata/model_le.bin", "path to MDL1 model")
	addr := flag.String("addr", ":8088", "listen address")
	thr := flag.Float64("threshold", 0.5, "spam decision threshold")
	flag.Parse()
	m, err := spam.Load(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	srv := &http.Server{Addr: *addr, Handler: spam.Handler(m, *thr), ReadHeaderTimeout: 5 * time.Second}
	log.Printf("serving %d features on %s", len(m.Terms), *addr)
	log.Fatal(srv.ListenAndServe())
}
