package spam

import (
	"encoding/json"
	"net/http"
)

type predictRequest struct {
	Text string `json:"text"`
}

type predictResponse struct {
	Spam        bool    `json:"spam"`
	Probability float64 `json:"probability"`
	Threshold   float64 `json:"threshold"`
}

// Handler returns the HTTP API: GET /health and POST /predict {"text": "..."}.
func Handler(m *Model, threshold float64) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status": "ok", "features": len(m.Terms), "model_byte_order": order(m),
		})
	})
	mux.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
		var req predictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "body must be {\"text\": \"...\"}"})
			return
		}
		p := m.Score(req.Text)
		writeJSON(w, http.StatusOK, predictResponse{Spam: p >= threshold, Probability: p, Threshold: threshold})
	})
	return mux
}

func order(m *Model) string {
	if m.Order.String() == "BigEndian" {
		return "big"
	}
	return "little"
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
