package spam

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

type golden struct {
	Text string  `json:"text"`
	P    float64 `json:"p"`
}

func loadGolden(t *testing.T) []golden {
	b, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g []golden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func TestParityWithPython(t *testing.T) {
	g := loadGolden(t)
	for _, f := range []struct{ file, order string }{{"testdata/model_le.bin", "LittleEndian"}, {"testdata/model_be.bin", "BigEndian"}} {
		m, err := Load(f.file)
		if err != nil {
			t.Fatal(err)
		}
		if m.Order.String() != f.order {
			t.Fatalf("%s: detected %s want %s", f.file, m.Order, f.order)
		}
		worst := 0.0
		for _, c := range g {
			d := math.Abs(m.Score(c.Text) - c.P)
			worst = math.Max(worst, d)
			if d > 1e-9 {
				t.Fatalf("%s: %q go=%v py=%v", f.file, c.Text, m.Score(c.Text), c.P)
			}
		}
		t.Logf("%s: %d cases, max |diff| = %.2e", f.file, len(g), worst)
	}
}

func TestBothByteOrdersIdentical(t *testing.T) {
	a, _ := Load("testdata/model_le.bin")
	b, _ := Load("testdata/model_be.bin")
	if len(a.Terms) != len(b.Terms) || a.Bias != b.Bias {
		t.Fatal("headers differ")
	}
	for i := range a.Coef {
		if a.Coef[i] != b.Coef[i] || a.IDF[i] != b.IDF[i] || a.Terms[i] != b.Terms[i] {
			t.Fatalf("index %d differs", i)
		}
	}
}

func TestParseRejectsCorruption(t *testing.T) {
	raw, _ := os.ReadFile("testdata/model_le.bin")
	for name, b := range map[string][]byte{
		"magic":     append([]byte("XXXX"), raw[4:]...),
		"truncated": raw[:len(raw)-8],
		"marker":    append(append(append([]byte{}, raw[:4]...), 9, 9, 9, 9), raw[8:]...),
		"empty":     {},
	} {
		if _, err := Parse(b); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestTokenizer(t *testing.T) {
	got := strings.Join(Analyze("FREE entry! Win a £1000 prize"), "|")
	want := "free|entry|win|1000|prize|free entry|entry win|win 1000|1000 prize"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestHTTP(t *testing.T) {
	m, _ := Load("testdata/model_le.bin")
	h := Handler(m, 0.5)
	post := func(body string) *httptest.ResponseRecorder {
		rr := httptest.NewRecorder()
		h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(body)))
		return rr
	}
	rr := post(`{"text":"WINNER!! Claim your free prize now, call 09061701461"}`)
	var out predictResponse
	_ = json.Unmarshal(rr.Body.Bytes(), &out)
	if rr.Code != 200 || !out.Spam {
		t.Fatalf("spam message: %d %+v", rr.Code, out)
	}
	_ = json.Unmarshal(post(`{"text":"are we still meeting for lunch tomorrow?"}`).Body.Bytes(), &out)
	if out.Spam {
		t.Fatalf("ham flagged: %+v", out)
	}
	if post(`{}`).Code != 400 || post(`nope`).Code != 400 {
		t.Fatal("expected 400")
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/predict", nil))
	if rr.Code != 405 {
		t.Fatal("expected 405")
	}
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rr.Code != 200 {
		t.Fatal("health")
	}
}
