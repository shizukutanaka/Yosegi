package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shizukutanaka/Yosegi/planner"
	"github.com/shizukutanaka/Yosegi/profiles"
)

func writeFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func templateFiles(t *testing.T) (dir, machine, task string) {
	t.Helper()
	dir = t.TempDir()
	var mb, tb bytes.Buffer
	if err := writeTemplate(&mb, "machine"); err != nil {
		t.Fatal(err)
	}
	if err := writeTemplate(&tb, "task"); err != nil {
		t.Fatal(err)
	}
	return dir, writeFile(t, dir, "m.json", mb.String()), writeFile(t, dir, "t.json", tb.String())
}

func TestPlanFromTemplateFilesMatchesInMemoryProfile(t *testing.T) {
	_, mp, tp := templateFiles(t)
	var got bytes.Buffer
	if err := runPlanCmd(&got, []string{"-machine", mp, "-task", tp}); err != nil {
		t.Fatal(err)
	}
	m := profiles.WholePC()
	var want bytes.Buffer
	showTo(&want, &m, "whole-pc", profiles.TranscodeTask(), planner.Options{TopK: 3})
	if got.String() != want.String() {
		t.Fatalf("file round-trip changed the plan:\n got: %s\nwant: %s", got.String(), want.String())
	}
}

func TestPlanBenchReplacesEstimates(t *testing.T) {
	dir, mp, tp := templateFiles(t)
	var before, after bytes.Buffer
	if err := runPlanCmd(&before, []string{"-machine", mp, "-task", tp, "-top", "1"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(before.String(), "plan: gpu0:1.000;gpu0:1.000") {
		t.Fatalf("expected the discrete GPU alone without measurements, got:\n%s", before.String())
	}
	// A measured discrete GPU far slower than its spec sheet must move work off it.
	bp := writeFile(t, dir, "b.json", `[{"unit_id":"gpu0","compute_rate":1e12,"sustained_rate":1e12,"source":"test"}]`)
	if err := runPlanCmd(&after, []string{"-machine", mp, "-task", tp, "-bench", bp, "-top", "1"}); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(after.String(), "plan: gpu0:1.000;gpu0:1.000") || !strings.Contains(after.String(), "igpu:") {
		t.Fatalf("measured rates did not change the decision:\n%s", after.String())
	}
}

func TestPlanRejectsBadInputWithFileName(t *testing.T) {
	dir, mp, tp := templateFiles(t)
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"missing task", []string{"-machine", mp}, "-machine and -task are required"},
		{"typo field", []string{"-machine", writeFile(t, dir, "typo.json", `{"name":"x","unitz":[]}`), "-task", tp}, `typo.json: json: unknown field "unitz"`},
		{"dangling memory", []string{"-machine", writeFile(t, dir, "bad.json",
			`{"name":"x","units":[{"id":"c","kind":"cpu","compute_rate":1,"memory_id":"nope"}],"memories":[]}`), "-task", tp},
			`bad.json: invalid machine: unit c: unknown memory domain "nope"`},
		{"unknown bench unit", []string{"-machine", mp, "-task", tp, "-bench", writeFile(t, dir, "b.json", `[{"unit_id":"gpu9","compute_rate":1}]`)},
			`b.json: unknown unit_id "gpu9"`},
		{"empty task", []string{"-machine", mp, "-task", writeFile(t, dir, "e.json", `{"name":"e","stages":[],"repeats":1}`)},
			"e.json: task needs at least one stage"},
	}
	for _, c := range cases {
		err := runPlanCmd(&bytes.Buffer{}, c.args)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want containing %q", c.name, err, c.want)
		}
	}
}

func TestTemplateRejectsUnknownKind(t *testing.T) {
	if err := writeTemplate(&bytes.Buffer{}, "gpu"); err == nil {
		t.Fatal("expected error for unknown template kind")
	}
}
