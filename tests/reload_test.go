package koanf_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/knadh/koanf/maps"
	"github.com/knadh/koanf/parsers/json"
	"github.com/knadh/koanf/providers/confmap"
	"github.com/knadh/koanf/providers/rawbytes"
	"github.com/knadh/koanf/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errProvider lets a test fail Read or ReadBytes.
type errProvider struct {
	readErr   error
	bytesErr  error
	bytes     []byte
	readCalls atomic.Int32
}

func (p *errProvider) Read() (map[string]any, error) {
	p.readCalls.Add(1)
	if p.readErr != nil {
		return nil, p.readErr
	}
	return nil, nil
}

func (p *errProvider) ReadBytes() ([]byte, error) {
	p.readCalls.Add(1)
	if p.bytesErr != nil {
		return nil, p.bytesErr
	}
	return p.bytes, nil
}

// blockProvider blocks Read until release is closed or ctxDone is closed. It is
// used to observe cancellation of an in-flight source.
type blockProvider struct {
	release chan struct{}
	started chan struct{}
}

func (p *blockProvider) Read() (map[string]any, error) {
	close(p.started)
	<-p.release
	return map[string]any{"late": true}, nil
}

func (p *blockProvider) ReadBytes() ([]byte, error) { return nil, errors.New("unused") }

func source(name string, mp map[string]any) koanf.Source {
	// Providers hand koanf a nested map, so unflatten delimited keys the same
	// way a file or env provider would.
	return koanf.Source{Name: name, Provider: confmap.Provider(mp, ".")}
}

func TestReloadAllOrNothing(t *testing.T) {
	k := koanf.New(".")
	require.NoError(t, k.Load(confmap.Provider(map[string]any{"keep": "old"}, "."), nil))

	boom := errors.New("provider unavailable")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": 1, "keep": "new"}),
		{Name: "remote", Provider: &errProvider{bytesErr: boom}, Parser: json.Parser()},
		source("never", map[string]any{"b": 2}),
	})

	require.Error(t, err)
	var re *koanf.ReloadError
	require.ErrorAs(t, err, &re)
	assert.Equal(t, "remote", re.Source)
	assert.ErrorIs(t, err, boom)
	assert.Empty(t, re.Key, "a read failure has no key path")

	// Nothing from the valid first source may have leaked into the live map.
	assert.Equal(t, "old", k.String("keep"))
	assert.False(t, k.Exists("a"))
	assert.False(t, k.Exists("b"))
	assert.Equal(t, uint64(0), k.Generation())
}

func TestReloadParseFailurePreservesConfig(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": "one", "b": "two"}),
	})
	require.NoError(t, err)

	_, err = k.Reload(context.Background(), []koanf.Source{
		{Name: "bad", Provider: rawbytes.Provider([]byte(`{not json`)), Parser: json.Parser()},
	})
	require.Error(t, err)
	assert.Equal(t, "one", k.String("a"), "published config must be preserved")
	assert.Equal(t, uint64(1), k.Generation())
}

func TestReloadStrictMergeFailureReportsSourceAndKey(t *testing.T) {
	k := koanf.NewWithConf(koanf.Conf{Delim: ".", StrictMerge: true})

	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"config": map[string]any{"value": "old"}}),
		source("override", map[string]any{"config": map[string]any{"value": 123}}),
	})
	require.Error(t, err)

	var re *koanf.ReloadError
	require.ErrorAs(t, err, &re)
	assert.Equal(t, "override", re.Source)
	assert.Equal(t, "config.value", re.Key)
	assert.Contains(t, err.Error(), "incorrect types at key config.value")
	assert.False(t, k.Exists("config.value"), "a failed candidate must not be published")
}

func TestReloadStrictMergeWithinCandidateIsAtomic(t *testing.T) {
	k := koanf.NewWithConf(koanf.Conf{Delim: ".", StrictMerge: true})
	require.NoError(t, k.Load(confmap.Provider(map[string]any{"stable": "value"}, "."), nil))

	// Merging on top of the published config makes the new candidate clash
	// with the existing key, and that clash must abort the whole candidate.
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"good": "yes"}),
		source("clash", map[string]any{"stable": 42}),
	}, koanf.WithRetainExisting())
	require.Error(t, err)
	assert.Equal(t, "value", k.String("stable"), "existing config must be preserved")
	assert.False(t, k.Exists("good"))
}

func TestReloadSourcePriorityAndProvenance(t *testing.T) {
	k := koanf.New(".")
	res, err := k.Reload(context.Background(), []koanf.Source{
		source("defaults", map[string]any{
			"db":   map[string]any{"host": "localhost", "port": 5432},
			"only": "defaults",
		}),
		source("file", map[string]any{
			"db": map[string]any{"host": "db.internal"},
		}),
		source("env", map[string]any{
			"db": map[string]any{"port": 6543},
		}),
	})
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.True(t, res.Published)
	assert.Equal(t, []string{"defaults", "file", "env"}, res.Sources)
	assert.Equal(t, "db.internal", k.String("db.host"))
	assert.Equal(t, int64(6543), k.Int64("db.port"))

	src, ok := k.SourceOf("db.host")
	assert.True(t, ok)
	assert.Equal(t, "file", src)

	src, ok = k.SourceOf("db.port")
	assert.True(t, ok)
	assert.Equal(t, "env", src, "the last source that sets a key owns it")

	src, ok = k.SourceOf("only")
	assert.True(t, ok)
	assert.Equal(t, "defaults", src)

	// Every key of the candidate has provenance, and it matches KeySources.
	assert.Equal(t, res.KeySources, k.KeySources())
	for _, key := range res.Keys {
		_, ok := k.SourceOf(key)
		assert.True(t, ok, "missing provenance for %s", key)
	}
}

func TestReloadProvenanceDropsShadowedNestedKeys(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("defaults", map[string]any{
			"parent": map[string]any{"child": map[string]any{"key": 1}},
		}),
		source("override", map[string]any{
			"parent": map[string]any{"child": "scalar"},
		}),
	})
	require.NoError(t, err)

	assert.True(t, k.Exists("parent.child"))
	src, ok := k.SourceOf("parent.child")
	assert.True(t, ok)
	assert.Equal(t, "override", src)

	_, ok = k.SourceOf("parent.child.key")
	assert.False(t, ok, "a key shadowed by a scalar must not keep provenance")
}

func TestReloadRemovesDeletedKeys(t *testing.T) {
	k := koanf.New(".")
	require.NoError(t, k.Load(confmap.Provider(map[string]any{"legacy": "present"}, "."), nil))

	res, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"current": true}),
	})
	require.NoError(t, err)
	assert.False(t, k.Exists("legacy"), "a full reload reflects deletions")
	assert.True(t, k.Exists("current"))

	var deleted *koanf.Change
	for i := range res.Changes {
		if res.Changes[i].Key == "legacy" {
			deleted = &res.Changes[i]
		}
	}
	require.NotNil(t, deleted)
	assert.Equal(t, "present", deleted.Old)
	assert.Nil(t, deleted.New)
	assert.Empty(t, deleted.Source)
}

func TestReloadGenerationRejectsStaleCandidate(t *testing.T) {
	k := koanf.New(".")

	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "new"}),
	}, koanf.WithGeneration(5))
	require.NoError(t, err)
	assert.Equal(t, uint64(5), k.Generation())

	_, err = k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "old"}),
	}, koanf.WithGeneration(4))
	require.Error(t, err)
	assert.True(t, errors.Is(err, koanf.ErrStaleReload))

	var stale *koanf.StaleReloadError
	require.ErrorAs(t, err, &stale)
	assert.Equal(t, uint64(4), stale.Candidate)
	assert.Equal(t, uint64(5), stale.Published)

	assert.Equal(t, "new", k.String("value"), "stale candidate must not clobber the newer config")
	assert.Equal(t, uint64(5), k.Generation())
}

func TestReloadStaleCandidateStillReportsCandidateView(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "new"}),
	}, koanf.WithGeneration(2))
	require.NoError(t, err)

	res, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "old"}),
	}, koanf.WithGeneration(1))
	require.Error(t, err)
	require.NotNil(t, res, "a stale result should still describe the discarded candidate")
	assert.False(t, res.Published)
	assert.Equal(t, []string{"value"}, res.Keys)
	assert.Equal(t, "base", res.KeySources["value"])
	assert.Empty(t, res.Changes)
}

func TestReloadRepublishIsIdempotent(t *testing.T) {
	k := koanf.New(".")

	var calls atomic.Int32
	var changedKeys []string
	cancel := k.OnReload(func(res koanf.ReloadResult) {
		calls.Add(1)
		for _, c := range res.Changes {
			changedKeys = append(changedKeys, c.Key)
		}
	})
	defer cancel()

	srcs := []koanf.Source{source("base", map[string]any{"a": 1})}
	first, err := k.Reload(context.Background(), srcs, koanf.WithGeneration(7))
	require.NoError(t, err)
	assert.True(t, first.Published)
	assert.Equal(t, uint64(7), k.Generation())
	assert.EqualValues(t, 1, calls.Load())
	require.Equal(t, []string{"a"}, changedKeys)

	second, err := k.Reload(context.Background(), srcs, koanf.WithGeneration(7))
	require.NoError(t, err)
	assert.False(t, second.Published, "republishing the current generation is a no-op")
	assert.Empty(t, second.Changes)
	assert.EqualValues(t, 1, calls.Load(), "no callback may fire for a no-op publish")
	assert.Equal(t, []string{"a"}, changedKeys, "each change is notified exactly once")
}

func TestReloadNoChangeFiresNoCallback(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": 1}),
	})
	require.NoError(t, err)

	var calls atomic.Int32
	cancel := k.OnReload(func(koanf.ReloadResult) { calls.Add(1) })
	defer cancel()

	res, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": 1}),
	})
	require.NoError(t, err)
	assert.True(t, res.Published)
	assert.Empty(t, res.Changes)
	assert.EqualValues(t, 0, calls.Load(), "identical content must not notify changes")
}

func TestReloadCallbacksRunAfterPublishAndSeeNewConfig(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "old"}),
	})
	require.NoError(t, err)

	var (
		mu      sync.Mutex
		seen    []string
		visible string
		gen     uint64
	)
	cancel := k.OnReload(func(res koanf.ReloadResult) {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range res.Changes {
			seen = append(seen, c.Key)
		}
		visible = k.String("value")
		gen = k.Generation()
	})
	defer cancel()

	_, err = k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "new"}),
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"value"}, seen)
	assert.Equal(t, "new", visible, "the new config must be visible in the callback")
	assert.Equal(t, uint64(2), gen)
}

func TestReloadCancellationLeavesConfigUntouched(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "stable"}),
	})
	require.NoError(t, err)
	before := k.Snapshot()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"value": "changed"}),
	})
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled))

	after := k.Snapshot()
	assert.Equal(t, before.Generation, after.Generation)
	assert.Equal(t, before.Raw, after.Raw)
	assert.Equal(t, before.Sources, after.Sources)
	assert.Equal(t, "stable", k.String("value"))
}

func TestReloadCancellationMidway(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": "stable"}),
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	bp := &blockProvider{release: make(chan struct{}), started: make(chan struct{})}

	done := make(chan error, 1)
	go func() {
		_, err := k.Reload(ctx, []koanf.Source{
			source("first", map[string]any{"partial": true}),
			{Name: "blocking", Provider: bp},
		})
		done <- err
	}()

	<-bp.started
	// Wait until the second source is in flight, then cancel. The in-flight
	// read cannot be interrupted, so release it; the reload must observe the
	// cancellation before publishing.
	cancel()
	close(bp.release)

	select {
	case err := <-done:
		require.Error(t, err)
		assert.True(t, errors.Is(err, context.Canceled))
	case <-time.After(5 * time.Second):
		t.Fatal("canceled reload did not return")
	}

	assert.False(t, k.Exists("partial"))
	assert.False(t, k.Exists("late"))
	assert.Equal(t, "stable", k.String("value"))
}

func TestReloadNilProvider(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{{Name: "missing"}})
	require.Error(t, err)
	assert.True(t, errors.Is(err, koanf.ErrNilProvider))

	var re *koanf.ReloadError
	require.ErrorAs(t, err, &re)
	assert.Equal(t, "missing", re.Source)
}

func TestReloadUnnamedSourcesGetStableNames(t *testing.T) {
	k := koanf.New(".")
	res, err := k.Reload(context.Background(), []koanf.Source{
		{Provider: confmap.Provider(map[string]any{"a": 1}, "")},
		{Provider: confmap.Provider(map[string]any{"b": 2}, "")},
	})
	require.NoError(t, err)
	require.Len(t, res.Sources, 2)
	assert.NotEqual(t, res.Sources[0], res.Sources[1],
		"unnamed sources of the same type must be distinguishable")
	assert.Equal(t, map[string]string{
		"a": res.Sources[0],
		"b": res.Sources[1],
	}, res.KeySources)
}

func TestReloadPromptCancellationDoesNotBumpGeneration(t *testing.T) {
	k := koanf.New(".")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	for i := 0; i < 5; i++ {
		_, err := k.Reload(ctx, []koanf.Source{
			source("base", map[string]any{"a": i}),
		})
		require.Error(t, err)
	}
	assert.Equal(t, uint64(0), k.Generation())
	// Auto-allocated generations may have been reserved, but publishing an
	// auto candidate afterwards must still succeed.
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": "ok"}),
	})
	require.NoError(t, err)
	assert.Equal(t, "ok", k.String("a"))
}

func TestSnapshotIsConsistentAcrossReload(t *testing.T) {
	k := koanf.New(".")
	ctx := context.Background()

	// Two keys that must always come from the same generation. Each reload
	// publishes a complete pair, so a snapshot may never observe a mix.
	for i := 0; i < 32; i++ {
		_, err := k.Reload(ctx, []koanf.Source{
			source("left", map[string]any{"pair.left": i}),
			source("right", map[string]any{"pair.right": i}),
		})
		require.NoError(t, err)
	}

	s := k.Snapshot()
	assert.Equal(t, s.All["pair.left"], s.All["pair.right"])
	assert.Equal(t, s.Generation, k.Generation())
}

func TestConcurrentReloadsNeverExposePartialConfig(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping race test in short mode")
	}

	k := koanf.New(".")
	ctx := context.Background()

	// Seed a complete pair.
	_, err := k.Reload(ctx, []koanf.Source{
		source("left", map[string]any{"pair.left": -1}),
		source("right", map[string]any{"pair.right": -1}),
	})
	require.NoError(t, err)

	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Readers assert that they never see a half-applied pair.
	var bad atomic.Int32
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				s := k.Snapshot()
				l, okL := s.All["pair.left"]
				r, okR := s.All["pair.right"]
				if !okL || !okR || fmt.Sprint(l) != fmt.Sprint(r) {
					bad.Add(1)
				}
				if l == nil || r == nil {
					bad.Add(1)
				}
			}
		}()
	}

	// Writers publish complete pairs with monotonically increasing generations.
	var writeErr atomic.Value
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for n := 0; n < 50; n++ {
				gen := uint64(id*1000 + n + 1)
				_, err := k.Reload(ctx, []koanf.Source{
					source("left", map[string]any{"pair.left": gen}),
					source("right", map[string]any{"pair.right": gen}),
				}, koanf.WithGeneration(gen))
				if err != nil && !errors.Is(err, koanf.ErrStaleReload) {
					writeErr.Store(err)
					return
				}
			}
		}(i)
	}

	time.Sleep(100 * time.Millisecond)
	close(stop)
	wg.Wait()

	if v := writeErr.Load(); v != nil {
		t.Fatalf("reload failed: %v", v)
	}
	assert.Zero(t, bad.Load(), "readers observed a partially published configuration")
}

func TestReloadRollbackRestoresPublishedState(t *testing.T) {
	k := koanf.New(".")
	ctx := context.Background()

	first, err := k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"value": "one", "only": 1}),
	})
	require.NoError(t, err)
	gen1 := first.Generation

	_, err = k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"value": "two", "extra": 2}),
	})
	require.NoError(t, err)
	assert.Equal(t, "two", k.String("value"))

	var calls atomic.Int32
	var got []koanf.Change
	cancel := k.OnReload(func(res koanf.ReloadResult) {
		calls.Add(1)
		got = append(got, res.Changes...)
	})
	defer cancel()

	res, err := k.Rollback(gen1)
	require.NoError(t, err)
	assert.True(t, res.Published)
	assert.Equal(t, uint64(3), res.Generation, "rollback publishes a fresh generation")
	assert.Equal(t, "one", k.String("value"))
	assert.EqualValues(t, 1, k.Get("only"))
	assert.False(t, k.Exists("extra"), "keys added after the target generation are removed")
	assert.EqualValues(t, 1, calls.Load())

	src, ok := k.SourceOf("value")
	assert.True(t, ok)
	assert.Equal(t, "base", src)

	// Rolling back to the generation that is currently published is a no-op.
	noop, err := k.Rollback(res.Generation)
	require.NoError(t, err)
	assert.False(t, noop.Published)
	assert.EqualValues(t, 1, calls.Load())

	// Repeating the rollback to a still-retained generation is a no-op and
	// must not publish or notify again.
	repeat, err := k.Rollback(gen1)
	require.NoError(t, err)
	assert.False(t, repeat.Published)
	assert.EqualValues(t, 1, calls.Load())

	// A generation that was never published fails cleanly.
	_, err = k.Rollback(999)
	require.Error(t, err)
	assert.True(t, errors.Is(err, koanf.ErrUnknownGeneration))
	assert.Equal(t, "one", k.String("value"))
}

func TestRollbackKeepsGenerationsMonotonic(t *testing.T) {
	k := koanf.New(".")
	ctx := context.Background()

	r1, err := k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"v": 1}),
	}, koanf.WithGeneration(10))
	require.NoError(t, err)

	_, err = k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"v": 2}),
	}, koanf.WithGeneration(20))
	require.NoError(t, err)

	rb, err := k.Rollback(r1.Generation)
	require.NoError(t, err)
	assert.Greater(t, rb.Generation, uint64(20))

	// A candidate that was newer than the rolled-back-to generation but
	// older than the rollback's fresh generation must be rejected.
	_, err = k.Reload(ctx, []koanf.Source{
		source("base", map[string]any{"v": 3}),
	}, koanf.WithGeneration(20))
	assert.True(t, errors.Is(err, koanf.ErrStaleReload))
	assert.EqualValues(t, 1, k.Get("v"))
}

func TestLoadInvalidatesReloadProvenance(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": 1}),
	})
	require.NoError(t, err)
	_, ok := k.SourceOf("value")
	require.True(t, ok)

	// Plain Load keeps its immediate-visibility behaviour and invalidates the
	// transactional provenance, which no longer describes the live map.
	require.NoError(t, k.Load(confmap.Provider(map[string]any{"value": 2}, "."), nil))
	assert.EqualValues(t, 2, k.Get("value"))
	_, ok = k.SourceOf("value")
	assert.False(t, ok)
	assert.Equal(t, uint64(1), k.Generation(), "generations stay monotonic across plain Load")
}

func TestSetAndDeleteInvalidateReloadProvenance(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": 1}),
	})
	require.NoError(t, err)

	require.NoError(t, k.Set("other", 2))
	_, ok := k.SourceOf("value")
	assert.False(t, ok)

	_, err = k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"value": 1}),
	})
	require.NoError(t, err)
	_, ok = k.SourceOf("value")
	require.True(t, ok)

	k.Delete("value")
	_, ok = k.SourceOf("value")
	assert.False(t, ok)
}

func TestReloadProviderMapIsNotAliased(t *testing.T) {
	k := koanf.New(".")
	mp := map[string]any{"nested": map[string]any{"key": "original"}}

	_, err := k.Reload(context.Background(), []koanf.Source{
		{Name: "mutable", Provider: confmap.Provider(mp, "")},
	})
	require.NoError(t, err)

	// Mutating the provider's source map after it was read must not change the
	// published configuration.
	mp["nested"].(map[string]any)["key"] = "mutated"
	assert.Equal(t, "original", k.String("nested.key"))

	// Mutating the published configuration must not change the provider's map.
	require.NoError(t, k.Set("nested.key", "changed"))
	assert.Equal(t, "mutated", mp["nested"].(map[string]any)["key"])
}

func TestReloadUnsubscribeIsIdempotent(t *testing.T) {
	k := koanf.New(".")
	var calls atomic.Int32
	cancel := k.OnReload(func(koanf.ReloadResult) { calls.Add(1) })

	cancel()
	cancel() // must not panic or unregister another callback

	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"a": 1}),
	})
	require.NoError(t, err)
	assert.EqualValues(t, 0, calls.Load())
}

func TestReloadRetainExistingKeepsUnsuppliedKeys(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{"keep": "me", "shared": "old"}),
	})
	require.NoError(t, err)

	_, err = k.Reload(context.Background(), []koanf.Source{
		source("overrides", map[string]any{"shared": "new"}),
	}, koanf.WithRetainExisting())
	require.NoError(t, err)

	assert.Equal(t, "me", k.String("keep"), "unsupplied keys survive retain mode")
	assert.Equal(t, "new", k.String("shared"))

	// A retained key keeps the provenance it had when it was published, while
	// a key supplied again points at the new source.
	src, ok := k.SourceOf("keep")
	assert.True(t, ok)
	assert.Equal(t, "base", src)
	src, ok = k.SourceOf("shared")
	assert.True(t, ok)
	assert.Equal(t, "overrides", src)
}

func TestReloadChangesAreReportedOnceAndSorted(t *testing.T) {
	k := koanf.New(".")
	_, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{
			"b": 1,
			"a": 1,
			"c": 1,
		}),
	})
	require.NoError(t, err)

	res, err := k.Reload(context.Background(), []koanf.Source{
		source("base", map[string]any{
			"b": 2,
			"a": 1, // unchanged, must not be reported
			"d": 4, // added
		}),
	})
	require.NoError(t, err)

	keys := make([]string, 0, len(res.Changes))
	for _, c := range res.Changes {
		keys = append(keys, c.Key)
	}
	assert.Equal(t, []string{"b", "c", "d"}, keys, "changes must be sorted and reported once")

	byKey := map[string]koanf.Change{}
	for _, c := range res.Changes {
		byKey[c.Key] = c
	}
	assert.EqualValues(t, 1, byKey["b"].Old)
	assert.EqualValues(t, 2, byKey["b"].New)
	assert.EqualValues(t, 1, byKey["c"].Old)
	assert.Nil(t, byKey["c"].New)
	assert.Nil(t, byKey["d"].Old)
	assert.EqualValues(t, 4, byKey["d"].New)
	assert.Equal(t, "base", byKey["d"].Source)
}

func TestOnReloadConcurrentRegistrationAndReload(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping race test in short mode")
	}

	k := koanf.New(".")
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for n := 0; n < 25; n++ {
				cancel := k.OnReload(func(koanf.ReloadResult) {})
				cancel()
			}
		}(i)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for n := 0; n < 50; n++ {
			_, err := k.Reload(ctx, []koanf.Source{
				source("base", map[string]any{"n": n}),
			})
			if err != nil && !errors.Is(err, koanf.ErrStaleReload) {
				t.Errorf("reload failed: %v", err)
				return
			}
		}
	}()

	wg.Wait()
}

func TestReloadDuringConcurrentLoadKeepsAtomicity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping race test in short mode")
	}

	// A non-transactional Load invalidates provenance, but a reader must never
	// see a torn map: each Reload publishes a whole pair.
	k := koanf.New(".")
	ctx := context.Background()
	require.NoError(t, k.Set("pair.left", -1))
	require.NoError(t, k.Set("pair.right", -1))

	stop := make(chan struct{})
	var readers, writers sync.WaitGroup
	var bad atomic.Int32

	for i := 0; i < 4; i++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				s := k.Snapshot()
				l, okL := s.All["pair.left"]
				r, okR := s.All["pair.right"]
				if !okL || !okR || l != r {
					bad.Add(1)
				}
			}
		}()
	}

	writers.Add(1)
	go func() {
		defer writers.Done()
		for n := 0; n < 200; n++ {
			_, err := k.Reload(ctx, []koanf.Source{
				source("a", map[string]any{"pair.left": n}),
				source("b", map[string]any{"pair.right": n}),
			}, koanf.WithRetainExisting())
			if err != nil {
				t.Errorf("reload failed: %v", err)
				return
			}
		}
	}()

	writers.Wait()
	close(stop)
	readers.Wait()
	assert.Zero(t, bad.Load())
}

func TestReloadStrictMergeMatchesMergeStrict(t *testing.T) {
	// The candidate merge must agree with maps.MergeStrict on which keys
	// conflict and what the resulting map looks like.
	cases := []struct {
		name    string
		first   map[string]any
		second  map[string]any
		wantErr bool
	}{
		{
			name:  "same scalar types merge",
			first: map[string]any{"a": 1, "b": "x"},
			second: map[string]any{
				"a": 2, "c": true,
			},
		},
		{
			name:    "scalar type conflict",
			first:   map[string]any{"a": 1},
			second:  map[string]any{"a": "string"},
			wantErr: true,
		},
		{
			name:    "map over scalar",
			first:   map[string]any{"a": 1},
			second:  map[string]any{"a": map[string]any{"b": 2}},
			wantErr: true,
		},
		{
			name:    "scalar over map",
			first:   map[string]any{"a": map[string]any{"b": 2}},
			second:  map[string]any{"a": 1},
			wantErr: true,
		},
		{
			name: "nested maps merge",
			first: map[string]any{
				"parent": map[string]any{"a": 1, "child": map[string]any{"x": "s"}},
			},
			second: map[string]any{
				"parent": map[string]any{"b": 2, "child": map[string]any{"y": "t"}},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			k := koanf.NewWithConf(koanf.Conf{Delim: ".", StrictMerge: true})
			_, err := k.Reload(context.Background(), []koanf.Source{
				source("first", c.first),
				source("second", c.second),
			})

			// Compare against applying maps.MergeStrict directly.
			direct := maps.Copy(c.first)
			directErr := maps.MergeStrict(maps.Copy(c.second), direct)

			if c.wantErr {
				require.Error(t, err)
				require.Error(t, directErr, "test case must conflict in maps.MergeStrict too")
				return
			}
			require.NoError(t, err)
			require.NoError(t, directErr)
			wantFlat, _ := maps.Flatten(direct, nil, ".")
			assert.Equal(t, wantFlat, k.All())
		})
	}
}
