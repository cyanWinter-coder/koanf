package koanf

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/knadh/koanf/maps"
	"github.com/mitchellh/copystructure"
)

var (
	// ErrNilProvider is returned when a reload source does not have a Provider.
	ErrNilProvider = errors.New("koanf: reload source has a nil provider")

	// ErrStaleReload is returned (wrapped in *StaleReloadError) when a candidate
	// carries a generation older than the one already published. The candidate
	// is discarded and the published configuration is left untouched.
	ErrStaleReload = errors.New("koanf: reload generation is older than the published generation")

	// ErrUnknownGeneration is returned by Rollback when the requested
	// generation is not in the retained history.
	ErrUnknownGeneration = errors.New("koanf: generation is not in the rollback history")
)

// maxReloadHistory is the number of published transactional states retained
// for Rollback. Older states are evicted. A non-transactional mutation such as
// Load, Merge, Set or Delete clears the history entirely because it breaks the
// linear chain of published generations.
const maxReloadHistory = 8

// publishedState is a deep copy of one published transactional configuration,
// kept so Rollback can restore it without aliasing live maps.
type publishedState struct {
	confMap    map[string]any
	keyMap     KeyMap
	keySources map[string]string
	names      []string
	generation uint64
}

// Source describes one provider in a transactional reload. Sources are applied
// in the order they are declared, so a later Source overrides keys of an
// earlier one, exactly like successive Load calls.
type Source struct {
	// Name identifies the provider in errors, ReloadResult.Changes and in the
	// per-key provenance returned by SourceOf/KeySources. If empty, the
	// provider's Go type is used, with a positional suffix when several
	// unnamed sources share the same type.
	Name string

	// Provider reads the configuration. It must be non-nil.
	Provider Provider

	// Parser parses the raw bytes returned by Provider. When nil, Provider.Read
	// is called instead, matching Load's behaviour.
	Parser Parser
}

// Change describes a single flattened key that changed when a candidate was
// published. Old is nil for a newly added key and New is nil for a deleted one.
type Change struct {
	Key    string
	Old    any
	New    any
	Source string // final source of New; empty for deletions
}

// ReloadResult describes the outcome of a transactional reload.
type ReloadResult struct {
	// Generation is the generation carried by the candidate.
	Generation uint64

	// PreviousGeneration is the generation that was published before this
	// reload. It is only meaningful when Published is true.
	PreviousGeneration uint64

	// Published reports whether the candidate was published. It is false for
	// an idempotent republish of the current generation and for a discarded
	// stale candidate.
	Published bool

	// Sources lists the resolved source names in declaration (merge) order.
	Sources []string

	// Keys lists the flattened keys of the candidate, sorted alphabetically.
	// It is populated when the candidate is valid, including stale candidates.
	Keys []string

	// KeySources maps every flattened key of the candidate to the source that
	// supplied its final value.
	KeySources map[string]string

	// Changes lists every changed key exactly once, sorted by key. It is empty
	// when nothing changed or when the candidate was not published.
	Changes []Change
}

// ReloadError reports a failed transactional reload. The published
// configuration is never modified when a ReloadError is returned, apart from
// *StaleReloadError which reports a candidate that lost against a newer one.
type ReloadError struct {
	// Source is the resolved name of the source that failed.
	Source string

	// Provider is the provider that failed, when known.
	Provider Provider

	// Key is the flattened key path involved in the failure. It is empty when
	// the failure happened before merging, for instance a read or parse error.
	Key string

	// Err is the underlying cause. Use errors.Is/errors.As against it via
	// errors.Unwrap.
	Err error
}

func (e *ReloadError) Error() string {
	var b strings.Builder

	b.WriteString("koanf: reload failed")
	if e.Source != "" {
		fmt.Fprintf(&b, " for source %q", e.Source)
	}
	if e.Key != "" {
		fmt.Fprintf(&b, " at key %q", e.Key)
	}
	if e.Err != nil {
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	}
	return b.String()
}

// Unwrap returns the underlying cause.
func (e *ReloadError) Unwrap() error {
	return e.Err
}

// StaleReloadError reports that a candidate was discarded because a newer
// generation had already been published. Use errors.Is(err, ErrStaleReload)
// to detect it.
type StaleReloadError struct {
	Candidate uint64
	Published uint64
}

func (e *StaleReloadError) Error() string {
	return fmt.Sprintf("koanf: reload generation %d is older than published generation %d",
		e.Candidate, e.Published)
}

// Unwrap returns ErrStaleReload.
func (e *StaleReloadError) Unwrap() error {
	return ErrStaleReload
}

// ConfigSnapshot is a consistent copy of the published configuration. Every
// field belongs to the same generation, which makes it suitable for a request
// that needs to read more than one key without observing a concurrent reload.
type ConfigSnapshot struct {
	Generation uint64
	Raw        map[string]any
	All        map[string]any
	KeyMap     KeyMap
	Sources    map[string]string

	// SourceNames is the declared merge order of the last published
	// transactional reload.
	SourceNames []string
}

type reloadOptions struct {
	generation uint64
	retain     bool
}

// ReloadOption modifies the behaviour of Reload.
type ReloadOption func(*reloadOptions)

// WithGeneration sets the monotonically increasing generation of a reload
// candidate. Generations identify candidates: a candidate is published only if
// its generation is newer than the published one. Republishing the current
// generation is an idempotent no-op, and an older generation is rejected with
// *StaleReloadError, so a slow candidate can never overwrite a newer config.
//
// A zero generation is the default and makes Reload allocate the next
// generation itself (the first allocated generation is 1).
func WithGeneration(gen uint64) ReloadOption {
	return func(o *reloadOptions) {
		o.generation = gen
	}
}

// WithRetainExisting makes Reload start the candidate from a copy of the
// published configuration instead of an empty one, then merge the declared
// sources on top. This preserves keys that no source supplies. A retained key
// keeps the provenance recorded by an earlier transactional reload, if any;
// otherwise it has no recorded source.
//
// It is useful when a provider supplies only a subset of the configuration,
// for instance a single feature's overrides.
//
// The default rebuilds the configuration from the declared sources only: keys
// that no source supplies are removed and reported as deletions, and every
// published key has provenance.
func WithRetainExisting() ReloadOption {
	return func(o *reloadOptions) {
		o.retain = true
	}
}

// Snapshot returns a consistent copy of the currently published configuration.
// Unlike calling Get/All/Raw separately, all fields come from the same
// generation even if a Reload publishes concurrently.
func (ko *Koanf) Snapshot() ConfigSnapshot {
	ko.mu.RLock()
	defer ko.mu.RUnlock()

	s := ConfigSnapshot{
		Generation:  ko.generation,
		Raw:         maps.Copy(ko.confMap),
		All:         maps.Copy(ko.confMapFlat),
		KeyMap:      make(KeyMap, len(ko.keyMap)),
		Sources:     make(map[string]string, len(ko.keySources)),
		SourceNames: append([]string(nil), ko.sourceNames...),
	}
	for key, parts := range ko.keyMap {
		s.KeyMap[key] = append([]string(nil), parts...)
	}
	for key, src := range ko.keySources {
		s.Sources[key] = src
	}
	return s
}

// Generation returns the generation of the most recently published
// transactional state. It is 0 until a transactional reload publishes
// successfully. Non-transactional mutations such as Load, Merge, Set and
// Delete clear provenance and rollback history but keep the generation counter
// monotonic, so an in-flight reload candidate cannot outrank a newer
// generation.
func (ko *Koanf) Generation() uint64 {
	ko.mu.RLock()
	gen := ko.generation
	ko.mu.RUnlock()
	return gen
}

// SourceOf returns the source that supplied the final value of a flattened key
// in the last published transactional reload. It returns false when the key is
// absent or its provenance is unknown, for instance because Load, Merge, Set or
// Delete modified the configuration after the last reload.
func (ko *Koanf) SourceOf(path string) (string, bool) {
	ko.mu.RLock()
	src, ok := ko.keySources[path]
	ko.mu.RUnlock()
	return src, ok
}

// KeySources returns a copy of the flattened key to source mapping produced by
// the last published transactional reload.
func (ko *Koanf) KeySources() map[string]string {
	ko.mu.RLock()
	out := make(map[string]string, len(ko.keySources))
	for key, src := range ko.keySources {
		out[key] = src
	}
	ko.mu.RUnlock()
	return out
}

// OnReload registers fn to be notified after a transactional reload has been
// published. fn is only called when the publish changed at least one key, and
// it receives every change exactly once. Callbacks are invoked after the new
// configuration is visible, on the goroutine that called Reload, without
// holding the config lock; they may therefore be concurrent with another
// publish. Callbacks must not mutate the result.
//
// The returned function unregisters the callback and is safe to call more than
// once.
func (ko *Koanf) OnReload(fn func(ReloadResult)) (cancel func()) {
	if fn == nil {
		return func() {}
	}

	ko.mu.Lock()
	if ko.reloadCallbacks == nil {
		ko.reloadCallbacks = make(map[uint64]func(ReloadResult))
	}
	id := ko.nextCallbackID
	ko.nextCallbackID++
	ko.reloadCallbacks[id] = fn
	ko.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			ko.mu.Lock()
			delete(ko.reloadCallbacks, id)
			ko.mu.Unlock()
		})
	}
}

// Reload transactionally loads all sources into a private candidate copy of
// the configuration and publishes it only when every source was read, parsed
// and merged successfully. Readers therefore observe either the previous
// complete configuration or the new complete one, never a partially merged
// map. The single-provider Load method is unchanged and keeps its immediate
// visibility semantics.
//
// Sources are applied in declaration order: later sources override earlier
// ones. When Conf.StrictMerge is set, a type conflict between sources fails the
// reload and is reported as a *ReloadError naming the source, the key path and
// the underlying cause.
//
// By default the candidate is rebuilt from the declared sources only, so keys
// that no source supplies are removed and every published key has provenance.
// WithRetainExisting seeds the candidate from the published configuration
// instead, which keeps unsupplied keys and their previous provenance.
//
// Candidates are ordered by generation. A candidate whose generation is older
// than the published one is discarded and reported as a *StaleReloadError;
// republishing the current generation is an idempotent no-op. Cancellation is
// observed before each source is read and before publishing, so a canceled
// reload leaves the published configuration untouched.
//
// The per-key source of the final value is recorded and exposed through
// SourceOf, KeySources and ConfigSnapshot. Plain Load, Merge, Set and Delete
// calls invalidate that provenance because they modify the configuration
// outside a transactional publish.
func (ko *Koanf) Reload(ctx context.Context, sources []Source, opts ...ReloadOption) (*ReloadResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	o := new(reloadOptions)
	for _, opt := range opts {
		opt(o)
	}

	if err := ctx.Err(); err != nil {
		return nil, &ReloadError{Err: err}
	}

	gen := o.generation
	if gen == 0 {
		gen = ko.nextReloadGeneration()
	}

	names := resolveSourceNames(sources)
	c := &reloadCandidate{
		keySources: make(map[string]string),
		generation: gen,
		names:      names,
	}

	if o.retain {
		// Seed from a deep copy of the published configuration so merging
		// into the candidate cannot touch live maps. Retained keys keep the
		// provenance recorded by an earlier transactional reload, if any.
		ko.mu.RLock()
		c.confMap = maps.Copy(ko.confMap)
		for key, src := range ko.keySources {
			c.keySources[key] = src
		}
		ko.mu.RUnlock()
		if c.confMap == nil {
			c.confMap = make(map[string]any)
		}
	} else {
		c.confMap = make(map[string]any)
	}

	for i, src := range sources {
		name := names[i]

		if src.Provider == nil {
			return nil, &ReloadError{Source: name, Err: ErrNilProvider}
		}
		if err := ctx.Err(); err != nil {
			return nil, &ReloadError{Source: name, Provider: src.Provider, Err: err}
		}

		mp, err := readReloadSource(src)
		if err != nil {
			return nil, &ReloadError{Source: name, Provider: src.Provider, Err: err}
		}
		if err := ctx.Err(); err != nil {
			return nil, &ReloadError{Source: name, Provider: src.Provider, Err: err}
		}

		// Copy the provider's map so merging into the candidate cannot mutate
		// data owned by the provider, and so the published configuration does
		// not alias maps that a provider may reuse on its next read.
		mp = cloneReloadMap(mp)
		maps.IntfaceKeysToStrings(mp)

		if ko.conf.StrictMerge {
			if key, err := mergeStrictKey(mp, c.confMap, nil, ko.conf.Delim); err != nil {
				return nil, &ReloadError{
					Source:   name,
					Provider: src.Provider,
					Key:      key,
					Err:      err,
				}
			}
		} else {
			maps.Merge(mp, c.confMap)
		}

		// Record the source of every key this provider supplied. A later
		// source overwrites both the value and its provenance.
		flat, _ := maps.Flatten(mp, nil, ko.conf.Delim)
		for key := range flat {
			c.keySources[key] = name
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, &ReloadError{Err: err}
	}

	// Freeze the candidate into the same representations the live instance
	// uses, so publishing is a single swap of complete maps.
	c.confMapFlat, c.keyMap = maps.Flatten(c.confMap, nil, ko.conf.Delim)
	c.keyMap = populateKeyParts(c.keyMap, ko.conf.Delim)

	// A key may have been shadowed by a later source that replaced a map with
	// a scalar. Drop provenance for keys that do not exist in the final map.
	for key := range c.keySources {
		if _, ok := c.confMapFlat[key]; !ok {
			delete(c.keySources, key)
		}
	}

	return ko.publishReload(c)
}

type reloadCandidate struct {
	confMap     map[string]any
	confMapFlat map[string]any
	keyMap      KeyMap
	keySources  map[string]string
	generation  uint64
	names       []string
}

// readReloadSource reads a source using Read or ReadBytes+Parser, mirroring
// Load's behaviour.
func readReloadSource(src Source) (map[string]any, error) {
	if src.Parser == nil {
		mp, err := src.Provider.Read()
		if err != nil {
			return nil, err
		}
		if mp == nil {
			mp = make(map[string]any)
		}
		return mp, nil
	}

	b, err := src.Provider.ReadBytes()
	if err != nil {
		return nil, err
	}
	mp, err := src.Parser.Unmarshal(b)
	if err != nil {
		return nil, err
	}
	if mp == nil {
		mp = make(map[string]any)
	}
	return mp, nil
}

// publishReload atomically swaps a fully built candidate into the live
// instance. It is the only point where a reload can change published state.
func (ko *Koanf) publishReload(c *reloadCandidate) (*ReloadResult, error) {
	ko.mu.Lock()

	prev := ko.generation
	if c.generation < prev {
		ko.mu.Unlock()
		return &ReloadResult{
				Generation:         c.generation,
				PreviousGeneration: prev,
				Published:          false,
				Sources:            append([]string(nil), c.names...),
				Keys:               sortedFlatKeys(c.confMapFlat),
				KeySources:         copyStringMap(c.keySources),
			}, &StaleReloadError{
				Candidate: c.generation,
				Published: prev,
			}
	}

	// Republishing an already published generation is an idempotent no-op.
	// It must not swap state, fire callbacks or report changes.
	if c.generation == prev {
		res := ko.noopReloadResultLocked(prev)
		ko.mu.Unlock()
		return res, nil
	}

	changes := diffFlatMaps(ko.confMapFlat, c.confMapFlat, c.keySources)

	// Retain the outgoing state so Rollback can restore it. The candidate's
	// maps are frozen (only ever read after this point), so storing them
	// directly is safe; the outgoing maps are frozen too because they are
	// about to be detached from the live instance.
	ko.recordHistoryLocked(&publishedState{
		confMap:    ko.confMap,
		keyMap:     ko.keyMap,
		keySources: ko.keySources,
		names:      ko.sourceNames,
		generation: prev,
	})

	// Swap all published views together while holding the write lock. A
	// reader can only see the old complete set or the new complete set.
	ko.confMap = c.confMap
	ko.confMapFlat = c.confMapFlat
	ko.keyMap = c.keyMap
	ko.keySources = c.keySources
	ko.sourceNames = c.names
	ko.generation = c.generation
	if ko.nextGeneration < c.generation {
		ko.nextGeneration = c.generation
	}

	callbacks := make([]func(ReloadResult), 0, len(ko.reloadCallbacks))
	for _, fn := range ko.reloadCallbacks {
		callbacks = append(callbacks, fn)
	}
	ko.mu.Unlock()

	res := &ReloadResult{
		Generation:         c.generation,
		PreviousGeneration: prev,
		Published:          true,
		Sources:            append([]string(nil), c.names...),
		Keys:               sortedFlatKeys(c.confMapFlat),
		KeySources:         copyStringMap(c.keySources),
		Changes:            changes,
	}

	// Change callbacks fire only after the new configuration is published,
	// and only when something actually changed. Each Change appears once.
	if len(changes) > 0 {
		for _, fn := range callbacks {
			// Give each callback its own slice header so one callback
			// appending to it cannot affect another.
			r := *res
			r.Changes = append([]Change(nil), res.Changes...)
			r.Keys = append([]string(nil), res.Keys...)
			fn(r)
		}
	}

	return res, nil
}

// nextReloadGeneration reserves a new auto-assigned generation. Auto-assigned
// generations always move forward from the published generation, even if a
// caller supplied a larger explicit generation earlier.
func (ko *Koanf) nextReloadGeneration() uint64 {
	ko.mu.Lock()
	defer ko.mu.Unlock()

	if ko.nextGeneration < ko.generation {
		ko.nextGeneration = ko.generation
	}
	ko.nextGeneration++
	return ko.nextGeneration
}

// recordHistoryLocked retains a published state for Rollback. The caller must
// hold the write lock. States whose generation is no longer in strict
// ascending order with the live one are not useful and are dropped.
func (ko *Koanf) recordHistoryLocked(ps *publishedState) {
	if ko.history == nil {
		ko.history = make(map[uint64]*publishedState, maxReloadHistory)
	}
	ko.history[ps.generation] = ps

	// Evict the oldest retained generation(s) once the budget is exceeded.
	for len(ko.history) > maxReloadHistory {
		var oldest uint64
		first := true
		for gen := range ko.history {
			if first || gen < oldest {
				oldest = gen
				first = false
			}
		}
		delete(ko.history, oldest)
	}
}

// invalidateReloadStateLocked clears provenance and the rollback history after
// a non-transactional mutation. The caller must hold the write lock. The
// published configuration is left untouched.
//
// The generation counters are deliberately preserved: keeping them monotonic
// means an in-flight reload candidate that was started before the mutation
// cannot win against a newer published generation later on.
func (ko *Koanf) invalidateReloadStateLocked() {
	ko.keySources = nil
	ko.sourceNames = nil
	ko.history = nil
}

// Rollback atomically restores the configuration published under generation
// gen, which must be present in the retained history. The restored state is
// installed as the new published configuration with a fresh, higher
// generation, so later reload candidates must still carry a newer generation
// to be published. Provenance returns to the state recorded with that
// generation.
//
// Rollback is idempotent in the same sense as Reload: rolling back to the
// generation that is currently published, or to a generation whose content is
// already published, is a no-op that changes nothing and fires no callbacks. A
// generation that is no longer retained yields ErrUnknownGeneration without
// changing anything. Change callbacks fire once for the keys that actually
// change.
func (ko *Koanf) Rollback(gen uint64) (*ReloadResult, error) {
	ko.mu.Lock()

	if gen == ko.generation {
		res := ko.noopReloadResultLocked(gen)
		ko.mu.Unlock()
		return res, nil
	}

	ps, ok := ko.history[gen]
	if !ok {
		verr := fmt.Errorf("%w: %d (published %d)", ErrUnknownGeneration, gen, ko.generation)
		ko.mu.Unlock()
		return nil, &ReloadError{Err: verr}
	}

	// Restore deep copies so the history entry stays intact for a later
	// rollback and so readers never alias state that is changing. The frozen
	// map is flattened again so the restored views match the live shape.
	restored := &publishedState{
		confMap:    maps.Copy(ps.confMap),
		keySources: copyStringMap(ps.keySources),
		names:      append([]string(nil), ps.names...),
	}
	restoredFlat, restoredKeyMap := maps.Flatten(restored.confMap, nil, ko.conf.Delim)
	restored.keyMap = populateKeyParts(restoredKeyMap, ko.conf.Delim)

	prev := ko.generation
	changes := diffFlatMaps(ko.confMapFlat, restoredFlat, restored.keySources)

	// Restoring a state that is already published is an idempotent no-op.
	// This makes a repeated Rollback(gen) cheap and side-effect free while the
	// target generation remains retained.
	if len(changes) == 0 {
		res := ko.noopReloadResultLocked(prev)
		res.Sources = append([]string(nil), restored.names...)
		ko.mu.Unlock()
		return res, nil
	}

	newGen := ko.nextGeneration + 1
	if newGen <= prev {
		newGen = prev + 1
	}
	restored.generation = newGen

	// Keep the state we are leaving behind so a rollback can be undone by
	// rolling forward to it while it is still retained. The target entry is
	// deliberately kept too, so a repeated rollback stays a no-op.
	ko.recordHistoryLocked(&publishedState{
		confMap:    ko.confMap,
		keyMap:     ko.keyMap,
		keySources: ko.keySources,
		names:      ko.sourceNames,
		generation: prev,
	})

	ko.confMap = restored.confMap
	ko.confMapFlat = restoredFlat
	ko.keyMap = restored.keyMap
	ko.keySources = restored.keySources
	ko.sourceNames = restored.names
	ko.generation = newGen
	ko.nextGeneration = newGen

	callbacks := make([]func(ReloadResult), 0, len(ko.reloadCallbacks))
	for _, fn := range ko.reloadCallbacks {
		callbacks = append(callbacks, fn)
	}
	ko.mu.Unlock()

	res := &ReloadResult{
		Generation:         newGen,
		PreviousGeneration: prev,
		Published:          true,
		Sources:            append([]string(nil), restored.names...),
		Keys:               sortedFlatKeys(restoredFlat),
		KeySources:         copyStringMap(restored.keySources),
		Changes:            changes,
	}

	for _, fn := range callbacks {
		r := *res
		r.Changes = append([]Change(nil), res.Changes...)
		r.Keys = append([]string(nil), res.Keys...)
		fn(r)
	}

	return res, nil
}

// noopReloadResultLocked builds the result of an idempotent, no-op publish.
// The caller must hold the lock.
func (ko *Koanf) noopReloadResultLocked(gen uint64) *ReloadResult {
	return &ReloadResult{
		Generation:         gen,
		PreviousGeneration: gen,
		Published:          false,
		Sources:            append([]string(nil), ko.sourceNames...),
		Keys:               sortedFlatKeys(ko.confMapFlat),
		KeySources:         copyStringMap(ko.keySources),
	}
}

func copyKeyMap(m KeyMap) KeyMap {
	out := make(KeyMap, len(m))
	for key, parts := range m {
		out[key] = append([]string(nil), parts...)
	}
	return out
}

// resolveSourceNames resolves explicit and default source names. Unnamed
// sources are named after their provider's Go type, disambiguated with a
// positional suffix when the same type is used more than once.
func resolveSourceNames(sources []Source) []string {
	out := make([]string, len(sources))

	// Explicit names are reserved up front so a generated name cannot
	// collide with one the caller chose.
	used := make(map[string]int, len(sources))
	for _, src := range sources {
		if src.Name != "" {
			used[src.Name]++
		}
	}

	for i, src := range sources {
		if src.Name != "" {
			out[i] = src.Name
			continue
		}

		base := "<nil>"
		if src.Provider != nil {
			base = fmt.Sprintf("%T", src.Provider)
		}

		name := base
		if used[name] > 0 {
			name = base + "#" + strconv.Itoa(i)
		}
		used[name]++
		out[i] = name
	}
	return out
}

// mergeStrictKey mirrors maps.MergeStrict but returns the conflicting key path
// so a ReloadError can name it. The error message keeps the same wording as
// maps.MergeStrict.
func mergeStrictKey(src, dst map[string]any, parts []string, delim string) (string, error) {
	for key, val := range src {
		path := make([]string, len(parts)+1)
		copy(path, parts)
		path[len(parts)] = key

		cur, exists := dst[key]
		if !exists {
			dst[key] = val
			continue
		}

		fullKey := joinKeyPath(path, delim)
		srcMap, srcIsMap := val.(map[string]any)
		if !srcIsMap {
			if reflect.TypeOf(cur) == reflect.TypeOf(val) {
				dst[key] = val
				continue
			}
			return fullKey, fmt.Errorf("incorrect types at key %v, type %T != %T", fullKey, cur, val)
		}

		dstMap, dstIsMap := cur.(map[string]any)
		if !dstIsMap {
			return fullKey, fmt.Errorf("incorrect types at key %v, type %T != %T", fullKey, cur, val)
		}
		if conflict, err := mergeStrictKey(srcMap, dstMap, path, delim); err != nil {
			return conflict, err
		}
	}
	return "", nil
}

func joinKeyPath(parts []string, delim string) string {
	if delim == "" {
		return strings.Join(parts, ".")
	}
	return strings.Join(parts, delim)
}

// diffFlatMaps returns every added, changed and deleted flattened key exactly
// once, sorted by key.
func diffFlatMaps(old, cur map[string]any, sources map[string]string) []Change {
	keys := make([]string, 0, len(old)+len(cur))
	seen := make(map[string]struct{}, len(old)+len(cur))
	for key := range old {
		seen[key] = struct{}{}
		keys = append(keys, key)
	}
	for key := range cur {
		if _, ok := seen[key]; !ok {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)

	var out []Change
	for _, key := range keys {
		oldVal, hadOld := old[key]
		newVal, hasNew := cur[key]

		switch {
		case hadOld && hasNew:
			if reflect.DeepEqual(oldVal, newVal) {
				continue
			}
			out = append(out, Change{
				Key:    key,
				Old:    copyReloadValue(oldVal),
				New:    copyReloadValue(newVal),
				Source: sources[key],
			})
		case hasNew:
			out = append(out, Change{
				Key:    key,
				New:    copyReloadValue(newVal),
				Source: sources[key],
			})
		default:
			out = append(out, Change{
				Key: key,
				Old: copyReloadValue(oldVal),
			})
		}
	}
	return out
}

func sortedFlatKeys(flat map[string]any) []string {
	out := make([]string, 0, len(flat))
	for key := range flat {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func copyStringMap(mp map[string]string) map[string]string {
	if mp == nil {
		return nil
	}
	out := make(map[string]string, len(mp))
	for key, val := range mp {
		out[key] = val
	}
	return out
}

// cloneReloadMap deep-copies a provider's map. It falls back to a structural
// copy for values that copystructure cannot traverse (for example values with
// unexported state) instead of dropping them.
func cloneReloadMap(mp map[string]any) map[string]any {
	out, err := copystructure.Copy(&mp)
	if err == nil {
		if p, ok := out.(*map[string]any); ok {
			return *p
		}
	}
	return cloneReloadMapFallback(mp)
}

func cloneReloadMapFallback(mp map[string]any) map[string]any {
	out := make(map[string]any, len(mp))
	for key, val := range mp {
		out[key] = cloneReloadValueFallback(val)
	}
	return out
}

func cloneReloadValueFallback(val any) any {
	switch v := val.(type) {
	case map[string]any:
		return cloneReloadMapFallback(v)
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = cloneReloadValueFallback(item)
		}
		return out
	}
	return val
}

// copyReloadValue copies a value exposed through a Change. Scalar values are
// returned as-is.
func copyReloadValue(val any) any {
	switch val.(type) {
	case nil, bool, string,
		int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64:
		return val
	}

	out, err := copystructure.Copy(&val)
	if err != nil {
		return val
	}
	if p, ok := out.(*any); ok {
		return *p
	}
	return val
}
