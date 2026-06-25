// Copyright 2019+ Klaus Post. All rights reserved.
// License information can be found in the LICENSE file.
//
// resume.go is a soci-snapshotter addition (NOT upstream klauspost/compress).
// It exposes the minimal mid-frame decoder state -- the back-reference window,
// the three recent match offsets, and the active entropy tables -- so a caller
// can snapshot decoder state at an interior block boundary of a single,
// unmodified zstd frame and resume decoding from there into a fresh decoder.
//
// Upstream exposes no mid-frame state: history, frameDec and blockDec are all
// unexported, so this must live in-package. The decode itself reuses the
// existing, battle-tested blockDec/history machinery (DecodeAll/sync path); the
// only thing added here is the ability to stop at a chosen block and to seed a
// fresh history with a previously captured state.

package zstd

import (
	"github.com/klauspost/compress/huff0"
	"github.com/klauspost/compress/zstd/internal/xxhash"
)

// ResumeState is the decoder state captured at a zstd block boundary. It is
// sufficient to resume decoding the following blocks of the same frame without
// re-decoding the blocks before the boundary.
//
// Window and RecentOffsets are always required. The entropy tables (huff,
// seqDecs) are required ONLY when the following block reuses them -- Treeless
// literals reuse the Huffman table, and Repeat sequence modes reuse the FSE
// tables. At a boundary whose following block redefines all of its entropy
// tables, the entropy fields are irrelevant and a caller may seed a state that
// carries only Window and RecentOffsets.
//
// The entropy fields are unexported live decoder objects: they survive being
// passed straight from CaptureResumeState into ResumeDecode within one process,
// but they are not serializable. A persistent checkpoint that lands on an
// entropy-reusing boundary must serialize the tables separately (e.g. via the
// zstd dictionary format); a checkpoint placed on an entropy-clean boundary
// needs only Window and RecentOffsets, both trivially serializable.
type ResumeState struct {
	// Window is the decoded output preceding the boundary. Matches in the
	// resumed blocks reference back into it (bounded by the frame window size).
	Window []byte

	// RecentOffsets are the three repeated match offsets in effect at the
	// boundary. At frame start these are {1, 4, 8}.
	RecentOffsets [3]int

	huff       *huff0.Scratch
	seqDecs    sequenceDecs
	hasEntropy bool
}

// runRawBlocks drives the block decoder over a raw block stream (a sequence of
// zstd blocks with NO frame header) using the DecodeAll sync path. h must be a
// freshly seeded history (window already in h.b, recentOffsets/entropy set).
// expectedOut is the exact number of NEW output bytes the decoded blocks
// produce (callers know it from the block/offset map); it sizes the output
// budget. If nBlocks > 0 the decode stops after nBlocks blocks; otherwise it
// runs until the Last block. It returns the number of blocks actually decoded.
func runRawBlocks(h *history, blocks []byte, windowSize uint64, nBlocks, expectedOut int) (int, error) {
	// Mirror runDecoder's DecodeAll setup so the sync decode path is happy:
	// ignoreBuffer == 0 makes blockDec.decodeBuf use h.b directly as the output
	// accumulator, so match back-references reach into the seeded window.
	h.ignoreBuffer = 0
	// Budget for the sync sequence decoder. A non-zero maxSyncLen lets the fast
	// path engage; it is decremented as bytes are produced, so it is the total
	// expected NEW output of this call.
	h.decoders.maxSyncLen = uint64(expectedOut)

	var bd blockDec
	bd.lowMem = false
	br := byteBuf(blocks)
	n := 0
	for {
		if err := bd.reset(&br, windowSize); err != nil {
			return n, err
		}
		if err := bd.decodeBuf(h); err != nil {
			return n, err
		}
		n++
		if bd.Last || (nBlocks > 0 && n >= nBlocks) {
			break
		}
	}
	return n, nil
}

// seedHistory builds a fresh history seeded with the given window, recent
// offsets and (optionally) entropy tables, with output capacity for expectedOut
// new bytes on top of the window.
func seedHistory(window []byte, windowSize uint64, recent [3]int, huff *huff0.Scratch, seqDecs *sequenceDecs, expectedOut int) *history {
	// Fill the package-global predefined FSE tables (fsePredef) before any block
	// decode. They are guarded by a process-global sync.Once and are normally
	// initialized as a side effect of NewReader/NewWriter -- but soci's resume
	// path constructs blockDec/history directly and never calls NewReader, so on
	// a cold process (an index build before any other klauspost decode) the
	// tables would be all-zero. A Compressed_Block that selects the Predefined
	// Compression_Mode for its sequences would then decode against zero tables,
	// produce a bogus match offset, and be rejected -- making indexing of a
	// valid, unmodified zstd layer depend on unrelated prior decoder activity in
	// the process (F-024). initPredefined is idempotent (sync.Once), so calling
	// it on every seed makes the resume path self-sufficient at negligible cost.
	initPredefined()

	var fd frameDec
	h := &fd.history
	h.reset()
	h.windowSize = int(windowSize)
	if h.windowSize <= 0 {
		// Defensive: a zero window would reject every match. Use the largest of
		// the seed window / expected output so back-references resolve.
		h.windowSize = len(window) + expectedOut
	}

	capacity := len(window) + expectedOut + maxCompressedBlockSizeAlloc + compressedBlockOverAlloc
	h.b = make([]byte, 0, capacity)
	h.b = append(h.b, window...)

	if recent == ([3]int{}) {
		recent = [3]int{1, 4, 8}
	}
	h.recentOffsets = recent
	if huff != nil {
		h.huffTree = huff
	}
	if seqDecs != nil {
		h.decoders = *seqDecs
	}
	return h
}

// CaptureResumeState decodes the first nBlocks blocks of a raw block stream
// (the frame body with NO frame header) starting from frame-initial decoder
// state, and returns the decoder state at that boundary. prefixOut is the exact
// number of bytes the first nBlocks blocks decode to (the uncompressed offset of
// the boundary), as produced e.g. by a block/offset map.
//
// windowSize is the frame's window size. nBlocks must be >= 1 and at most the
// number of blocks in the stream; passing the full block count captures the
// end-of-frame state.
func CaptureResumeState(blocks []byte, windowSize uint64, nBlocks, prefixOut int) (ResumeState, error) {
	h := seedHistory(nil, windowSize, [3]int{}, nil, nil, prefixOut)
	if _, err := runRawBlocks(h, blocks, windowSize, nBlocks, prefixOut); err != nil {
		return ResumeState{}, err
	}
	window := make([]byte, len(h.b))
	copy(window, h.b)
	return ResumeState{
		Window:        window,
		RecentOffsets: h.recentOffsets,
		huff:          h.huffTree,
		seqDecs:       h.decoders,
		hasEntropy:    true,
	}, nil
}

// BlockUncompOffsets decodes a raw block stream (a frame body with NO frame
// header) ONCE from frame-initial decoder state and returns the cumulative
// uncompressed output length at every block boundary: a slice whose element 0 is
// 0 (frame start) and whose element k is the number of bytes the first k blocks
// decode to. The final element is the frame's total decompressed size, so the
// slice has length (number of blocks + 1).
//
// It is the O(N) replacement for decoding each prefix [0:k] independently. A
// single forward decode records each boundary's offset as it crosses it, so the
// total work is one decode of the frame rather than N partial decodes -- the
// block-boundary offset map costs the same as decoding the frame once.
//
// windowSize is the frame's window size. The output buffer is reserved from the
// COMPRESSED bytes present (len(blocks)) and GROWS from there as the decode
// produces output -- maxSyncLen is left 0 so the sync decoder grows on demand
// instead of trusting a pre-sized budget. The allocation therefore tracks the
// REAL decoded size and is never inflated by a frame that declares a huge
// Frame_Content_Size over little real content: the caller deliberately does not
// pass the declared size, so a crafted layer cannot drive an unbounded make()
// (F-019). The decode runs to the stream's Last block; blocks must therefore end
// on that block.
//
// It also returns the zstd content checksum of the decoded output: the low 32
// bits of its XXH64, computed over the full frame content this single decode pass
// already produced (h.b), at no extra decode cost. A caller that knows the frame
// declares a trailing content checksum (the zstd CLI sets one by default) can
// compare this against the stored 4-byte trailer to reject a corrupt layer at
// build time, matching what the production streaming decoder verifies -- rather
// than indexing a frame whose checksum the streaming decoder will later fail,
// which aborts the whole image build (F-012 class). The value is meaningful only
// when the frame carries a checksum; for a checksum-less frame the caller ignores
// it.
//
// maxOut bounds the cumulative decoded output this pass will buffer. Unlike the
// streaming decoder, which flushes each block and never holds the whole frame,
// this pass must retain the entire decode in h.b to report every boundary offset
// and the content checksum, so a small but highly compressible (or crafted)
// frame -- a decompression bomb -- would otherwise grow h.b without bound and OOM
// the indexer at build time. When maxOut > 0 and the running output exceeds it,
// the decode stops and returns ErrDecoderSizeExceeded (overshooting by at most one
// block, whose decoded size the zstd format caps), so the caller can benign-skip
// an unindexably large layer instead of being driven into an unbounded make().
// maxOut == 0 disables the bound (decode to completion regardless of size).
func BlockUncompOffsets(blocks []byte, windowSize uint64, maxOut uint64) ([]int, uint32, error) {
	// Stream the decode (window-trimmed, no whole-frame retention): callers that
	// want only the offset map and checksum must not pay the F-029 memory cost of
	// materializing the full frame. BlockUncompOffsetsAndContent remains for the
	// (now test-only) callers that genuinely need the decoded bytes resident.
	return BlockUncompStreaming(blocks, windowSize, maxOut, nil, nil)
}

// BlockUncompOffsetsAndContent is BlockUncompOffsets that ALSO returns the full
// decoded frame content this single forward pass already materializes in h.b --
// the bytes whose cumulative lengths ARE the returned offsets. A caller that needs
// per-span uncompressed digests slices them straight out of this content in a
// pure-Go pass (content[offsets[spanStart]:offsets[spanEnd]]) instead of decoding
// the frame a second time, so the offset map and the digest table both come from
// ONE authoritative decode. The returned slice has length offsets[len(offsets)-1]
// (the frame's total decompressed size) and is the decoder's own buffer: the
// caller treats it as read-only and must not retain it past its own use, since the
// underlying array is sized by the decode and bounded by maxOut exactly as the
// offsets are. BlockUncompOffsets is the thin offsets-only wrapper for callers that
// do not need the content.
func BlockUncompOffsetsAndContent(blocks []byte, windowSize uint64, maxOut uint64) ([]int, uint32, []byte, error) {
	// Seed the initial output capacity from the compressed input length: a sane
	// head start (output is at least the Raw/RLE-expanded size of these bytes)
	// that is bounded by what is on the wire, never by any declared length.
	h := seedHistory(nil, windowSize, [3]int{}, nil, nil, len(blocks))
	h.ignoreBuffer = 0
	// maxSyncLen stays 0: the sync sequence decoder then grows its output buffer
	// on demand (the same path a frame with no declared size takes) rather than
	// relying on a pre-sized budget, so an undersized initial reservation is
	// corrected by growth, never a hard "output bigger than max block size" error.

	// offsets[0] == 0 (frame start); offsets[k] is filled after block k-1
	// decodes, so it holds the cumulative output length at that boundary.
	offsets := []int{0}
	var bd blockDec
	bd.lowMem = false
	br := byteBuf(blocks)
	for {
		if err := bd.reset(&br, windowSize); err != nil {
			return nil, 0, nil, err
		}
		if err := bd.decodeBuf(h); err != nil {
			return nil, 0, nil, err
		}
		// h.b holds the full decode so far (it is never trimmed to the window in
		// this mode). Bounding it here caps the indexer's build-time memory: a
		// frame whose real output exceeds maxOut is unindexably large and the
		// caller skips it, rather than this loop growing h.b toward OOM. The check
		// follows the block decode, so the overshoot is one block's worth.
		if maxOut > 0 && uint64(len(h.b)) > maxOut {
			return nil, 0, nil, ErrDecoderSizeExceeded
		}
		offsets = append(offsets, len(h.b))
		if bd.Last {
			break
		}
	}
	// h.b holds the full decoded frame content (it is never trimmed to the
	// window -- the offsets above rely on len(h.b) being the cumulative total),
	// so its XXH64 low 32 bits are exactly the zstd content checksum the frame
	// trailer carries, and the slice between any two boundary offsets is a span's
	// exact decoded output (the caller's per-span digest source).
	checksum := uint32(xxhash.Sum64(h.b))
	return offsets, checksum, h.b, nil
}

// BlockUncompStreaming decodes a raw block stream (a frame body with NO frame
// header) ONCE from frame-initial state and returns the same block-boundary offset
// map and content checksum BlockUncompOffsets does -- but WITHOUT materializing the
// whole decoded frame. Between blocks it trims its working buffer down to the frame
// window (the most any later block can reference back into), so its peak memory is
// O(windowSize + one block) rather than O(full decoded size). This is the build-time
// memory bound for an honest, highly-compressible single-frame layer whose decoded
// size far exceeds its compressed size: a few-KiB layer that truthfully decodes to
// many GiB is indexed without the indexer ever holding the whole decode (F-029).
//
// onBlock, if non-nil, is called once per block in decode order with that block's
// freshly decoded output bytes. uncompOffset is the cumulative output length BEFORE
// the block (== offsets[blockIndex]); out is the block's new bytes. out is valid
// ONLY for the duration of the call -- it aliases the trimmed working buffer and is
// overwritten by the next block -- so onBlock must hash/copy it, never retain it. A
// caller computes per-span uncompressed digests over the full decode in this single
// pass by feeding out to a per-span XXH64Digest (see soci's phase-2 build path).
// A non-nil onBlock that returns an error aborts the decode with that error.
//
// onBoundary, if non-nil, is called at EACH block boundary BEFORE that block is
// decoded, with the decoder state a resume from there would seed: blockIndex, the
// cumulative uncompressed offset at the boundary (== offsets[blockIndex]), the
// trailing window (the last windowSize decoded bytes -- all a later match can reach
// back into), and the three recent match offsets in effect. window aliases the
// working buffer and is valid ONLY for the duration of the call, so onBoundary must
// copy any bytes it keeps. This lets a caller snapshot a serializable mid-frame
// checkpoint DURING this single forward decode instead of re-decoding the prefix
// [0:blockIndex] from frame start once per checkpoint -- the O(N^2) build-CPU cost
// CaptureResumeState incurs when called per span (soci F-030). A non-nil onBoundary
// that returns an error aborts the decode.
//
// maxOut bounds the cumulative decoded output, exactly as in BlockUncompOffsets:
// the running TOTAL (already-trimmed bytes plus the live buffer) is checked after
// each block and the decode stops with ErrDecoderSizeExceeded past the bound, so a
// decompression bomb is rejected even though no single allocation ever approaches
// the total. maxOut == 0 disables the bound.
func BlockUncompStreaming(blocks []byte, windowSize uint64, maxOut uint64, onBlock func(blockIndex, uncompOffset int, out []byte) error, onBoundary func(blockIndex, uncompOffset int, window []byte, recent [3]int) error) ([]int, uint32, error) {
	// Seed exactly like BlockUncompOffsetsAndContent: capacity from the compressed
	// bytes present, never from any declared length (F-019). The buffer grows from
	// real output and is trimmed back to the window below, so it stabilizes at
	// ~windowSize regardless of the frame's true decoded size.
	h := seedHistory(nil, windowSize, [3]int{}, nil, nil, len(blocks))
	h.ignoreBuffer = 0

	offsets := []int{0}
	var crc xxhash.Digest
	crc.Reset()
	// trimmed counts the output bytes already discarded off the FRONT of h.b. The
	// true cumulative output at any point is trimmed + len(h.b); h.b itself never
	// exceeds the window (plus the one block just decoded, before the trim).
	trimmed := 0
	var bd blockDec
	bd.lowMem = false
	br := byteBuf(blocks)
	for idx := 0; ; idx++ {
		before := trimmed + len(h.b)
		if onBoundary != nil {
			// The trailing windowSize bytes of h.b are the most a resume from this
			// boundary can reference back into; the front-trim only ever drops bytes
			// already beyond that reach, so they are exactly the last windowSize decoded
			// bytes (the whole prefix when it is shorter than the window). This is
			// byte-identical to trimCheckpointWindow over a full prefix decode, so a
			// checkpoint captured here matches one captured by CaptureResumeState.
			keep := len(h.b)
			if h.windowSize > 0 && keep > h.windowSize {
				keep = h.windowSize
			}
			if err := onBoundary(idx, before, h.b[len(h.b)-keep:], h.recentOffsets); err != nil {
				return nil, 0, err
			}
		}
		if err := bd.reset(&br, windowSize); err != nil {
			return nil, 0, err
		}
		if err := bd.decodeBuf(h); err != nil {
			return nil, 0, err
		}
		// The block appended its output to the end of h.b (the sync path's RLE/Raw
		// appendKeep and the compressed path's grow-in-place both extend h.b), so the
		// new bytes are the suffix past where it stood before this block. trimmed did
		// not change during the block, so the new length is the h.b length delta.
		newLen := (trimmed + len(h.b)) - before
		newBytes := h.b[len(h.b)-newLen:]
		crc.Write(newBytes)
		if onBlock != nil {
			if err := onBlock(idx, before, newBytes); err != nil {
				return nil, 0, err
			}
		}
		total := before + newLen
		// Bound the cumulative output, not the live buffer (which the trim keeps at
		// ~windowSize): a bomb is caught on TOTAL produced, one block's overshoot.
		if maxOut > 0 && uint64(total) > maxOut {
			return nil, 0, ErrDecoderSizeExceeded
		}
		offsets = append(offsets, total)
		// Trim h.b back to the window, but AMORTIZED: only once it has grown to about
		// twice the window, not after every block. A later block's match offset is
		// bounded by windowSize (blockDec rejects mo > windowSize), so any bytes beyond
		// the trailing windowSize are dead and keeping a windowSize slack before
		// reclaiming them is harmless. Match copies index from the END of the output
		// buffer (len(out)-mo), so trimming the front leaves all in-window matches
		// resolvable -- the same invariant history.append maintains on the streaming
		// Read path. Trimming after EVERY block would copy windowSize bytes per block,
		// i.e. O(blocks * window) total work -- which is quadratic in the layer size
		// when the window scales with it (an incompressible layer whose encoder widens
		// the window with the input). Reclaiming only after a windowSize of new output
		// has accumulated amortizes the copy to O(1) per output byte (O(decoded) total)
		// while capping live memory at ~2*windowSize + one block.
		if h.windowSize > 0 && len(h.b)-h.windowSize >= h.windowSize {
			discard := len(h.b) - h.windowSize
			copy(h.b, h.b[discard:])
			h.b = h.b[:h.windowSize]
			trimmed += discard
		}
		if bd.Last {
			break
		}
	}
	return offsets, uint32(crc.Sum64()), nil
}

// XXH64 returns the 64-bit xxhash of b, the same hash zstd uses for its frame
// content checksum. It is exported so soci's phase-2 build and serve paths
// compute a span's uncompressed digest with the identical primitive: the build
// path hashes a span's AUTHORITATIVE decoded output here, the serve path hashes
// the bytes its resume actually produced, and a mismatch flags a resume that
// diverged from what the index recorded.
func XXH64(b []byte) uint64 {
	return xxhash.Sum64(b)
}

// XXH64Digest is the streaming form of XXH64: a resettable hasher whose Sum64 over
// a sequence of Write calls equals XXH64 of the concatenated bytes. soci's phase-2
// build hashes a span's decoded bytes incrementally as a window-trimmed streaming
// decode (BlockUncompStreaming) produces them, so it computes the per-span
// uncompressed digest without ever holding the whole span in memory (F-029). The
// zero value is NOT ready; obtain one from NewXXH64Digest.
type XXH64Digest struct {
	d xxhash.Digest
}

// NewXXH64Digest returns a reset streaming XXH64 hasher.
func NewXXH64Digest() *XXH64Digest {
	x := &XXH64Digest{}
	x.d.Reset()
	return x
}

// Reset returns the hasher to its initial state so it can hash the next span.
func (x *XXH64Digest) Reset() { x.d.Reset() }

// Write adds b to the running hash. It never returns an error.
func (x *XXH64Digest) Write(b []byte) { _, _ = x.d.Write(b) }

// Sum64 returns the XXH64 of all bytes written since the last Reset.
func (x *XXH64Digest) Sum64() uint64 { return x.d.Sum64() }


// ResumeDecode decodes raw blocks (a frame body with NO frame header) seeded
// from a previously captured ResumeState, and returns ONLY the bytes decoded by
// these blocks (the tail), not the seed window. tailOut is the exact number of
// bytes the supplied blocks decode to.
//
// blocks must be the frame body starting exactly at the boundary st was
// captured at (i.e. the blocks AFTER the captured prefix). windowSize is the
// frame's window size. The returned tail is byte-identical to the corresponding
// region of a full decode of the whole frame.
func ResumeDecode(blocks []byte, windowSize uint64, st ResumeState, tailOut int) ([]byte, error) {
	return ResumeDecodeN(blocks, windowSize, st, 0, tailOut)
}

// ResumeDecodeN is ResumeDecode but stops after nBlocks blocks instead of always
// running to the frame's Last block. It is what serving a single INTERIOR span
// needs: the span's blocks end on a boundary that is not the frame's last block
// (its Last_Block flag is 0), so a decode that ran to the Last block would read
// past the supplied buffer. With nBlocks > 0 the decode stops after exactly that
// many blocks; nBlocks == 0 runs until the Last block (identical to ResumeDecode).
//
// tailOut is the number of bytes the decoded blocks are EXPECTED to produce. It is
// NOT used to size the allocation: on the serve path tailOut is the stored
// uncompOffsets delta, an attacker-influenceable length in a tampered ztoc, so
// reserving it up front would let a tiny span declaring a huge delta drive an
// unbounded make() -> OOM (F-020). The output buffer is instead reserved from the
// bytes PRESENT (the seed window plus the compressed blocks) and GROWS from what
// the decode actually produces (maxSyncLen left 0), so the allocation tracks the
// REAL tail regardless of any declared length. The caller is responsible for
// checking the returned tail length against tailOut and rejecting a mismatch.
//
// blocks must be the frame body starting exactly at the boundary st was captured
// at and contain at least nBlocks blocks. The returned tail is byte-identical to
// the corresponding region of a full decode of the whole frame, because a zstd
// block's output never depends on the Last_Block flag or on any later block.
func ResumeDecodeN(blocks []byte, windowSize uint64, st ResumeState, nBlocks, tailOut int) ([]byte, error) {
	var huff *huff0.Scratch
	var seqDecs *sequenceDecs
	if st.hasEntropy {
		huff = st.huff
		seqDecs = &st.seqDecs
	}
	seedLen := len(st.Window)
	// Seed capacity from the compressed bytes present, never from the caller's
	// tailOut (see the doc comment / F-020). The decode grows beyond this as needed.
	h := seedHistory(st.Window, windowSize, st.RecentOffsets, huff, seqDecs, 0)
	// Pass 0 (not tailOut) so maxSyncLen stays 0 and the sync decoder grows its
	// output on demand rather than trusting the attacker-influenceable declared
	// length; nBlocks alone bounds how far the decode runs.
	if _, err := runRawBlocks(h, blocks, windowSize, nBlocks, 0); err != nil {
		return nil, err
	}
	tail := make([]byte, len(h.b)-seedLen)
	copy(tail, h.b[seedLen:])
	return tail, nil
}

// ResumeProbeN decodes nBlocks blocks of a raw block stream seeded from st, like
// ResumeDecodeN, but returns ONLY whether the decode succeeds -- it never
// materializes or returns the decoded tail. Between blocks it trims its working
// buffer back to the frame window (the same amortized trim BlockUncompStreaming
// uses), so its peak memory is O(windowSize + one block) regardless of how many
// bytes the span decodes to.
//
// This is what shrinkResumeWindow's probe needs. shrinkResumeWindow searches for
// the shortest leading suffix of a captured window that still decodes the span
// without a "match offset bigger than current history" rejection; it needs only
// that success/failure signal, never the decoded bytes. Using ResumeDecodeN for
// the probe materialized the whole span tail in h.b, so an interior span that
// grew far past spanSize -- one spanning many spanSize of output between two
// sparse entropy-clean boundaries, or the LAST interior span running to frame end
// -- drove the per-span shrink allocation to O(span) and hence O(decoded) build
// peak RAM, the same memory class F-029 closed on the offset-map and
// checkpoint-capture paths but on the shrink path the dense-checkpoint guard does
// not reach (its small spans never make a span large). ResumeProbeN bounds that
// transient to the window.
//
// The trimmed probe is outcome-identical to a non-trimmed ResumeDecodeN: a
// too-small seed window is rejected by the same match-reach check, and that
// rejection necessarily fires BEFORE any trim occurs. The trim only reclaims bytes
// once the buffer exceeds 2*windowSize, by which point the in-span output already
// exceeds windowSize, so the available history is at least windowSize and no legal
// match (offset <= windowSize) can be rejected from then on. Any rejection that
// signals a too-small suffix therefore happens while the buffer is still untrimmed,
// so trimming never changes success or failure -- shrinkResumeWindow picks the
// identical suffix it would have with ResumeDecodeN.
//
// blocks must be the frame body starting exactly at the boundary st was captured
// at and contain at least nBlocks blocks. nBlocks > 0 stops after that many
// blocks; nBlocks == 0 runs to the Last block.
func ResumeProbeN(blocks []byte, windowSize uint64, st ResumeState, nBlocks int) error {
	var huff *huff0.Scratch
	var seqDecs *sequenceDecs
	if st.hasEntropy {
		huff = st.huff
		seqDecs = &st.seqDecs
	}
	// Reserve NO output capacity up front: the buffer grows from real output via
	// append and is trimmed back to the window below, so it stabilizes at ~windowSize
	// regardless of the span's true decoded size. Seeding the reservation from
	// len(blocks) (the compressed span bytes present) was safe from a declared length
	// (F-019) but still made the INITIAL make() O(compressed-span): for an
	// INCOMPRESSIBLE wide span (sparse entropy-clean boundaries, or the last interior
	// span running to frame end) len(blocks) ~ the decoded span, so the up-front
	// reservation -- which the trim cannot reclaim, it predates the first decode --
	// reintroduced an O(span) probe peak, the F-029 memory class this primitive exists
	// to avoid, on the incompressible case the highly-compressible shrink test cannot
	// see. expectedOut feeds ONLY seedHistory's initial capacity here (maxSyncLen is
	// forced to 0 below, so it never reaches the sync decoder), so passing 0 is purely
	// a memory tightening with no effect on what the probe decodes: the grow+trim path
	// caps the working set at ~windowSize independent of how large the span is.
	h := seedHistory(st.Window, windowSize, st.RecentOffsets, huff, seqDecs, 0)
	h.ignoreBuffer = 0
	h.decoders.maxSyncLen = 0
	var bd blockDec
	bd.lowMem = false
	br := byteBuf(blocks)
	for n := 0; ; {
		if err := bd.reset(&br, windowSize); err != nil {
			return err
		}
		if err := bd.decodeBuf(h); err != nil {
			return err
		}
		n++
		// Trim h.b back to the window, AMORTIZED exactly as BlockUncompStreaming:
		// reclaim only once it has grown to ~2*windowSize so the copy is O(1)/byte
		// (O(span) total), never O(blocks*window). A match offset is bounded by
		// windowSize, so dropping bytes beyond the trailing windowSize is harmless.
		if h.windowSize > 0 && len(h.b)-h.windowSize >= h.windowSize {
			discard := len(h.b) - h.windowSize
			copy(h.b, h.b[discard:])
			h.b = h.b[:h.windowSize]
		}
		if bd.Last || (nBlocks > 0 && n >= nBlocks) {
			break
		}
	}
	return nil
}
