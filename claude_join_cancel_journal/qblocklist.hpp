// qblocklist.hpp — per-price-level priority queue: single block for up to 16
// CONCURRENT orders (the common case, forever — no list machinery), spilling
// to a doubly linked blocklist only beyond that.
//
// Priority is decoupled from storage by a 16-nibble permutation per block:
// nibble i of `map` = slot index of the i-th priority order. Slots never
// move, so handles (block<<4 | slot) are stable for the order's lifetime;
// dead slots are reused via a 16-bit occupancy mask + tzcnt.
//
// Spill rule: slot reuse is permitted only while the queue is a single block
// (head == tail). With a chain, pushes go append-style to the tail block
// only — a reused early-block hole would outrank older orders in later
// blocks. When the chain drains back to one block, reuse resumes.
//
// The map holds ONLY live entries, so the ahead-count inside the target
// block is simply the nibble position — no dead-entry filtering. Skipped
// blocks contribute their {live, bsz} header aggregates: one line each.
//
// Tracking: `trk` (slot-indexed, always a subset of `occ`) flags orders a
// consumer follows — our own orders, registered with the cancel tactic.
// track() sets or clears it. `ahd` (slot-indexed, subset of `occ`) flags
// orders that held at least a round lot when one of our orders behind them
// was placed: scan_ahead(mark) sets it, never on one of ours. Between the
// scanning order and the nearest tracked order ahead of it the flags are
// rewritten — set for a round lot now, cleared otherwise — since nothing
// else of ours depends on them; from that order on they are only added, so
// a flag set for it stays. As each flagged order leaves, the cancel tactic
// tells from the event whether it was cancelled or filled. remove(),
// reduce() and grow()
// report the order's flags, so the E/C/X/D/U path learns "one of ours" or
// "was ahead of one of ours" from the header line it already touched;
// remove() clears them with the slot, so a reused slot starts clean.
// (remove() and reduce() return these flags, not a bool: test kQTracked for
// "one of ours".) ahead_of(), ahead_n(), scan_ahead(), clear_ahead(),
// alone() and size_of() answer from a handle alone (the chain is walked
// backwards through prev), for holders without the list state.
// clear_ahead(lot): nothing ahead of the order is tracked or holds a round
// lot — it leads the tracked orders of its queue from the front, or from
// behind odd lots only.
//
// Events: push_ev(), remove_ev(), reduce_ev() and grow_ev() do what push(),
// remove(), reduce() and grow() do and return a QEv — the order's handle, its
// size before and after, and the flags — for a consumer that keeps a history
// of the queue's events: the cancel tactic's journal, from which an order
// recognised as ours only later is tracked as if from its insertion. The
// plain calls return the flags alone and cost what they did.
//
// List state {head, tail} is 8 bytes: store it inside PrcLevelGeneric::plist.
// behind = level totals − ahead − self (PrcLevelGeneric num_orders/agg_sz
// ARE the queue totals).
//
// Single writer. Build: -std=c++23 -O3 -march=native.

#pragma once

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace md {

struct QAhead { uint32_t n; uint64_t sz; };
struct QScan  { uint32_t n; uint64_t sz; uint32_t n_lot; };   // + flagged (round-lot) orders ahead, not ours

// What remove() / reduce() report about the order they touched (bit flags).
inline constexpr uint8_t kQGone    = 1;   // remove(): the order left the queue
inline constexpr uint8_t kQTracked = 2;   // it is tracked (one of ours)
inline constexpr uint8_t kQAhead   = 4;   // it held at least a round lot when a tracked order
                                          //   behind it was placed (flagged by scan_ahead)
inline constexpr uint8_t kQGrew    = 8;   // grow(): its size increased in place, keeping priority
inline constexpr uint8_t kQNew     = 16;  // push_ev(): the order entered the queue

// What one queue call did to one order: handle, size before and after, and
// the flags above. Converts from the flags alone (handle unknown), so a
// consumer that takes a QEv also takes what remove()/reduce()/grow() return.
struct QEv {
    uint32_t h     = 0xFFFFFFFFu;         // the order's handle; kBadHandle: not given
    uint32_t sz0   = 0;                   // its size before the call (0: a push)
    uint32_t sz1   = 0;                   //   and after (0: a remove)
    uint8_t  flags = 0;                   // kQNew | kQGone | kQTracked | kQAhead | kQGrew
    constexpr QEv() noexcept = default;
    constexpr QEv(uint8_t f) noexcept : flags(f) {}                       // NOLINT: implicit on purpose
    constexpr QEv(uint32_t handle, uint32_t before, uint32_t after, uint8_t f) noexcept
        : h(handle), sz0(before), sz1(after), flags(f) {}
};
struct QPos2  { uint32_t ahead_n; uint64_t ahead_sz;
                uint32_t behind_n; uint64_t behind_sz; };

class QBlockPool {
public:
    static constexpr unsigned kCap = 16;

    // 32-byte aligned: the 32-byte header never straddles a cache line (with
    // 8-byte alignment and a 224-byte stride, every other block's header did
    // whenever the array started 16 bytes into a line, as glibc's large
    // allocations do), and the two szp loads in prefix_sum never split one.
    struct alignas(32) Block {         // header first: block-skip = 1 line
        uint32_t next, prev;           // pool idx + 1; 0 = none
        uint16_t live;
        uint16_t occ;                  // slot occupancy bitmask
        uint16_t trk;                  // tracked slots (subset of occ)
        uint16_t ahd;                  // round lots flagged ahead of our orders (subset of occ, never trk)
        uint64_t bsz;                  // live size aggregate
        uint64_t map;                  // nibble i = slot of priority-i order
        uint64_t id[kCap];             // slot-indexed; 0 = dead
        uint32_t szp[kCap];            // PRIORITY-indexed, dense over live:
    };                                 // szp[i] = size of priority-i order
    static_assert(sizeof(Block) == 224 && alignof(Block) == 32);  // 32B header + 128 ids + 64 szs

    explicit QBlockPool(std::size_t nblocks) {
        assert(nblocks <= (1u << 27));
        blocks_.assign(nblocks ? nblocks : 1, Block{});
    }
    uint32_t alloc() noexcept {        // idx+1; 0 = exhausted
        uint32_t i;
        if (free_head_) { i = free_head_; free_head_ = blocks_[i - 1].next; }
        else if (hw_ < blocks_.size()) i = static_cast<uint32_t>(++hw_);
        else return 0;
        Block& b = blocks_[i - 1];
        b.next = b.prev = 0;           // entries need no clearing: a slot is
        b.live = 0; b.occ = 0;         // only read while in `map`, and every
        b.bsz  = 0; b.map = 0;         // push fully writes its slot
        b.trk  = 0; b.ahd = 0;
        ++used_;
        return i;
    }
    void free(uint32_t i) noexcept {
        blocks_[i - 1].next = free_head_; free_head_ = i; --used_;
    }
    Block&       at(uint32_t i)       noexcept { return blocks_[i - 1]; }
    const Block& at(uint32_t i) const noexcept { return blocks_[i - 1]; }
    std::size_t  in_use()   const noexcept { return used_; }
    std::size_t  capacity() const noexcept { return blocks_.size(); }

private:
    std::vector<Block> blocks_;
    uint32_t free_head_ = 0;
    std::size_t hw_ = 0, used_ = 0;
};

class QBlockList {
public:
    static constexpr uint32_t kBadHandle = 0xFFFFFFFFu;

    [[nodiscard]] bool empty() const noexcept { return head_ == 0; }

    uint32_t push(QBlockPool& p, uint64_t id, uint32_t sz) noexcept {
        assert(id != 0 && sz != 0);
        if (head_ == 0) {
            const uint32_t nb = p.alloc();
            if (nb == 0) [[unlikely]] return kBadHandle;
            head_ = tail_ = nb;
        } else if (p.at(tail_).live == QBlockPool::kCap) {
            const uint32_t nb = p.alloc();               // genuine 17th+ order
            if (nb == 0) [[unlikely]] return kBadHandle;
            p.at(nb).prev = tail_;
            p.at(tail_).next = nb;
            tail_ = nb;
        }
        QBlockPool::Block& b = p.at(tail_);
        // Free slot via occupancy mask. Single block: any hole is reusable.
        // Chained tail block: also fine — priority is map order, not slot
        // order, and the tail block always holds the newest orders.
        const uint32_t slot = static_cast<uint32_t>(
            std::countr_zero(static_cast<unsigned>(~b.occ & 0xFFFFu)));
        assert(!(((b.trk | b.ahd) >> slot) & 1u));              // trk, ahd subsets of occ
        b.occ |= static_cast<uint16_t>(1u << slot);
        b.id[slot] = id;
        b.szp[b.live] = sz;                                     // append dense
        b.map |= static_cast<uint64_t>(slot) << (4u * b.live);  // append nibble
        b.live += 1;
        b.bsz  += sz;
        return ((tail_ - 1) << 4) | slot;
    }
    // push(), as an event: h is the new order's handle (kBadHandle and no
    // flags if the pool is exhausted).
    QEv push_ev(QBlockPool& p, uint64_t id, uint32_t sz) noexcept {
        const uint32_t h = push(p, id, sz);
        return {h, 0u, sz, h == kBadHandle ? uint8_t{0} : kQNew};
    }

    // Returns kQGone plus the removed order's kQTracked / kQAhead flags; the
    // flags go with the slot.
    uint8_t remove(QBlockPool& p, uint32_t h) noexcept { return remove_ev(p, h).flags; }
    // remove(), as an event: with the handle and the size the order had.
    QEv remove_ev(QBlockPool& p, uint32_t h) noexcept {
        const uint32_t bi = (h >> 4) + 1, slot = h & 15u;
        QBlockPool::Block& b = p.at(bi);
        assert((b.occ >> slot) & 1u);
        const uint8_t f = static_cast<uint8_t>(flags(b, slot) | kQGone);
        const uint32_t pos = map_pos(b, slot);
        const uint32_t was = b.szp[pos];
        b.bsz -= was;
        map_erase(b, pos);
        std::memmove(b.szp + pos, b.szp + pos + 1,     // keep szp dense
                     (b.live - 1u - pos) * sizeof(uint32_t));
        const uint16_t keep = static_cast<uint16_t>(~(1u << slot));
        b.occ &= keep; b.trk &= keep; b.ahd &= keep;
        b.id[slot] = 0;
        if (--b.live == 0) {
            if (b.prev) p.at(b.prev).next = b.next; else head_ = b.next;
            if (b.next) p.at(b.next).prev = b.prev; else tail_ = b.prev;
            p.free(bi);
        }
        return {h, was, 0u, f};
    }

    // Returns the reduced order's kQTracked / kQAhead flags (no kQGone: it
    // stays, in place).
    uint8_t reduce(QBlockPool& p, uint32_t h, uint32_t delta) noexcept { return reduce_ev(p, h, delta).flags; }
    QEv reduce_ev(QBlockPool& p, uint32_t h, uint32_t delta) noexcept {
        const uint32_t bi = (h >> 4) + 1, slot = h & 15u;
        QBlockPool::Block& b = p.at(bi);
        const uint32_t pos = map_pos(b, slot);
        assert(((b.occ >> slot) & 1u) && b.szp[pos] > delta);
        const uint32_t was = b.szp[pos];
        b.szp[pos] = was - delta;
        b.bsz     -= delta;
        return {h, was, was - delta, flags(b, slot)};
    }

    // Size increase in place, keeping time priority — for venues that allow
    // it (elsewhere an increase is a remove() and a push() to the back).
    // Returns kQGrew plus the order's kQTracked / kQAhead flags.
    uint8_t grow(QBlockPool& p, uint32_t h, uint32_t delta) noexcept { return grow_ev(p, h, delta).flags; }
    QEv grow_ev(QBlockPool& p, uint32_t h, uint32_t delta) noexcept {
        const uint32_t bi = (h >> 4) + 1, slot = h & 15u;
        QBlockPool::Block& b = p.at(bi);
        const uint32_t pos = map_pos(b, slot);
        assert(((b.occ >> slot) & 1u) && delta > 0 && b.szp[pos] <= 0xFFFFFFFFu - delta);
        const uint32_t was = b.szp[pos];
        b.szp[pos] = was + delta;
        b.bsz     += delta;
        return {h, was, was + delta, static_cast<uint8_t>(flags(b, slot) | kQGrew)};
    }

    [[nodiscard]] QAhead ahead(const QBlockPool& p, uint32_t h) const noexcept {
        const uint32_t bi = (h >> 4) + 1, slot = h & 15u;
        uint32_t an = 0; uint64_t as = 0;
        uint32_t cur = head_;
        while (cur != bi) {                    // skipped blocks: header only
            const QBlockPool::Block& b = p.at(cur);
            an += b.live;
            as += b.bsz;
            cur = b.next;
            assert(cur != 0 && "handle's block not on this list");
        }
        const QBlockPool::Block& b = p.at(bi);
        const uint32_t pos = map_pos(b, slot); // ahead-in-block == nibble pos
        an += pos;
        as += prefix_sum(b.szp, pos);          // contiguous: two masked loads
        return {an, as};
    }

    [[nodiscard]] QPos2 position(const QBlockPool& p, uint32_t h,
                                 uint32_t level_n, uint64_t level_sz) const noexcept {
        const QAhead a = ahead(p, h);
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t self = b.szp[map_pos(b, h & 15u)];
        return {a.n, a.sz, level_n - a.n - 1, level_sz - a.sz - self};
    }

    template <class F>
    void for_each(const QBlockPool& p, F&& f) const {
        for (uint32_t cur = head_; cur != 0; ) {
            const QBlockPool::Block& b = p.at(cur);
            for (uint32_t i = 0; i < b.live; ++i) {
                const uint32_t s = (b.map >> (4u * i)) & 15u;
                if (!f(((cur - 1) << 4) | s, b.id[s], b.szp[i])) return;
            }
            cur = b.next;
        }
    }

    // ---- tracking and handle-only queries (static: a handle addresses its
    //      slot without the list state) -----------------------------------

    static void track(QBlockPool& p, uint32_t h, bool on) noexcept {
        QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint16_t bit = static_cast<uint16_t>(1u << (h & 15u));
        assert(b.occ & bit);
        b.trk = on ? static_cast<uint16_t>(b.trk | bit) : static_cast<uint16_t>(b.trk & ~bit);
    }
    [[nodiscard]] static bool tracked(const QBlockPool& p, uint32_t h) noexcept {
        return (p.at((h >> 4) + 1).trk >> (h & 15u)) & 1u;
    }
    // The kQAhead flag of one order, read and written by handle: for an
    // order recognised as ours late, whose flag other orders of ours set
    // while it passed for a market order, and for the orders ahead of it
    // that held a round lot when it was inserted and no longer do.
    [[nodiscard]] static bool flagged_ahead(const QBlockPool& p, uint32_t h) noexcept {
        return (p.at((h >> 4) + 1).ahd >> (h & 15u)) & 1u;
    }
    static void flag_ahead(QBlockPool& p, uint32_t h, bool on) noexcept {
        QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint16_t bit = static_cast<uint16_t>(1u << (h & 15u));
        assert(b.occ & bit);
        b.ahd = on ? static_cast<uint16_t>(b.ahd | bit) : static_cast<uint16_t>(b.ahd & ~bit);
    }

    // ahead(), walking the chain backwards from the handle's block: same
    // result, no list head needed.
    [[nodiscard]] static QAhead ahead_of(const QBlockPool& p, uint32_t h) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t pos = map_pos(b, h & 15u);
        uint32_t an = pos; uint64_t as = prefix_sum(b.szp, pos);
        for (uint32_t cur = b.prev; cur != 0; ) {    // earlier blocks: header only
            const QBlockPool::Block& x = p.at(cur);
            an += x.live;
            as += x.bsz;
            cur = x.prev;
        }
        return {an, as};
    }
    // Orders ahead of the handle's order: ahead_of() without the shares.
    [[nodiscard]] static uint32_t ahead_n(const QBlockPool& p, uint32_t h) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        uint32_t n = map_pos(b, h & 15u);
        for (uint32_t cur = b.prev; cur != 0; ) {
            const QBlockPool::Block& x = p.at(cur);
            n += x.live;
            cur = x.prev;
        }
        return n;
    }
    [[nodiscard]] static uint32_t size_of(const QBlockPool& p, uint32_t h) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        return b.szp[map_pos(b, h & 15u)];
    }

    // The handle's order is the only order in its queue (its price level).
    [[nodiscard]] static bool alone(const QBlockPool& p, uint32_t h) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        return b.live == 1 && b.prev == 0 && b.next == 0;
    }

    // Nothing ahead of the handle's order is tracked or holds `lot` shares or
    // more: it is the first tracked order of its queue, and whatever is ahead
    // of it — if anything — is odd lots. Stops at the first block that says
    // otherwise, nearest first; a block ahead costs its header and its sizes.
    // A call, never inlined, like scan_ahead().
    [[nodiscard, gnu::noinline]] static bool clear_ahead(const QBlockPool& p, uint32_t h, uint32_t lot) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t slot = h & 15u, pos = map_pos(b, slot);
        if (pos) {
            const uint32_t ahead = (1u << pos) - 1u;                // priority positions < pos
            if (lot_mask(b.szp, lot) & ahead) return false;
            if ((b.trk & ~(1u << slot)) && (b.trk & slots_of(b.map, ahead))) return false;
        }
        for (uint32_t cur = b.prev; cur != 0; ) {                   // earlier blocks: all ahead
            const QBlockPool::Block& x = p.at(cur);
            const uint32_t all = x.live == QBlockPool::kCap ? 0xFFFFu : (1u << x.live) - 1u;
            if (x.trk || (lot_mask(x.szp, lot) & all)) return false;
            cur = x.prev;
        }
        return true;
    }
    // No order ahead of the handle's order is tracked: of the tracked orders
    // in its queue it is the first (or would be).
    [[nodiscard]] static bool leads(const QBlockPool& p, uint32_t h) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t slot = h & 15u;
        if ((b.trk & ~(1u << slot)) && (b.trk & slots_of(b.map, (1u << map_pos(b, slot)) - 1u))) return false;
        for (uint32_t cur = b.prev; cur != 0; cur = p.at(cur).prev)
            if (p.at(cur).trk) return false;
        return true;
    }
    // How many orders ahead of the handle's order are tracked or hold `lot`
    // shares or more: 0 exactly when clear_ahead(). The whole queue ahead is
    // read; for the cancel tactic's replay of an order recognised late.
    [[nodiscard, gnu::noinline]] static uint32_t blockers_ahead(const QBlockPool& p, uint32_t h,
                                                                uint32_t lot) noexcept {
        const QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t pos = map_pos(b, h & 15u);
        uint32_t n = static_cast<uint32_t>(std::popcount((lot_mask(b.szp, lot) | positions_of(b, b.trk)) &
                                                         ((1u << pos) - 1u)));
        for (uint32_t cur = b.prev; cur != 0; ) {
            const QBlockPool::Block& x = p.at(cur);
            const uint32_t all = x.live == QBlockPool::kCap ? 0xFFFFu : (1u << x.live) - 1u;
            n += static_cast<uint32_t>(std::popcount((lot_mask(x.szp, lot) | positions_of(x, x.trk)) & all));
            cur = x.prev;
        }
        return n;
    }

    // ahead_of(), plus n_lot: the round-lot orders ahead of the handle's
    // order that are not tracked (not ours) — and, with mark, their flags
    // (kQAhead), so that n_lot is the flagged orders ahead. From the scanning
    // order forward to the nearest tracked order ahead of it, an order counts
    // if it holds at least `lot` shares now, and its flag is rewritten to
    // say so: no other tracked order depends on those flags. From that
    // tracked order on, flags are only added — an order flagged for it also
    // counts for us, as does one holding `lot` now. Flags are rewritten only
    // for a scan from the tail of the queue (the usual case: a new order);
    // behind any other order a tracked one may rest, so that scan only adds
    // — unless the caller knows that none does and says so (rewrite): an
    // order recognised as ours late, with none of ours behind it.
    // One pass over the blocks ahead: each skipped block costs its header
    // and its size line. A call, never inlined: its 256-bit compares do not
    // belong in the caller's frame.
    [[gnu::noinline]] static QScan scan_ahead(QBlockPool& p, uint32_t h, uint32_t lot, bool mark,
                                              bool rewrite = false) noexcept {
        QBlockPool::Block& b = p.at((h >> 4) + 1);
        const uint32_t pos = map_pos(b, h & 15u);
        const uint32_t ahead = (1u << pos) - 1u;                    // priority positions < pos
        QScan r{pos, prefix_sum(b.szp, pos), 0};
        bool exact = rewrite || (pos + 1u == b.live && b.next == 0);   // nothing behind us, nothing of
        if (pos) {                                                  //   ours met yet
            const uint32_t live = b.live == QBlockPool::kCap ? 0xFFFFu : (1u << b.live) - 1u;
            // slots ahead = occupied minus ours and later ones (usually just ours: one step)
            const uint16_t in = static_cast<uint16_t>(b.occ & ~slots_of(b.map, live & ~ahead));
            r.n_lot = lots_ahead(b, in, ahead, lot, mark, exact);
        }
        for (uint32_t cur = b.prev; cur != 0; ) {                   // earlier blocks: all ahead
            QBlockPool::Block& x = p.at(cur);
            const uint32_t all = x.live == QBlockPool::kCap ? 0xFFFFu : (1u << x.live) - 1u;
            r.n += x.live; r.sz += x.bsz;
            r.n_lot += lots_ahead(x, x.occ, all, lot, mark, exact);
            cur = x.prev;
        }
        return r;
    }

private:
    // scan_ahead() for one block: `in` the slots ahead (slot-indexed), `pm`
    // their priority positions; exact (in, out): no tracked order between
    // these slots and the scanning order. Tracked slots are never counted
    // and their flags never touched.
    [[gnu::always_inline]] static uint32_t lots_ahead(QBlockPool::Block& b, uint16_t in, uint32_t pm,
                                                      uint32_t lot, bool mark, bool& exact) noexcept {
        const uint16_t big = static_cast<uint16_t>(slots_of(b.map, lot_mask(b.szp, lot) & pm) & ~b.trk);
        uint16_t keep = in;                                         // slots whose flags stay set
        if (exact) {
            const uint16_t ours = static_cast<uint16_t>(b.trk & in);
            if (!ours) [[likely]] {
                keep = 0;
            } else {                                                // up to the nearest of ours: exact
                uint32_t last = 0;
                for (uint32_t t = ours; t; t &= t - 1) {
                    const uint32_t at = map_pos(b, static_cast<uint32_t>(std::countr_zero(t)));
                    if (at > last) last = at;
                }
                keep  = static_cast<uint16_t>(in & slots_of(b.map, pm & ((2u << last) - 1u)));
                exact = false;
            }
        }
        const uint16_t f = static_cast<uint16_t>((b.ahd & keep) | big);
        if (mark) b.ahd = static_cast<uint16_t>((b.ahd & ~in) | f);
        return static_cast<uint32_t>(std::popcount(static_cast<unsigned>(f & ~b.trk)));
    }
    static uint8_t flags(const QBlockPool::Block& b, uint32_t slot) noexcept {
        return static_cast<uint8_t>((((b.trk >> slot) & 1u) * kQTracked) |
                                    (((b.ahd >> slot) & 1u) * kQAhead));
    }
    // Priority-indexed mask of szp[i] >= lot (all 16 entries; callers mask
    // to the live or ahead positions). Two ymm compares, not one zmm: no
    // 512-bit instruction on the placement path, so no license or upper-lane
    // power-up transition when an order is placed after a quiet spell.
    [[nodiscard]] static uint32_t lot_mask(const uint32_t* szp, uint32_t lot) noexcept {
#if defined(__AVX512VL__)
        const __m256i l = _mm256_set1_epi32(static_cast<int>(lot));
        const __m256i* p = reinterpret_cast<const __m256i*>(szp);
        return static_cast<uint32_t>(_mm256_cmpge_epu32_mask(_mm256_loadu_si256(p), l)) |
               (static_cast<uint32_t>(_mm256_cmpge_epu32_mask(_mm256_loadu_si256(p + 1), l)) << 8);
#else
        uint32_t m = 0;
        for (uint32_t i = 0; i < QBlockPool::kCap; ++i) m |= (szp[i] >= lot ? 1u : 0u) << i;
        return m;
#endif
    }
    // Slot-indexed mask of the slots at the priority positions in pm.
    [[nodiscard]] static uint16_t slots_of(uint64_t map, uint32_t pm) noexcept {
        uint32_t s = 0;
        for (; pm; pm &= pm - 1)
            s |= 1u << ((map >> (4u * static_cast<unsigned>(std::countr_zero(pm)))) & 15u);
        return static_cast<uint16_t>(s);
    }

    // Priority-indexed mask of the live slots in `slots` (the inverse of slots_of).
    [[nodiscard]] static uint32_t positions_of(const QBlockPool::Block& b, uint32_t slots) noexcept {
        uint32_t pm = 0;
        for (; slots; slots &= slots - 1)
            pm |= 1u << map_pos(b, static_cast<uint32_t>(std::countr_zero(slots)));
        return pm;
    }

    // Priority position of `slot` within the block: index of its nibble.
    // Zero-nibble bit trick on map XOR broadcast(slot); the lowest flag is
    // always a true match, and slot occurs exactly once among live nibbles.
    [[nodiscard]] static uint32_t map_pos(const QBlockPool::Block& b,
                                          uint32_t slot) noexcept {
        constexpr uint64_t kOnes   = 0x1111111111111111ull;
        constexpr uint64_t kEights = 0x8888888888888888ull;
        uint64_t x = b.map ^ (kOnes * slot);
        if (b.live < QBlockPool::kCap)                 // invalidate dead region
            x |= ~0ull << (4u * b.live);
        const uint64_t y = (x - kOnes) & ~x & kEights;
        assert(y != 0);
        if (y == 0) __builtin_unreachable();           // a live slot always matches: without
                                                       // this, NDEBUG builds keep a pos-16 path
                                                       // and -Warray-bounds flags szp[pos]
        return static_cast<uint32_t>(std::countr_zero(y)) >> 2;
    }
    // Sum of szp[0..pos). Contiguous and dense (live-only), so AVX2 is two
    // masked loads + widen + reduce; scalar fallback is a simple loop.
    [[nodiscard]] static uint64_t prefix_sum(const uint32_t* szp,
                                             uint32_t pos) noexcept {
#if defined(__AVX2__)
        const __m256i lanes_lo = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i lanes_hi = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
        const __m256i vpos = _mm256_set1_epi32(static_cast<int>(pos));
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(szp));
        __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(szp + 8));
        a = _mm256_and_si256(a, _mm256_cmpgt_epi32(vpos, lanes_lo));
        c = _mm256_and_si256(c, _mm256_cmpgt_epi32(vpos, lanes_hi));
        const __m256i s64 = _mm256_add_epi64(
            _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(a)),
                             _mm256_cvtepu32_epi64(_mm256_extracti128_si256(a, 1))),
            _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(c)),
                             _mm256_cvtepu32_epi64(_mm256_extracti128_si256(c, 1))));
        // reduced in registers: an aligned array on the stack would make every
        // function this is inlined into realign its frame
        const __m128i s2 = _mm_add_epi64(_mm256_castsi256_si128(s64), _mm256_extracti128_si256(s64, 1));
        return static_cast<uint64_t>(_mm_cvtsi128_si64(s2)) + static_cast<uint64_t>(_mm_extract_epi64(s2, 1));
#else
        uint64_t as = 0;
        for (uint32_t i = 0; i < pos; ++i) as += szp[i];
        return as;
#endif
    }
    static void map_erase(QBlockPool::Block& b, uint32_t pos) noexcept {
        const uint64_t low = (pos == 0) ? 0
            : (b.map & ((1ull << (4u * pos)) - 1));
        const uint64_t high = (pos + 1 == QBlockPool::kCap) ? 0
            : ((b.map >> (4u * (pos + 1))) << (4u * pos));
        b.map = low | high;
    }

    uint32_t head_ = 0, tail_ = 0;
};

static_assert(sizeof(QBlockList) == 8,
              "list state must fit inside PrcLevelGeneric::plist");

}  // namespace md
