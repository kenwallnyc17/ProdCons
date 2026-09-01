/* MOCK ef_vi — test double for OpenOnload's etherfabric/ef_vi.h.
 * Same public function names/signatures as the real API (transcribed from
 * the ef_vi documentation); implements a software VI with a descriptor FIFO,
 * an event FIFO, and a test-only injection hook (mock_efvi_inject).
 * Production builds use the real headers and -lciul1 unchanged.
 * Verify signatures against your installed etherfabric/ef_vi.h once.        */
#pragma once
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <deque>
#include <vector>

typedef int      ef_driver_handle;
typedef uint64_t ef_addr;
typedef uint32_t ef_request_id;
typedef struct { uint64_t v; } ef_filter_cookie;

enum ef_vi_flags {
    EF_VI_FLAGS_DEFAULT   = 0,
    EF_VI_RX_TIMESTAMPS   = 0x1,
    EF_VI_RX_EVENT_MERGE  = 0x2,
    EF_VI_RX_PACKED_STREAM= 0x4,
};
enum ef_pd_flags { EF_PD_DEFAULT = 0 };
enum ef_filter_flags { EF_FILTER_FLAG_NONE = 0 };

enum {
    EF_EVENT_TYPE_RX, EF_EVENT_TYPE_TX, EF_EVENT_TYPE_RX_DISCARD,
    EF_EVENT_TYPE_TX_ERROR, EF_EVENT_TYPE_RX_NO_DESC_TRUNC,
    EF_EVENT_TYPE_SW, EF_EVENT_TYPE_OFLOW, EF_EVENT_TYPE_TX_WITH_TIMESTAMP,
    EF_EVENT_TYPE_RX_PACKED_STREAM, EF_EVENT_TYPE_RX_MULTI,
    EF_EVENT_TYPE_TX_ALT, EF_EVENT_TYPE_RX_MULTI_DISCARD,
    EF_EVENT_TYPE_RESET, EF_EVENT_TYPE_RX_MULTI_PKTS, EF_EVENT_TYPE_RX_REF,
    EF_EVENT_TYPE_RX_REF_DISCARD,
};
#define EF_EVENT_FLAG_SOP        0x1
#define EF_EVENT_FLAG_CONT       0x2
#define EF_EVENT_FLAG_ISCSI_OK   0x4
#define EF_EVENT_FLAG_MULTICAST  0x8
#define EF_VI_SYNC_FLAG_CLOCK_SET     1
#define EF_VI_SYNC_FLAG_CLOCK_IN_SYNC 2

typedef union {
    struct { unsigned type; } generic;
    struct { unsigned type; unsigned q_id; ef_request_id rq_id;
             unsigned len; unsigned flags; unsigned ofs; } rx;
    struct { unsigned type; unsigned q_id; ef_request_id rq_id;
             unsigned len; unsigned flags; unsigned subtype; } rx_discard;
    struct { unsigned type; unsigned q_id; unsigned n_descs;
             unsigned flags; } rx_multi;
    struct { unsigned type; unsigned q_id; unsigned n_descs;
             unsigned flags; unsigned subtype; } rx_multi_discard;
    struct { unsigned type; unsigned q_id; } rx_no_desc_trunc;
} ef_event;

#define EF_EVENT_TYPE(e)        ((e).generic.type)
#define EF_EVENT_RX_RQ_ID(e)    ((e).rx.rq_id)
#define EF_EVENT_RX_BYTES(e)    ((e).rx.len)
#define EF_EVENT_RX_SOP(e)      ((e).rx.flags & EF_EVENT_FLAG_SOP)
#define EF_EVENT_RX_CONT(e)     ((e).rx.flags & EF_EVENT_FLAG_CONT)
#define EF_EVENT_RX_MULTI_SOP(e)  ((e).rx_multi.flags & EF_EVENT_FLAG_SOP)
#define EF_EVENT_RX_MULTI_CONT(e) ((e).rx_multi.flags & EF_EVENT_FLAG_CONT)

struct ef_pd { int ifindex; };
struct ef_memreg { uint8_t* base; size_t len; };

/* mock VI: DMA addrs are host pointers (memreg base + offset) */
struct ef_vi {
    int      prefix_len;      /* bytes before the frame (holds mock ts) */
    unsigned flags;
    unsigned rxq_capacity;
    std::deque<std::pair<ef_addr, ef_request_id>>* posted;  /* init'd, not pushed */
    std::deque<std::pair<ef_addr, ef_request_id>>* ring;    /* pushed (FIFO) */
    std::deque<ef_event>* events;
    std::deque<ef_request_id>* multi_ids;   /* ids behind RX_MULTI events */
    unsigned multi_batch;                    /* 0/1 = single RX events */
    std::vector<ef_request_id>* multi_pending;
    unsigned no_desc_drops;
};

typedef struct { int proto; uint32_t ip_be; uint16_t port_be; } ef_filter_spec;

static inline int ef_driver_open(ef_driver_handle* dh) { *dh = 1; return 0; }
static inline int ef_driver_close(ef_driver_handle) { return 0; }
static inline int ef_pd_alloc(ef_pd* pd, ef_driver_handle, int ifindex, enum ef_pd_flags)
{ pd->ifindex = ifindex; return 0; }
static inline int ef_pd_free(ef_pd*, ef_driver_handle) { return 0; }

static inline int ef_vi_alloc_from_pd(ef_vi* vi, ef_driver_handle, ef_pd*, ef_driver_handle,
                                      int, int rxq_capacity, int, ef_vi*, ef_driver_handle,
                                      enum ef_vi_flags flags) {
    memset(vi, 0, sizeof(*vi));
    vi->prefix_len = (flags & EF_VI_RX_TIMESTAMPS) ? 16 : 0;
    vi->flags = flags;
    vi->rxq_capacity = rxq_capacity < 0 ? 512 : (unsigned)rxq_capacity;
    vi->posted = new std::deque<std::pair<ef_addr, ef_request_id>>();
    vi->ring   = new std::deque<std::pair<ef_addr, ef_request_id>>();
    vi->events = new std::deque<ef_event>();
    vi->multi_ids = new std::deque<ef_request_id>();
    vi->multi_pending = new std::vector<ef_request_id>();
    vi->multi_batch = (flags & EF_VI_RX_EVENT_MERGE) ? 4 : 1;
    return 0;
}
static inline int ef_vi_free(ef_vi* vi, ef_driver_handle) {
    delete vi->posted; delete vi->ring; delete vi->events; delete vi->multi_ids;
    delete vi->multi_pending; return 0;
}
static inline int ef_memreg_alloc(ef_memreg* mr, ef_driver_handle, ef_pd*, ef_driver_handle,
                                  void* base, size_t len) {
    mr->base = (uint8_t*)base; mr->len = len; return 0;
}
static inline int ef_memreg_free(ef_memreg*, ef_driver_handle) { return 0; }
static inline ef_addr ef_memreg_dma_addr(ef_memreg* mr, size_t off) {
    return (ef_addr)(uintptr_t)(mr->base + off);
}
static inline int ef_vi_receive_prefix_len(ef_vi* vi) { return vi->prefix_len; }
static inline int ef_vi_receive_capacity(ef_vi* vi)   { return (int)vi->rxq_capacity; }
static inline int ef_vi_receive_space(ef_vi* vi) {
    return (int)vi->rxq_capacity - (int)(vi->ring->size() + vi->posted->size());
}
static inline int ef_vi_receive_init(ef_vi* vi, ef_addr addr, ef_request_id id) {
    if (ef_vi_receive_space(vi) <= 0) return -1;
    vi->posted->emplace_back(addr, id); return 0;
}
static inline void ef_vi_receive_push(ef_vi* vi) {
    while (!vi->posted->empty()) { vi->ring->push_back(vi->posted->front()); vi->posted->pop_front(); }
}
static inline int ef_eventq_poll(ef_vi* vi, ef_event* evs, int evs_len) {
    int n = 0;
    while (n < evs_len && !vi->events->empty()) { evs[n++] = vi->events->front(); vi->events->pop_front(); }
    return n;
}
static inline int ef_vi_receive_unbundle(ef_vi* vi, const ef_event* ev, ef_request_id* ids) {
    unsigned n = ev->rx_multi.n_descs;
    for (unsigned i = 0; i < n; ++i) { ids[i] = vi->multi_ids->front(); vi->multi_ids->pop_front(); }
    return (int)n;
}
/* mock prefix layout: [0..7] hw ns, [8..9] frame len */
static inline int ef_vi_receive_get_bytes(ef_vi*, const void* pkt, uint16_t* bytes_out) {
    memcpy(bytes_out, (const uint8_t*)pkt + 8, 2); return 0;
}
static inline int ef_vi_receive_get_timestamp_with_sync_flags(ef_vi*, const void* pkt,
                                                              struct timespec* ts, unsigned* fl) {
    uint64_t ns; memcpy(&ns, pkt, 8);
    ts->tv_sec = (time_t)(ns / 1000000000ull); ts->tv_nsec = (long)(ns % 1000000000ull);
    *fl = EF_VI_SYNC_FLAG_CLOCK_SET | EF_VI_SYNC_FLAG_CLOCK_IN_SYNC; return 0;
}
static inline void ef_filter_spec_init(ef_filter_spec* fs, enum ef_filter_flags) { memset(fs, 0, sizeof(*fs)); }
static inline int ef_filter_spec_set_ip4_local(ef_filter_spec* fs, int proto, unsigned ip_be, int port_be) {
    fs->proto = proto; fs->ip_be = ip_be; fs->port_be = (uint16_t)port_be; return 0;
}
static inline int ef_vi_filter_add(ef_vi*, ef_driver_handle, const ef_filter_spec* fs, ef_filter_cookie* c) {
    c->v = ((uint64_t)fs->ip_be << 16) | fs->port_be; return 0;
}

static inline void mock_efvi_flush_multi(ef_vi* vi);
/* ---- test-only: deliver a frame into the next posted descriptor ---- */
static inline int mock_efvi_inject(ef_vi* vi, const void* frame, uint16_t len, uint64_t hw_ns) {
    if (vi->ring->empty()) { ++vi->no_desc_drops; return -1; }
    auto [addr, id] = vi->ring->front(); vi->ring->pop_front();
    uint8_t* buf = (uint8_t*)(uintptr_t)addr;
    memcpy(buf, &hw_ns, 8); memcpy(buf + 8, &len, 2);
    memcpy(buf + vi->prefix_len, frame, len);
    if (vi->multi_batch <= 1) {
        ef_event ev{}; ev.rx.type = EF_EVENT_TYPE_RX; ev.rx.rq_id = id; ev.rx.len = len;
        ev.rx.flags = EF_EVENT_FLAG_SOP; vi->events->push_back(ev);
    } else {
        vi->multi_pending->push_back(id);
        if (vi->multi_pending->size() >= vi->multi_batch) mock_efvi_flush_multi(vi);
    }
    return 0;
}
static inline void mock_efvi_flush_multi(ef_vi* vi) {
    if (vi->multi_pending->empty()) return;
    ef_event ev{}; ev.rx_multi.type = EF_EVENT_TYPE_RX_MULTI;
    ev.rx_multi.n_descs = (unsigned)vi->multi_pending->size(); ev.rx_multi.flags = EF_EVENT_FLAG_SOP;
    for (auto id : *vi->multi_pending) vi->multi_ids->push_back(id);
    vi->multi_pending->clear(); vi->events->push_back(ev);
}
static inline void mock_efvi_inject_discard(ef_vi* vi) {
    if (vi->ring->empty()) return;
    auto [addr, id] = vi->ring->front(); vi->ring->pop_front();
    ef_event ev{}; ev.rx_discard.type = EF_EVENT_TYPE_RX_DISCARD; ev.rx_discard.rq_id = id;
    ev.rx_discard.len = 60; vi->events->push_back(ev);
}
