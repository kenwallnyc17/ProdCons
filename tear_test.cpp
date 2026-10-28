// tear_test.cpp — can another core observe a 64-byte AVX-512 store or load torn?
//
// Build:  g++ -O2 -std=c++20 -pthread -o tear_test tear_test.cpp      (clang++ works too)
//         No -march flag is needed; the hot loops carry their own target attributes.
//
// Quick start
//   ./tear_test --writers 2 --readers 3 --seconds 60     aligned vmovdqa64 store + load
//   ./tear_test --writers 2 --readers 3 --offset 32      control: same instruction split across lines
//   ./run_tear_matrix.sh                                 SMT / core / node / socket sweep
//
// Method
//   Each store writes eight qword lanes derived from one tag, tag = writer_id << 56 | counter:
//       lane[i] = tag + i * K          (K = 0x9E3779B97F4A7C15, so no two lanes are equal)
//   A reader loads the 64 bytes and checks lane[i] == lane[0] + i * K for every i. A load that
//   returns parts of two different stores fails the check; so does a torn qword. Decoding each
//   lane back to its tag shows where the tear boundaries fell (byte 32 for 32-byte halves).
//
//   The instructions under test are emitted with inline asm, so the compiler cannot split,
//   merge, hoist or drop them:
//       store zmm   vmovdqa64 %zmm, (obj)     (vmovdqu64 when --offset != 0)
//       load  zmm   vmovdqa64 (obj), %zmm     (vmovdqu64 when --offset != 0)
//
//   "transitions" counts reader loads that returned a different tag than that reader's
//   previous load, i.e. loads that raced a store. Zero tears only means something relative
//   to that number, so the summary also prints a rule-of-three bound (3 / transitions).
//
// Positive controls (these should tear; if one doesn't, that placement had little power):
//   --offset 32             the same single zmm instruction, split across two cache lines
//   --store ymm2 --gap 1    two 32-byte stores with a pause between them
//   --load ymm2 --gap 1     two 32-byte loads with a pause between them
//
// More store modes: q8 (eight 8-byte stores), nt (vmovntdq), movdir64b (64-byte write with
// architectural write atomicity; WC protocol, so the line leaves every cache on each store).
//
// Exit status: 0 no tears, 2 tears seen, 1 setup error.

#include <cpuid.h>
#include <dirent.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

#define T512 __attribute__((target("avx512f")))
#define HOT_INLINE __attribute__((always_inline)) inline

constexpr uint64_t kK = 0x9E3779B97F4A7C15ull;
constexpr int kIdShift = 56;
constexpr uint64_t kCtrMask = (uint64_t{1} << kIdShift) - 1;
constexpr int kMaxSamples = 64;
constexpr size_t kPage = 4096;

enum class StoreMode { Zmm, Ymm2, Q8, Nt, Movdir64b };
enum class LoadMode { Zmm, Ymm2 };
enum class Role { Writer, Reader, Flusher };

struct Config {
  std::vector<int> writers{0};
  std::vector<int> readers{1};
  int flusher = -1;
  double seconds = 10.0;
  StoreMode store = StoreMode::Zmm;
  LoadMode load = LoadMode::Zmm;
  int offset = 0;
  int gap = 0;
  int store_pause = 0;
  int load_pause = 0;
  int flush_pause = 0;
  int node = -1;
  int samples = 8;
  double progress = 0;
  bool allow_shared = false;
};

struct Shared {
  char* obj = nullptr;    // the 64-byte object under test
  char* line0 = nullptr;  // cache line holding the start of the object
  char* line1 = nullptr;  // second line when the object straddles a boundary, else null
  bool aligned = true;
  bool clflushopt = false;
  StoreMode store = StoreMode::Zmm;
  LoadMode load = LoadMode::Zmm;
  int gap = 0, store_pause = 0, load_pause = 0, flush_pause = 0;
  uint64_t nwriters = 1;
  int max_samples = 8;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::atomic<bool> pin_failed{false};
};

struct alignas(64) Stats {
  std::atomic<uint64_t> pub_ops{0}, pub_trans{0}, pub_tears{0};  // read by --progress
  alignas(64) uint64_t ops = 0;
  uint64_t trans = 0, tears = 0, invalid = 0;
  uint32_t boundaries = 0;   // bit b: lanes b and b+1 decoded to different tags (byte 8*(b+1))
  uint64_t shape[256] = {};  // bit i: lane i decoded to a different tag than lane 0
  int nsamples = 0;
  uint64_t sample[kMaxSamples][8] = {};
  Role role = Role::Reader;
  int cpu = -1;
  int id = 0;
};

std::atomic<bool> g_stop{false};

// ---------------------------------------------------------------- memory operations
// Obj64/Obj32 only give the asm memory operands the right size for alias analysis.
struct Obj64 { uint64_t q[8]; };
struct Obj32 { uint64_t q[4]; };

HOT_INLINE void spin(int n) {
  for (int i = 0; i < n; ++i) _mm_pause();
}

T512 HOT_INLINE __m512i lane_offsets() {
  return _mm512_set_epi64(static_cast<long long>(7 * kK), static_cast<long long>(6 * kK),
                          static_cast<long long>(5 * kK), static_cast<long long>(4 * kK),
                          static_cast<long long>(3 * kK), static_cast<long long>(2 * kK),
                          static_cast<long long>(kK), 0);
}

T512 HOT_INLINE void st_zmm_a(char* p, __m512i v) {
  asm volatile("vmovdqa64 %1, %0" : "=m"(*reinterpret_cast<Obj64*>(p)) : "v"(v));
}
T512 HOT_INLINE void st_zmm_u(char* p, __m512i v) {
  asm volatile("vmovdqu64 %1, %0" : "=m"(*reinterpret_cast<Obj64*>(p)) : "v"(v));
}
T512 HOT_INLINE void st_ymm(char* p, __m256i v) {
  asm volatile("vmovdqu %1, %0" : "=m"(*reinterpret_cast<Obj32*>(p)) : "x"(v));
}
T512 HOT_INLINE void st_nt(char* p, __m512i v) {
  asm volatile("vmovntdq %1, %0" : "=m"(*reinterpret_cast<Obj64*>(p)) : "v"(v));
}
HOT_INLINE void st_q(char* p, uint64_t x) {
  asm volatile("movq %1, %0" : "=m"(*reinterpret_cast<uint64_t*>(p)) : "r"(x));
}
HOT_INLINE void st_movdir64b(char* dst, const void* src) {
  asm volatile("movdir64b %1, %0" : : "r"(dst), "m"(*reinterpret_cast<const Obj64*>(src))
               : "memory");
}

T512 HOT_INLINE __m512i ld_zmm_a(const char* p) {
  __m512i v;
  asm volatile("vmovdqa64 %1, %0" : "=v"(v) : "m"(*reinterpret_cast<const Obj64*>(p)));
  return v;
}
T512 HOT_INLINE __m512i ld_zmm_u(const char* p) {
  __m512i v;
  asm volatile("vmovdqu64 %1, %0" : "=v"(v) : "m"(*reinterpret_cast<const Obj64*>(p)));
  return v;
}
T512 HOT_INLINE __m256i ld_ymm(const char* p) {
  __m256i v;
  asm volatile("vmovdqu %1, %0" : "=x"(v) : "m"(*reinterpret_cast<const Obj32*>(p)));
  return v;
}

// ---------------------------------------------------------------- thread loops
template <StoreMode M, bool Aligned>
T512 __attribute__((noinline)) void writer_loop(Shared& s, Stats& st) {
  const __m512i offs = lane_offsets();
  const uint64_t id = static_cast<uint64_t>(st.id) << kIdShift;
  const int gap = s.gap;
  const int pause = s.store_pause;
  char* const p = s.obj;
  alignas(64) uint64_t src[8];  // movdir64b source
  uint64_t ctr = 0;
  while (!g_stop.load(std::memory_order_relaxed)) {
    for (int j = 0; j < 256; ++j) {
      ++ctr;
      const uint64_t tag = id | (ctr & kCtrMask);
      if constexpr (M == StoreMode::Q8) {
        uint64_t x = tag;
        for (int i = 0; i < 8; ++i) {
          st_q(p + 8 * i, x);
          x += kK;
          if (i < 7) spin(gap);
        }
      } else {
        const __m512i v = _mm512_add_epi64(_mm512_set1_epi64(static_cast<long long>(tag)), offs);
        if constexpr (M == StoreMode::Zmm) {
          if constexpr (Aligned) st_zmm_a(p, v); else st_zmm_u(p, v);
        } else if constexpr (M == StoreMode::Ymm2) {
          st_ymm(p, _mm512_castsi512_si256(v));
          spin(gap);
          st_ymm(p + 32, _mm512_extracti64x4_epi64(v, 1));
        } else if constexpr (M == StoreMode::Nt) {
          st_nt(p, v);
        } else {
          _mm512_store_si512(src, v);
          st_movdir64b(p, src);
        }
      }
      spin(pause);
    }
    st.pub_ops.store(ctr, std::memory_order_relaxed);
  }
  st.ops = ctr;
}

T512 __attribute__((noinline, cold)) void record_tear(const Shared& s, Stats& st, __m512i v) {
  alignas(64) uint64_t lane[8];
  _mm512_store_si512(lane, v);
  uint64_t tag[8];
  for (int i = 0; i < 8; ++i) tag[i] = lane[i] - static_cast<uint64_t>(i) * kK;
  unsigned mask = 0;
  for (int i = 1; i < 8; ++i)
    if (tag[i] != tag[0]) mask |= 1u << i;
  ++st.shape[mask];
  for (int i = 0; i < 7; ++i)
    if (tag[i] != tag[i + 1]) st.boundaries |= 1u << i;
  for (int i = 0; i < 8; ++i)
    if ((tag[i] >> kIdShift) >= s.nwriters) { ++st.invalid; break; }
  if (st.nsamples < s.max_samples) std::memcpy(st.sample[st.nsamples++], tag, sizeof tag);
}

template <LoadMode M, bool Aligned>
T512 __attribute__((noinline)) void reader_loop(Shared& s, Stats& st) {
  const __m512i offs = lane_offsets();
  const int gap = s.gap;
  const int pause = s.load_pause;
  const char* const p = s.obj;
  uint64_t loads = 0, trans = 0, tears = 0, prev = 0;
  while (!g_stop.load(std::memory_order_relaxed)) {
    for (int j = 0; j < 1024; ++j) {
      __m512i v;
      if constexpr (M == LoadMode::Zmm) {
        if constexpr (Aligned) v = ld_zmm_a(p); else v = ld_zmm_u(p);
      } else {
        const __m256i lo = ld_ymm(p);
        spin(gap);
        const __m256i hi = ld_ymm(p + 32);
        v = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
      }
      const uint64_t l0 = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm512_castsi512_si128(v)));
      const __m512i expect = _mm512_add_epi64(_mm512_set1_epi64(static_cast<long long>(l0)), offs);
      const __mmask8 ok = _mm512_cmpeq_epi64_mask(v, expect);
      trans += (l0 != prev);
      prev = l0;
      if (__builtin_expect(ok != 0xFF, 0)) {
        ++tears;
        record_tear(s, st, v);
      }
      spin(pause);
    }
    loads += 1024;
    st.pub_ops.store(loads, std::memory_order_relaxed);
    st.pub_trans.store(trans, std::memory_order_relaxed);
    st.pub_tears.store(tears, std::memory_order_relaxed);
  }
  st.ops = loads;
  st.trans = trans;
  st.tears = tears;
}

void flusher_loop(Shared& s, Stats& st) {
  const int pause = s.flush_pause;
  uint64_t n = 0;
  while (!g_stop.load(std::memory_order_relaxed)) {
    for (int j = 0; j < 64; ++j) {
      if (s.clflushopt) {
        asm volatile("clflushopt %0" : "+m"(*s.line0));
        if (s.line1) asm volatile("clflushopt %0" : "+m"(*s.line1));
      } else {
        asm volatile("clflush %0" : "+m"(*s.line0));
        if (s.line1) asm volatile("clflush %0" : "+m"(*s.line1));
      }
      asm volatile("mfence" ::: "memory");
      spin(pause);
    }
    n += 64;
    st.pub_ops.store(n, std::memory_order_relaxed);
  }
  st.ops = n;
}

void run_writer(Shared& s, Stats& st) {
  switch (s.store) {
    case StoreMode::Zmm:
      if (s.aligned) writer_loop<StoreMode::Zmm, true>(s, st);
      else writer_loop<StoreMode::Zmm, false>(s, st);
      break;
    case StoreMode::Ymm2: writer_loop<StoreMode::Ymm2, false>(s, st); break;
    case StoreMode::Q8: writer_loop<StoreMode::Q8, false>(s, st); break;
    case StoreMode::Nt: writer_loop<StoreMode::Nt, true>(s, st); break;
    case StoreMode::Movdir64b: writer_loop<StoreMode::Movdir64b, true>(s, st); break;
  }
}

void run_reader(Shared& s, Stats& st) {
  if (s.load == LoadMode::Ymm2) reader_loop<LoadMode::Ymm2, false>(s, st);
  else if (s.aligned) reader_loop<LoadMode::Zmm, true>(s, st);
  else reader_loop<LoadMode::Zmm, false>(s, st);
}

bool pin_self(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) return false;
  return sched_getcpu() == cpu;
}

void thread_main(Shared* s, Stats* st) {
  if (!pin_self(st->cpu)) {
    std::fprintf(stderr, "error: cannot pin a thread to cpu %d (offline, or outside this "
                         "process's cpuset?)\n", st->cpu);
    s->pin_failed.store(true);
  }
  s->ready.fetch_add(1);
  while (!s->go.load(std::memory_order_acquire)) _mm_pause();
  if (g_stop.load()) return;
  switch (st->role) {
    case Role::Writer: run_writer(*s, *st); break;
    case Role::Reader: run_reader(*s, *st); break;
    case Role::Flusher: flusher_loop(*s, *st); break;
  }
}

// ---------------------------------------------------------------- setup helpers
bool bind_to_node(void* addr, size_t len, int node) {
  constexpr int kWords = 16;
  unsigned long mask[kWords] = {};
  if (node < 0 || node >= kWords * 64) return false;
  mask[node / 64] |= 1ul << (node % 64);
  constexpr int kMpolBind = 2, kMfStrictMove = 3;
  return syscall(SYS_mbind, addr, len, kMpolBind, mask,
                 static_cast<unsigned long>(kWords * 64 + 1), kMfStrictMove) == 0;
}

int page_node(void* addr) {
  int node = -1;
  constexpr unsigned long kNodeOfAddr = 3;  // MPOL_F_NODE | MPOL_F_ADDR
  if (syscall(SYS_get_mempolicy, &node, nullptr, 0ul, addr, kNodeOfAddr) != 0) return -1;
  return node;
}

bool parse_cpus(const std::string& spec, std::vector<int>& out) {
  out.clear();
  size_t pos = 0;
  while (pos <= spec.size()) {
    size_t end = spec.find(',', pos);
    if (end == std::string::npos) end = spec.size();
    const std::string part = spec.substr(pos, end - pos);
    if (part.empty()) return false;
    char* e = nullptr;
    const long a = std::strtol(part.c_str(), &e, 10);
    long b = a;
    if (*e == '-') b = std::strtol(e + 1, &e, 10);
    if (*e != '\0' || a < 0 || b < a || b >= CPU_SETSIZE) return false;
    for (long c = a; c <= b; ++c) out.push_back(static_cast<int>(c));
    pos = end + 1;
  }
  return !out.empty();
}

std::string join_cpus(const std::vector<int>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s;
}

std::string trim(std::string s) {
  const auto b = s.find_first_not_of(" \t");
  const auto e = s.find_last_not_of(" \t\r\n");
  return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

std::string cpuinfo_field(int cpu, const char* key) {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  bool in = false;
  while (std::getline(f, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string k = trim(line.substr(0, colon));
    const std::string v = trim(line.substr(colon + 1));
    if (k == "processor") in = std::atoi(v.c_str()) == cpu;
    else if (in && k == key) return v;
  }
  return "?";
}

std::string first_line(const std::string& path) {
  std::ifstream f(path);
  std::string s;
  if (!std::getline(f, s)) return "?";
  return trim(s);
}

int cpu_node(int cpu) {
  const std::string dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
  DIR* d = opendir(dir.c_str());
  if (!d) return -1;
  int node = -1;
  while (dirent* e = readdir(d)) {
    if (std::strncmp(e->d_name, "node", 4) == 0 && std::isdigit(static_cast<unsigned char>(e->d_name[4]))) {
      node = std::atoi(e->d_name + 4);
      break;
    }
  }
  closedir(d);
  return node;
}

std::string cpu_desc(int cpu) {
  const std::string t = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
  const int node = cpu_node(cpu);
  return "pkg " + first_line(t + "physical_package_id") + ", die " + first_line(t + "die_id") +
         ", core " + first_line(t + "core_id") + ", node " + (node >= 0 ? std::to_string(node) : "?");
}

std::string commas(uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
  return s;
}

std::string lanes_desc(unsigned mask) {
  std::string s;
  for (int i = 1; i < 8;) {
    if (!(mask >> i & 1)) { ++i; continue; }
    int j = i;
    while (j + 1 < 8 && (mask >> (j + 1) & 1)) ++j;
    if (!s.empty()) s += ",";
    s += std::to_string(i) + (j > i ? "-" + std::to_string(j) : "");
    i = j + 1;
  }
  return s;
}

std::string boundaries_str(uint32_t b, char sep) {
  std::string s;
  for (int i = 0; i < 7; ++i)
    if (b >> i & 1) s += (s.empty() ? "" : std::string(1, sep)) + std::to_string(8 * (i + 1));
  return s.empty() ? "-" : s;
}

std::string tag_str(uint64_t tag) {
  return "w" + std::to_string(tag >> kIdShift) + ":" + std::to_string(tag & kCtrMask);
}

const char* store_name(StoreMode m) {
  switch (m) {
    case StoreMode::Zmm: return "zmm";
    case StoreMode::Ymm2: return "ymm2";
    case StoreMode::Q8: return "q8";
    case StoreMode::Nt: return "nt";
    case StoreMode::Movdir64b: return "movdir64b";
  }
  return "?";
}

const char* store_instr(StoreMode m, bool aligned) {
  switch (m) {
    case StoreMode::Zmm: return aligned ? "vmovdqa64" : "vmovdqu64";
    case StoreMode::Ymm2: return "2xvmovdqu.ymm";
    case StoreMode::Q8: return "8xmovq";
    case StoreMode::Nt: return "vmovntdq";
    case StoreMode::Movdir64b: return "movdir64b";
  }
  return "?";
}

const char* load_instr(LoadMode m, bool aligned) {
  if (m == LoadMode::Ymm2) return "2xvmovdqu.ymm";
  return aligned ? "vmovdqa64" : "vmovdqu64";
}

void usage() {
  std::puts(
      "usage: tear_test [options]\n"
      "  --writers LIST       writer CPUs, e.g. 2 or 2,4 (default 0)\n"
      "  --readers LIST       reader CPUs, e.g. 3 or 3,5-7 (default 1)\n"
      "  --flusher CPU        extra thread that clflushes the line in a loop (memory round trips)\n"
      "  --seconds S          run time (default 10); Ctrl-C stops early and still reports\n"
      "  --store MODE         zmm | ymm2 | q8 | nt | movdir64b   (default zmm)\n"
      "  --load MODE          zmm | ymm2                         (default zmm)\n"
      "  --offset BYTES       0..56, multiple of 8; non-zero puts the object across a line boundary\n"
      "  --gap N              pause instructions between the parts of a ymm2/q8 store or ymm2 load\n"
      "  --store-pause N      pause instructions after each store (default 0)\n"
      "  --load-pause N       pause instructions after each load (default 0)\n"
      "  --flush-pause N      pause instructions after each flush (default 0)\n"
      "  --node N             bind the test page to NUMA node N (default: first touch on the\n"
      "                       first writer's CPU, i.e. the writer's node)\n"
      "  --samples N          tear samples to print, up to 64 (default 8)\n"
      "  --progress S         print running totals every S seconds\n"
      "  --allow-shared-cpus  allow several test threads on one CPU (small test boxes only)");
}

// Returns -1 to continue, otherwise the exit code.
int parse_args(int argc, char** argv, Config& c) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i], val;
    const auto eq = a.find('=');
    bool has_val = false;
    if (a.rfind("--", 0) == 0 && eq != std::string::npos) {
      val = a.substr(eq + 1);
      a = a.substr(0, eq);
      has_val = true;
    }
    auto need = [&](std::string& out) {
      if (has_val) { out = val; return true; }
      if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", a.c_str()); return false; }
      out = argv[++i];
      return true;
    };
    auto need_int = [&](int& out, int lo, int hi) {
      std::string v;
      if (!need(v)) return false;
      char* e = nullptr;
      const long x = std::strtol(v.c_str(), &e, 10);
      if (v.empty() || *e || x < lo || x > hi) {
        std::fprintf(stderr, "error: bad value for %s: %s\n", a.c_str(), v.c_str());
        return false;
      }
      out = static_cast<int>(x);
      return true;
    };
    auto need_double = [&](double& out) {
      std::string v;
      if (!need(v)) return false;
      char* e = nullptr;
      out = std::strtod(v.c_str(), &e);
      if (v.empty() || *e || out < 0) {
        std::fprintf(stderr, "error: bad value for %s: %s\n", a.c_str(), v.c_str());
        return false;
      }
      return true;
    };
    std::string v;
    if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (a == "--writers" || a == "--readers") {
      if (!need(v)) return 1;
      if (!parse_cpus(v, a == "--writers" ? c.writers : c.readers)) {
        std::fprintf(stderr, "error: bad CPU list for %s: %s\n", a.c_str(), v.c_str());
        return 1;
      }
    }
    else if (a == "--flusher") { if (!need_int(c.flusher, 0, CPU_SETSIZE - 1)) return 1; }
    else if (a == "--seconds") { if (!need_double(c.seconds)) return 1; }
    else if (a == "--store") {
      if (!need(v)) return 1;
      if (v == "zmm") c.store = StoreMode::Zmm;
      else if (v == "ymm2") c.store = StoreMode::Ymm2;
      else if (v == "q8") c.store = StoreMode::Q8;
      else if (v == "nt") c.store = StoreMode::Nt;
      else if (v == "movdir64b") c.store = StoreMode::Movdir64b;
      else { std::fprintf(stderr, "error: unknown store mode %s\n", v.c_str()); return 1; }
    }
    else if (a == "--load") {
      if (!need(v)) return 1;
      if (v == "zmm") c.load = LoadMode::Zmm;
      else if (v == "ymm2") c.load = LoadMode::Ymm2;
      else { std::fprintf(stderr, "error: unknown load mode %s\n", v.c_str()); return 1; }
    }
    else if (a == "--offset") { if (!need_int(c.offset, 0, 56)) return 1; }
    else if (a == "--gap") { if (!need_int(c.gap, 0, 1 << 20)) return 1; }
    else if (a == "--store-pause") { if (!need_int(c.store_pause, 0, 1 << 20)) return 1; }
    else if (a == "--load-pause") { if (!need_int(c.load_pause, 0, 1 << 20)) return 1; }
    else if (a == "--flush-pause") { if (!need_int(c.flush_pause, 0, 1 << 20)) return 1; }
    else if (a == "--node") { if (!need_int(c.node, 0, 1023)) return 1; }
    else if (a == "--samples") { if (!need_int(c.samples, 0, kMaxSamples)) return 1; }
    else if (a == "--progress") { if (!need_double(c.progress)) return 1; }
    else if (a == "--allow-shared-cpus") { c.allow_shared = true; }
    else { std::fprintf(stderr, "error: unknown option %s (see --help)\n", argv[i]); return 1; }
  }
  return -1;
}

void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  if (const int rc = parse_args(argc, argv, cfg); rc >= 0) return rc;

  if (!__builtin_cpu_supports("avx512f")) {
    std::fprintf(stderr, "error: this CPU/OS does not support AVX-512F\n");
    return 1;
  }
  unsigned a = 0, b = 0, c = 0, d = 0;
  bool has_movdir64b = false, has_clflushopt = false, hypervisor = false;
  if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
    has_movdir64b = (c >> 28) & 1;
    has_clflushopt = (b >> 23) & 1;
  }
  if (__get_cpuid(1, &a, &b, &c, &d)) hypervisor = (c >> 31) & 1;

  if (cfg.writers.size() > 255) { std::fprintf(stderr, "error: at most 255 writers\n"); return 1; }
  if (cfg.offset % 8) { std::fprintf(stderr, "error: --offset must be a multiple of 8\n"); return 1; }
  if ((cfg.store == StoreMode::Nt || cfg.store == StoreMode::Movdir64b) && cfg.offset != 0) {
    std::fprintf(stderr, "error: --store %s needs a 64-byte aligned destination (--offset 0)\n",
                 store_name(cfg.store));
    return 1;
  }
  if (cfg.store == StoreMode::Movdir64b && !has_movdir64b) {
    std::fprintf(stderr, "error: CPU does not report MOVDIR64B\n");
    return 1;
  }
  std::vector<int> used = cfg.writers;
  used.insert(used.end(), cfg.readers.begin(), cfg.readers.end());
  if (cfg.flusher >= 0) used.push_back(cfg.flusher);
  if (!cfg.allow_shared) {
    std::vector<int> sorted = used;
    std::sort(sorted.begin(), sorted.end());
    if (const auto it = std::adjacent_find(sorted.begin(), sorted.end()); it != sorted.end()) {
      std::fprintf(stderr, "error: cpu %d is used by two test threads (--allow-shared-cpus to permit)\n", *it);
      return 1;
    }
  }

  // The object lives alone on its own page: lines 0 and 1 hold nothing else.
  void* mem = mmap(nullptr, kPage, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) { std::perror("mmap"); return 1; }
  char* const page = static_cast<char*>(mem);
  cpu_set_t orig;
  CPU_ZERO(&orig);
  sched_getaffinity(0, sizeof orig, &orig);
  bool bound = false;
  if (cfg.node >= 0) {
    bound = bind_to_node(page, kPage, cfg.node);
    if (!bound)
      std::fprintf(stderr, "warning: mbind to node %d failed (%s); using first touch on cpu %d\n",
                   cfg.node, std::strerror(errno), cfg.writers[0]);
  }
  if (!bound && !pin_self(cfg.writers[0]))
    std::fprintf(stderr, "warning: could not run first touch on cpu %d\n", cfg.writers[0]);
  char* const obj = page + cfg.offset;
  for (int i = 0; i < 8; ++i)  // a consistent initial value: writer 0, counter 0
    reinterpret_cast<volatile uint64_t*>(obj)[i] = static_cast<uint64_t>(i) * kK;
  const int home = page_node(page);
  cpu_set_t rest = orig;  // keep the main thread off the test CPUs when possible
  for (int cpu : used) CPU_CLR(cpu, &rest);
  sched_setaffinity(0, sizeof(cpu_set_t), CPU_COUNT(&rest) ? &rest : &orig);

  Shared s;
  s.obj = obj;
  s.line0 = page;
  s.line1 = cfg.offset ? page + 64 : nullptr;
  s.aligned = cfg.offset == 0;
  s.clflushopt = has_clflushopt;
  s.store = cfg.store;
  s.load = cfg.load;
  s.gap = cfg.gap;
  s.store_pause = cfg.store_pause;
  s.load_pause = cfg.load_pause;
  s.flush_pause = cfg.flush_pause;
  s.nwriters = cfg.writers.size();
  s.max_samples = cfg.samples;

  const size_t nthreads = cfg.writers.size() + cfg.readers.size() + (cfg.flusher >= 0 ? 1 : 0);
  std::vector<Stats> st(nthreads);
  size_t k = 0;
  for (size_t i = 0; i < cfg.writers.size(); ++i, ++k) {
    st[k].role = Role::Writer; st[k].cpu = cfg.writers[i]; st[k].id = static_cast<int>(i);
  }
  for (size_t i = 0; i < cfg.readers.size(); ++i, ++k) {
    st[k].role = Role::Reader; st[k].cpu = cfg.readers[i]; st[k].id = static_cast<int>(i);
  }
  if (cfg.flusher >= 0) { st[k].role = Role::Flusher; st[k].cpu = cfg.flusher; st[k].id = 0; }

  // ---- header
  utsname un{};
  uname(&un);
  const int c0 = cfg.writers[0];
  std::printf("cpu:     %s\n", cpuinfo_field(c0, "model name").c_str());
  std::printf("         family %s model %s stepping %s, microcode %s, kernel %s%s\n",
              cpuinfo_field(c0, "cpu family").c_str(), cpuinfo_field(c0, "model").c_str(),
              cpuinfo_field(c0, "stepping").c_str(), cpuinfo_field(c0, "microcode").c_str(),
              un.release, hypervisor ? ", under a hypervisor" : "");
  std::printf("store:   %s (%s)%s\n", store_instr(cfg.store, s.aligned), store_name(cfg.store),
              cfg.gap && (cfg.store == StoreMode::Ymm2 || cfg.store == StoreMode::Q8)
                  ? (", gap " + std::to_string(cfg.gap) + " pause").c_str() : "");
  std::printf("load:    %s%s\n", load_instr(cfg.load, s.aligned),
              cfg.gap && cfg.load == LoadMode::Ymm2 ? (", gap " + std::to_string(cfg.gap) + " pause").c_str() : "");
  std::printf("object:  64 bytes at line offset %d (%s), page on node %s\n", cfg.offset,
              cfg.offset ? "spans two cache lines" : "one aligned cache line",
              home >= 0 ? std::to_string(home).c_str() : "unknown");
  if (cfg.store == StoreMode::Movdir64b)
    std::printf("note:    movdir64b is a WC direct store; every store evicts the line, so loads miss to memory\n");
  for (const Stats& x : st) {
    const char* r = x.role == Role::Writer ? "writer" : x.role == Role::Reader ? "reader" : "flusher";
    std::printf("%-8s %c%d on cpu %d (%s)\n", r, r[0], x.id, x.cpu, cpu_desc(x.cpu).c_str());
  }
  std::printf("running %.1f s ...\n", cfg.seconds);
  std::fflush(stdout);

  // ---- run
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  for (Stats& x : st) threads.emplace_back(thread_main, &s, &x);
  while (s.ready.load() < static_cast<int>(nthreads)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (s.pin_failed.load()) {
    g_stop.store(true);
    s.go.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();
    return 1;
  }
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  s.go.store(true, std::memory_order_release);
  double next_progress = cfg.progress;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const double el = std::chrono::duration<double>(clk::now() - t0).count();
    if (g_stop.load() || el >= cfg.seconds) break;
    if (cfg.progress > 0 && el >= next_progress) {
      uint64_t sto = 0, lo = 0, tr = 0, te = 0;
      for (const Stats& x : st) {
        if (x.role == Role::Writer) sto += x.pub_ops.load(std::memory_order_relaxed);
        if (x.role == Role::Reader) {
          lo += x.pub_ops.load(std::memory_order_relaxed);
          tr += x.pub_trans.load(std::memory_order_relaxed);
          te += x.pub_tears.load(std::memory_order_relaxed);
        }
      }
      std::printf("  [%7.1f s] stores %s  loads %s  transitions %s  tears %s\n", el, commas(sto).c_str(),
                  commas(lo).c_str(), commas(tr).c_str(), commas(te).c_str());
      std::fflush(stdout);
      next_progress += cfg.progress;
    }
  }
  g_stop.store(true);
  for (auto& t : threads) t.join();
  const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();

  // ---- report
  uint64_t stores = 0, loads = 0, trans = 0, tears = 0, invalid = 0, flushes = 0;
  uint32_t bounds = 0;
  uint64_t shape[256] = {};
  std::printf("\n%-8s %-4s %5s %18s %16s %10s\n", "role", "id", "cpu", "ops", "transitions", "tears");
  for (const Stats& x : st) {
    if (x.role == Role::Writer) {
      stores += x.ops;
      std::printf("%-8s w%-3d %5d %18s\n", "writer", x.id, x.cpu, commas(x.ops).c_str());
    } else if (x.role == Role::Reader) {
      loads += x.ops; trans += x.trans; tears += x.tears; invalid += x.invalid; bounds |= x.boundaries;
      for (int m = 0; m < 256; ++m) shape[m] += x.shape[m];
      std::printf("%-8s r%-3d %5d %18s %16s %10s\n", "reader", x.id, x.cpu, commas(x.ops).c_str(),
                  commas(x.trans).c_str(), commas(x.tears).c_str());
    } else {
      flushes += x.ops;
      std::printf("%-8s f%-3d %5d %18s\n", "flusher", x.id, x.cpu, commas(x.ops).c_str());
    }
  }
  std::printf("\nelapsed %.2f s: %.3g stores/s, %.3g loads/s, %.3g transitions/s%s\n", elapsed,
              stores / elapsed, loads / elapsed, trans / elapsed,
              flushes ? (", " + commas(flushes) + " flushes").c_str() : "");

  char bound95[32] = "-";
  if (tears == 0 && trans > 0) std::snprintf(bound95, sizeof bound95, "%.3g", 3.0 / static_cast<double>(trans));
  if (tears == 0) {
    std::printf("result: no tears in %s transitions", commas(trans).c_str());
    if (trans > 0) std::printf(" (95%% bound: tear rate < %s per transition)", bound95);
    std::printf("\n");
    if (trans == 0) std::printf("warning: readers never saw a store change the line; this run tested nothing\n");
  } else {
    // A reader can load the same torn state repeatedly, so torn loads can exceed transitions.
    std::printf("result: %s TORN LOADS (%s transitions); tear boundaries at byte %s\n",
                commas(tears).c_str(), commas(trans).c_str(), boundaries_str(bounds, ',').c_str());
    if (invalid) std::printf("        %s tears contained a lane that decodes to no writer (torn qword?)\n",
                             commas(invalid).c_str());
    std::vector<std::pair<uint64_t, unsigned>> shapes;
    for (unsigned m = 0; m < 256; ++m)
      if (shape[m]) shapes.emplace_back(shape[m], m);
    std::sort(shapes.rbegin(), shapes.rend());
    std::printf("shapes (lanes whose tag differs from lane 0):\n");
    for (size_t i = 0; i < shapes.size() && i < 8; ++i)
      std::printf("  lanes %-12s %s\n", lanes_desc(shapes[i].second).c_str(), commas(shapes[i].first).c_str());
    std::printf("samples (decoded tag per lane, lane 0 first):\n");
    for (const Stats& x : st) {
      for (int n = 0; n < x.nsamples; ++n) {
        std::string line;
        for (int i = 0; i < 8;) {
          int j = i;
          while (j + 1 < 8 && x.sample[n][j + 1] == x.sample[n][i]) ++j;
          if (!line.empty()) line += " | ";
          line += tag_str(x.sample[n][i]) + " x" + std::to_string(j - i + 1);
          i = j + 1;
        }
        std::printf("  r%d: %s\n", x.id, line.c_str());
      }
    }
  }
  std::printf("RESULT store=%s load=%s st_instr=%s ld_instr=%s offset=%d gap=%d writers=%s readers=%s "
              "flusher=%s node=%s secs=%.2f stores=%llu loads=%llu transitions=%llu tears=%llu "
              "invalid=%llu boundaries=%s bound95=%s\n",
              store_name(cfg.store), cfg.load == LoadMode::Zmm ? "zmm" : "ymm2",
              store_instr(cfg.store, s.aligned), load_instr(cfg.load, s.aligned), cfg.offset, cfg.gap,
              join_cpus(cfg.writers).c_str(), join_cpus(cfg.readers).c_str(),
              cfg.flusher >= 0 ? std::to_string(cfg.flusher).c_str() : "-",
              home >= 0 ? std::to_string(home).c_str() : "?", elapsed,
              static_cast<unsigned long long>(stores), static_cast<unsigned long long>(loads),
              static_cast<unsigned long long>(trans), static_cast<unsigned long long>(tears),
              static_cast<unsigned long long>(invalid), boundaries_str(bounds, '/').c_str(), bound95);
  return tears ? 2 : 0;
}
