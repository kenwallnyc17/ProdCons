// l3_tests.hpp — unit tests for the L3 containers (OpenTable, OrderRoute, LevelSide, Book),
// run by main_l3.cpp before the end-to-end test.
#pragma once
#include "l3.hpp"
#include <cstdio>
#include <map>
#include <random>
#include <unordered_map>

namespace rawmd::l3::test {

#define CHECK(c) do { if (!(c)) { std::printf("  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); return false; } } while (0)

// One cluster from the last home slot: fills the padding exactly, survives deletes at the
// front (backward shift runs into the sentinel), then overflows into a grow.
inline bool open_table_tail() {
  OpenTable<uint64_t> t(10);
  const std::size_t cap = t.capacity();
  std::vector<uint64_t> ks;
  for (uint64_t k = 1; ks.size() < 300; ++k) if (t.home(k) == cap - 1) ks.push_back(k);
  for (std::size_t i = 0; i < 257; ++i) CHECK(t.insert(ks[i], ks[i]).fresh);
  CHECK(t.capacity() == cap);                                       // 1023..1279: exactly cap + kPad - 1
  for (std::size_t i = 0; i < 257; ++i) { const auto j = t.find(ks[i]); CHECK(j != t.npos && t.val(j) == ks[i]); }
  for (std::size_t i = 0; i < 40; ++i) t.erase(t.find(ks[i]));
  for (std::size_t i = 0; i < 40; ++i) CHECK(t.find(ks[i]) == t.npos);
  for (std::size_t i = 40; i < 257; ++i) CHECK(t.find(ks[i]) != t.npos);
  for (std::size_t i = 0; i < 40; ++i) CHECK(t.insert(ks[i], ks[i]).fresh);
  CHECK(!t.insert(ks[5], 0).fresh);                                 // duplicate detected, value untouched
  CHECK(t.val(t.find(ks[5])) == ks[5]);
  for (std::size_t i = 257; i < 300; ++i) CHECK(t.insert(ks[i], ks[i]).fresh);   // no room in the tail: grows
  CHECK(t.capacity() > cap);
  for (auto k : ks) { const auto j = t.find(k); CHECK(j != t.npos && t.val(j) == k); }
  CHECK(t.size() == ks.size());
  return true;
}

inline bool open_table_churn() {
  std::mt19937_64 rng(7);
  std::vector<uint64_t> universe(6000);
  for (auto& u : universe) u = rng();
  universe[0] = 0;                                                  // id 0 must work
  OpenTable<uint64_t> t(3);
  std::unordered_map<uint64_t, uint64_t> ref;
  for (int op = 0; op < 3'000'000; ++op) {
    const uint64_t k = universe[rng() % universe.size()];
    const auto j = t.find(k);
    const auto it = ref.find(k);
    CHECK((j == t.npos) == (it == ref.end()));
    if (j != t.npos) CHECK(t.val(j) == it->second);
    if (rng() & 1) { const auto r = t.insert(k, k * 3); CHECK(r.fresh == (it == ref.end())); if (r.fresh) ref[k] = k * 3; }
    else if (j != t.npos) { t.erase(j); ref.erase(it); }
  }
  CHECK(t.size() == ref.size());
  for (auto& [k, v] : ref) { const auto j = t.find(k); CHECK(j != t.npos && t.val(j) == v); }
  return true;
}

inline bool order_route() {
  {   // a fixed table: the tail fills, then insert refuses instead of corrupting
    OrderRoute r(10);
    std::vector<uint64_t> ks;
    for (uint64_t k = 1; ks.size() < 1100; ++k) if (r.home(k) == r.capacity() - 1) ks.push_back(k);
    std::size_t accepted = 0;
    for (auto k : ks) if (r.insert(k, uint32_t(k), uint32_t(k >> 3)).i != OrderRoute::npos) ++accepted;
    CHECK(accepted == 1025);                                        // slots 1023 .. 1023 + 1024
    for (std::size_t i = 0; i < accepted; ++i) { const auto j = r.find(ks[i]); CHECK(j != OrderRoute::npos && r.at(j).ks == uint32_t(ks[i])); }
    for (std::size_t i = 0; i < 500; ++i) r.erase(r.find(ks[i]));
    for (std::size_t i = 0; i < 500; ++i) CHECK(r.find(ks[i]) == OrderRoute::npos);
    for (std::size_t i = 500; i < accepted; ++i) CHECK(r.find(ks[i]) != OrderRoute::npos);
    CHECK(!r.insert(ks[600], 1, 1).fresh);
  }
  std::mt19937_64 rng(11);
  OrderRoute r(14);
  std::vector<uint64_t> universe(9000);
  for (auto& u : universe) u = rng();
  universe[0] = 0;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> ref;
  for (int op = 0; op < 3'000'000; ++op) {
    const uint64_t k = universe[rng() % universe.size()];
    const auto j = r.find(k);
    const auto it = ref.find(k);
    CHECK((j == OrderRoute::npos) == (it == ref.end()));
    if (j != OrderRoute::npos) CHECK(r.at(j).ks == it->second.first && r.at(j).rem == it->second.second);
    if (rng() & 1) {
      const auto ks = uint32_t(rng()), rem = uint32_t(rng());
      const auto ins = r.insert(k, ks, rem);
      CHECK(ins.i != OrderRoute::npos && ins.fresh == (it == ref.end()));
      if (ins.fresh) ref[k] = {ks, rem};
    } else if (j != OrderRoute::npos) { r.erase(j); ref.erase(it); }
  }
  CHECK(r.size() == ref.size());
  return true;
}

// Random adds and removals against std::map; full ladder compared every 997 ops, the
// returned rank checked on every op against a counted rank.
inline bool level_side() {
  for (const bool bid : {true, false}) {
    std::mt19937_64 rng(bid ? 3 : 5);
    LevelSide s(bid);
    std::map<int64_t, std::pair<uint32_t, uint32_t>> ref;           // px -> qty, count
    std::vector<std::pair<int64_t, uint32_t>> live;
    auto rank_of = [&](int64_t px) {
      uint32_t r = 0;
      for (auto& [p, l] : ref) if (bid ? p > px : p < px) ++r;
      return r;
    };
    std::vector<Level> got(4096);
    for (int op = 0; op < 200'000; ++op) {
      if (live.empty() || (rng() % 100 < 52 && live.size() < 20000)) {
        const int64_t px = 1'000'000 + int64_t(rng() % 3000) * 100;
        const uint32_t q = 1 + uint32_t(rng() % 1000);
        const uint32_t exp = rank_of(px);
        CHECK(s.add(px, q, UINT32_MAX - 1) == exp);
        auto& l = ref[px]; l.first += q; l.second += 1;
        live.push_back({px, q});
      } else {
        const std::size_t i = rng() % live.size();
        const auto [px, q] = live[i];
        const uint32_t exp = rank_of(px);
        CHECK(s.reduce(px, q, true, UINT32_MAX - 1) == exp);
        auto it = ref.find(px); it->second.first -= q; if (--it->second.second == 0) ref.erase(it);
        live[i] = live.back(); live.pop_back();
      }
      if (op % 997 == 0) {
        CHECK(s.levels() == ref.size());
        const uint32_t n = s.top(uint32_t(got.size()), got.data());
        CHECK(n == ref.size());
        uint32_t k = 0;
        if (bid) { for (auto it = ref.rbegin(); it != ref.rend(); ++it, ++k) CHECK(got[k] == (Level{it->first, it->second.first, it->second.second})); }
        else     { for (auto it = ref.begin(); it != ref.end(); ++it, ++k) CHECK(got[k] == (Level{it->first, it->second.first, it->second.second})); }
      }
    }
  }
  return true;
}

inline bool book_bad_input() {
  Book b(5);
  CHECK(b.apply({Ev::Add, 'B', 100, 990000, 1, 0}));
  CHECK(!b.apply({Ev::Add, 'S', 300, 1010000, 1, 0}));              // id 1 reused while live
  CHECK(!b.apply({Ev::Delete, 0, 0, 0, 99, 0}));                    // unknown id
  CHECK(b.apply({Ev::Add, 'B', 200, 980000, 2, 0}));
  CHECK(b.apply({Ev::Replace, 0, 50, 970000, 2, 1}));               // new id 1 already live: old removed, counted
  Level lv[8];
  CHECK(b.top_bids(lv) == 1 && lv[0] == (Level{990000, 100, 1}));
  CHECK(b.top_asks(lv) == 0);
  CHECK(b.stats().dup_order == 2 && b.stats().unknown_order == 1 && !b.consistent());
  CHECK(b.orders() == 1);
  return true;
}

inline bool run_units() {
  bool ok = true;
  auto t = [&](const char* name, bool (*f)()) { const bool r = f(); std::printf("unit %-18s %s\n", name, r ? "ok" : "FAILED"); ok &= r; };
  t("open_table_tail", open_table_tail);
  t("open_table_churn", open_table_churn);
  t("order_route", order_route);
  t("level_side", level_side);
  t("book_bad_input", book_bad_input);
  return ok;
}
#undef CHECK
}  // namespace rawmd::l3::test
