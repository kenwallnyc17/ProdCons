
#include <windows.h>
#include <processthreadsapi.h>
#include <winbase.h>

#include <iostream>
#include <cstdint>
#include <cassert>
#include <bitset>

#include <streambuf>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
#include <string>
#include <list>
#include <deque>
#include <array>
#include <flat_map>
#include <initializer_list>
#include <functional>
#include <utility>
#include <concepts>
#include <type_traits>
#include <string.h>
#include <limits>
#include <optional>
#include <thread>
#include <chrono>
#include <mutex>
#include <time.h>
#include <sys/stat.h>

using namespace std;


template <typename T>
constexpr auto type_name() {
  std::string_view name, prefix, suffix;
#ifdef __clang__
  name = __PRETTY_FUNCTION__;
  prefix = "auto type_name() [T = ";
  suffix = "]";
#elif defined(__GNUC__)
  name = __PRETTY_FUNCTION__;
  prefix = "constexpr auto type_name() [with T = ";
  suffix = "]";
#elif defined(_MSC_VER)
  name = __FUNCSIG__;
  prefix = "auto __cdecl type_name<";
  suffix = ">(void)";
#endif
  name.remove_prefix(prefix.size());
  name.remove_suffix(suffix.size());
  return name;
}

//#include "BookBuilder.h"
//#include "MktDataSystem.h"
#include "MktDataSystemRun.h"

//#include "SymIdxTest.h"

int main()
{
    cout << "Hello world!" << endl;

  //  BookBuilder bb("pi_file.dat", BookBuilder::logLevel::DATA);

  //  bb();

 //   MKTDATASYSTEM::test_mktsystem_1();

   // fread();

   // int bkts{20};

    //cin >> bkts;

    startFeedHandler();

    //run_instructionset();

    return 0;
}
