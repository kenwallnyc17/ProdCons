As is, it will only compile on windows.


recent gcc  with -Wall -fexceptions -g -march=corei7 -pedantic -Wfatal-errors -Wextra -Wall -std=c++20 -m64 -latomic -march=native -ggdb -pthread -Wvolatile -ftemplate-depth=10000 -mcx16 -cmain.cpp
link with libatomic.a
