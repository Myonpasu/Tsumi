#pragma once
#include "position.h"

uint64_t perft(const Position& pos, int depth);
uint64_t perft_divide(const Position& pos, int depth);
bool run_perft_tests();
