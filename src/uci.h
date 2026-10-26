#pragma once

void init_engine();   // one-time table initialisation and NNUE kernel selection; called first from main()
int uci_loop(int argc, char** argv);
int datagen_main(int argc, char** argv);
int rescore_main(int argc, char** argv);
