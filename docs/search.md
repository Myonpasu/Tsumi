Tsumi's search is an alpha-beta search with the usual machinery: iterative deepening with aspiration
windows, principal variation search, a hash table, late move reductions and pruning, futility and reverse
futility pruning, internal iterative reductions, and lazy SMP for several threads. Moves are ordered by the
killer and counter-move heuristics and by history, including capture and continuation history. The parts
specific to selfmate:

- **Every searched node generates its legal moves**, so mates and stalemates are exact everywhere and the
  number of legal moves is known: a side with one or two legal moves is about to be forced, which the network
  is told ([Evaluation](evaluation.md)). The hash table is probed first, so a hit that ends a node costs no move
  generation.
- **No null-move pruning.** Zugzwang is the whole game, and null-move pruning assumes there is none.
- **Forcing moves first.** Checks are searched right after captures and reduced less than quiet moves, and a
  position with a single legal move is extended.
- **Quiescence search** covers captures, promotions, quiet checks at its first ply, and every evasion when in
  check. Capture and check sequences can run very deep in this game, so it is bounded: losing captures are
  skipped, after its fifth ply only recaptures are searched, and after its tenth a node not in check stands
  pat.
- **A bare king means a draw.** The side facing a bare king can never win, and the bare king wins only if
  its opponent runs out of non-mating moves, which the search has to prove move by move. Such positions are
  evaluated as a draw, shaded a little against the side with the material while the game goes on, and more
  the more squares the bare king has, so that the search drives the king into a stalemate and ends the game.
  A repetition does not count as a draw against a bare king until it can be claimed (the bare side need not
  repeat), and at the root a stalemate on the board is preferred to a draw further ahead.

The search's scores are in millipawns internally and reported in centipawns, and `score mate N` is a forced
selfmate in N moves ([Options and commands](options-and-commands.md)).
