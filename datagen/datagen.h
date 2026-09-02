// JHBR5 self-play data generation (docs/DESIGN.md §10.3).
//
//   jhbr5 datagen --value V.nn --policy P.nn --out shard.rec --games N [options]
//
// Each thread plays independent games with a one-thread search and writes
// one record per position (root visit distribution, root score, game result)
// in the format of docs/DATA_FORMAT.md.

#pragma once

namespace jhbr5::datagen {

int Run(int argc, char** argv);

}  // namespace jhbr5::datagen
