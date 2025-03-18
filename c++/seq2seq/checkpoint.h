#ifndef AUTOGRAD_CHECKPOINT_H
#define AUTOGRAD_CHECKPOINT_H

#include <string>
#include "seq2seq.h"

namespace autograd {
    void save_checkpoint(const std::string & prefix, int epoch, autograd::RnnLM &lm);
    void loadfrom_checkpoint(autograd::RnnLM &lm, const std::string &filename);
} // namespace autograd

#endif