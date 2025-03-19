#ifndef AUTOGRAD_CHECKPOINT_H
#define AUTOGRAD_CHECKPOINT_H

#include <string>
#include "seq2seq.h"

namespace autograd {
    std::string generateDateTimeSuffix();
    void save_checkpoint(const std::string & prefix, int epoch, autograd::Seq2SeqEncoderDecoder &lm);
    void loadfrom_checkpoint(autograd::Seq2SeqEncoderDecoder &lm, const std::string &filename);
} // namespace autograd

#endif