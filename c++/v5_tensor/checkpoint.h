#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "optimizers/parameter.h"

void save_checkpoint(const std::string & prefix, int epoch, const std::vector<Parameter*> &parameters);

#endif