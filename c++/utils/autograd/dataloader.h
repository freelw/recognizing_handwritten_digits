#ifndef AUTOGRAD_DATALOADER_H
#define AUTOGRAD_DATALOADER_H


#include <fstream>
#include <assert.h>
#include "matrix/matrix.h"
#include <vector>
#include "lmcommon/common.h"

#define INPUT_NUM 28

namespace autograd {

    class DataLoader {

        public:
            DataLoader(const std::string &filename, int batch_size) {
                std::ifstream ifs(filename);
                std::string content((std::istreambuf_iterator<char>(ifs)),
                                    (std::istreambuf_iterator<char>()));
                this->content = content;
                labelss.clear();
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wsign-compare"
                for (int i = 0; i < content.size()-1; i ++) {
                #pragma GCC diagnostic pop
                    int cur_batch_size = std::min(batch_size, (int)content.size()-1 - i);
                    Matrix *m = new Matrix(Shape(INPUT_NUM, cur_batch_size));
                    m->fill(0);
                    std::vector<uint> labels;
                    for (int j = 0; j < cur_batch_size; j ++) {
                        int index = i + j;
                        #pragma GCC diagnostic push
                        #pragma GCC diagnostic ignored "-Wsign-compare"
                        assert(index+1 < content.size());
                        #pragma GCC diagnostic pop
                        assert((content[index] >= 'a' && content[index] <= 'z') || content[index] == ' ');
                        (*m)[to_index(content[index])][j] = 1;
                        labels.push_back(to_index(content[index+1]));
                    }
                    data.push_back(m);
                }
            }

            ~DataLoader() {
                for (uint i = 0; i < data.size(); i++) {
                    delete data[i];
                }
            }

        public:
            std::vector<Matrix *> data;
            std::vector<std::vector<uint>> labelss;
            std::string content;
    };
} // namespace autograd
#endif