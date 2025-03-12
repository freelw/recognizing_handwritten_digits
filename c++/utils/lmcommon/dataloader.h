#ifndef DATALOADER_H
#define DATALOADER_H


#include <fstream>
#include <assert.h>
#include "matrix/matrix.h"
#include <vector>
#include "lmcommon/common.h"

#define INPUT_NUM 28

class DataLoader {

public:
    DataLoader(const std::string &filename) {
        std::ifstream ifs(filename);
        std::string _content((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
        content = _content;
        for (uint i = 0; i < content.size(); i++) {
            assert((content[i] >= 'a' && content[i] <= 'z') || content[i] == ' ');
            Matrix *m = new Matrix(Shape(INPUT_NUM, 1));
            (*m)[to_index(content[i])][0] = 1;
            data.push_back(m);
            labels.push_back(to_index(content[i]));
        }
    }

    ~DataLoader() {
        for (uint i = 0; i < data.size(); i++) {
            delete data[i];
        }
    }

public:
    std::vector<Matrix *> data;
    std::vector<uint> labels;
    std::string content;

};
#endif