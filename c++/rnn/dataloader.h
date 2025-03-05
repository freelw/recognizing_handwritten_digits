#ifndef DATALOADER_H
#define DATALOADER_H


#include <fstream>
#include <assert.h>
#include "matrix/matrix.h"
#include <vector>

#define INPUT_NUM 28

class DataLoader {

public:
    DataLoader(const std::string &filename) {
        std::ifstream ifs(filename);
        std::string content((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));
        content = content.substr(0, 1000);
        std::cout << "content length : " << content.length() << std::endl;
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

    static uint to_index(char c) {
        if (c == ' ') {
            return 26;
        } else if (c >= 'a' && c <= 'z') {
            return c - 'a';
        }
        return 27;
    }

public:
    std::vector<Matrix *> data;
    std::vector<uint> labels;

};
#endif