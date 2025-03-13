#ifndef GRU_EMBEDDING_DATALOADER_H
#define GRU_EMBEDDING_DATALOADER_H
#include <fstream>
#include <map>
#include <vector>
#include <assert.h>
#include <string>

namespace gru {

    class Vocab {
        public:
            Vocab(const std::string &filename) {
                std::ifstream ifs(filename);
                std::string line;
                uint index = 0;
                while (std::getline(ifs, line)) {
                    word2index[line] = index;
                    index2word.push_back(line);
                    index++;
                }
            }
            ~Vocab() {}
            uint size() {
                return word2index.size();
            }
            uint to_index(const std::string &word){
                return word2index[word];
            }
            std::string to_word(uint index) {
                assert(index < index2word.size());
                return index2word[index];
            }
        private:
            std::map<std::string, uint> word2index;
            std::vector<std::string> index2word;
    };
    class DataLoader {
        public:
            DataLoader(const std::string &filename, const std::string &vacab_name) : vocab(vacab_name) {
                std::ifstream ifs(filename);
                std::string token;
                while (ifs >> token) {
                    token_ids.push_back(vocab.to_index(token));
                }
            }
            ~DataLoader() {}
            uint size() {
                return token_ids.size();
            }
            uint vocab_size() {
                return vocab.size();
            }
            std::string to_word(uint index) {
                return vocab.to_word(index);
            }
        public:
            std::string content;
            Vocab vocab;
            std::vector<uint> token_ids;
    };
} // namespace gru
#endif
