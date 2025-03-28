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
                index2word.push_back("<unk>");
                uint index = 1; // 0 is <unk>
                while (std::getline(ifs, line)) {
                    word2index[line] = index;
                    index2word.push_back(line);
                    index++;
                }
            }
            ~Vocab() {}
            uint size() {
                return word2index.size() + 1; // 0 is <unk>
            }
            uint to_index(const std::string &word) const {
                if (word2index.find(word) == word2index.end()) {
                    return 0; // 0 is <unk>
                }
                return word2index.find(word)->second;
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
            DataLoader(
                const std::string &filename,
                const std::string &vacab_name) : vocab(vacab_name) {
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
            uint get_token_id(uint index) {
                return token_ids[index];
            }
            std::vector<uint> to_token_ids(const std::string &sentence) {
                std::vector<uint> res;
                std::string token;
                std::istringstream iss(sentence);
                while (iss >> token) {
                    res.push_back(vocab.to_index(token));
                }
                return res;
            }
        private:
            Vocab vocab;
            std::vector<uint> token_ids;
    };
} // namespace gru
#endif
