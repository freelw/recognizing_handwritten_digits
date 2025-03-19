#ifndef SEQ2SEQ_DATALOADER_H
#define SEQ2SEQ_DATALOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

namespace seq2seq {

    class Vocab {
        public:
            Vocab(const std::string &vocab_file);
            ~Vocab();
            uint get_token_id(const std::string &token);
            std::string get_token(uint token_id);
            uint size();
        private:
            std::map<std::string, uint> token2id;
            std::vector<std::string> id2token;
    };

    class DataLoader {
        public:
            DataLoader(
                const std::string &_corpus_path,
                const std::string &_src_vocab_path,
                const std::string &_tgt_vocab_path
            );
            ~DataLoader();
            void get_token_ids(
                std::vector<std::vector<uint>> &src_token_ids,
                std::vector<std::vector<uint>> &tgt_token_ids
            );
            std::string get_src_token(uint token_id);
            std::string get_tgt_token(uint token_id);
            uint src_pad_id();
            uint tgt_pad_id();
            uint tgt_bos_id();
            uint src_vocab_size();
            uint tgt_vocab_size();
        private:
            std::string corpus_path;
            std::string src_vocab_path;
            std::string tgt_vocab_path;
            Vocab src_vocab;
            Vocab tgt_vocab;
    };
} // namespace seq2seq
#endif