#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "layernorm.h"
#include "attention.h"
#include "posencoding.h"
#include "addnorm.h"
#include "ffn.h"
#include "test.h"
#include "dataloader.h"
#include "optimizers.h"
#include "seq2seq.h"

using namespace std;

bool shutdown = false;

#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/vocab_fr.txt"
#define SRC_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_en_tiny.txt"
#define TGT_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_fr_tiny.txt"
#define HIDDEN_SIZE 256
#define EMBED_SIZE 256
#define TINY_HIDDEN_SIZE 2
#define TINY_EMBED_SIZE 2

#define TEST_FILE "./test.txt"
#define BATCH_SIZE 128

void signal_callback_handler(int signum);

void print_progress(uint i, uint tot) {
    std::cout << "\r[" << i << "/" << tot << "]" << std::flush;
}

std::vector<uint> trim_or_padding(const std::vector<uint> &src, uint max_len, uint pad_id) {
    std::vector<uint> res = src;
    if (src.size() > max_len) {
        res.resize(max_len);
    } else {
        res.resize(max_len, pad_id);
    }
    return res;
}

std::vector<uint> add_bos(const std::vector<uint> &src, uint bos_id) {
    std::vector<uint> res = src;
    res.insert(res.begin(), bos_id);
    return res;
}

Seq2SeqEncoderDecoder *allocEncoderDecoder(
    uint enc_vocab_size,
    uint dec_vocab_size,
    uint bos_id,
    uint eos_id,
    uint num_hiddens,
    uint num_blks,
    uint ffn_num_hiddens,
    uint num_heads,
    DATATYPE dropout
) {
    
    Encoder *encoder = new Encoder(
        enc_vocab_size, num_hiddens, 
        ffn_num_hiddens, num_heads,
        num_blks, dropout
    );
    Decoder *decoder = new Decoder(
        dec_vocab_size, num_hiddens, 
        ffn_num_hiddens, num_heads,
        num_blks, dropout
    );
    Seq2SeqEncoderDecoder *encoder_decoder = new Seq2SeqEncoderDecoder(
        encoder, decoder, bos_id, eos_id
    );
    return encoder_decoder;
}

void releaseEncoderDecoder(Seq2SeqEncoderDecoder *encoder_decoder){
    delete encoder_decoder->get_decoder();
    delete encoder_decoder->get_encoder();
    delete encoder_decoder;
}

/*
    只有在warmUp之后LazyLinear才能正常工作，才能get_parameters
    注意optimizer的初始化时机一定要在warmUp之后
    注意checkpoint的加载时机一定要在warmUp之后
*/
void warmUp(Seq2SeqEncoderDecoder *encoder_decoder) {
    std::vector<std::vector<uint>> src_token_ids = {{0}};
    std::vector<std::vector<uint>> tgt_token_ids = {{0}};
    std::vector<uint> valid_lens = {};
    std::vector<autograd::Node *> enc_out_embs;
    std::vector<autograd::Node *> dec_out_embs;
    encoder_decoder->forward(
        src_token_ids, tgt_token_ids, valid_lens,
        enc_out_embs, dec_out_embs
    );
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();
}

void train(
    const std::string &corpus,
    const std::string &checkpoint,
    uint epochs,
    DATATYPE dropout,
    DATATYPE lr,
    bool tiny) {

    uint num_steps = 9;
    std::string src_vocab_name = tiny ? SRC_VOCAB_TINY_NAME : SRC_VOCAB_NAME;
    std::string tgt_vocab_name = tiny ? TGT_VOCAB_TINY_NAME : TGT_VOCAB_NAME;
    seq2seq::DataLoader loader(corpus, src_vocab_name, tgt_vocab_name, TEST_FILE);
    std::cout << "data loaded" << std::endl;
    uint enc_vocab_size = loader.src_vocab_size();
    uint dec_vocab_size = loader.tgt_vocab_size();

    uint num_hiddens = 256;
    uint num_blks = 2;
    // DATATYPE dropout = 0.2;
    uint ffn_num_hiddens = 64;
    uint num_heads = 4;

    Seq2SeqEncoderDecoder *encoder_decoder = allocEncoderDecoder(
        enc_vocab_size, dec_vocab_size,
        loader.tgt_bos_id(), loader.tgt_eos_id(),
        num_hiddens, num_blks, ffn_num_hiddens, num_heads, dropout
    );
    warmUp(encoder_decoder);

    auto parameters = encoder_decoder->get_parameters();
    /*
    parameter size = 

    enc_vocab_size + dec_vocab_size // vocab embeddings

    + num_blks * (
        // encoder block
        4 // multi-head attention Wq Wk Wv Wo no bias
        4 // ffn dense1 dense2 with bias
        2 + 2 // addnorm1 gamma beta addnorm2 gamma beta
        // decoder block
        
        4 // self attention
        4 // enc attention
        4 // ffn
        2 + 2 + 2 // addnorm1 addnorm2 addnorm3
    ) + 2 // decoder final linear with bias
    */
    assert(
        parameters.size() == 
        enc_vocab_size + dec_vocab_size +
        num_blks * (
            4 + 4 + 2 + 2 + 4 + 4 + 4 + 2 + 2 + 2
        ) + 2
    );

    releaseEncoderDecoder(encoder_decoder);
}

int main(int argc, char *argv[]) {
    cout << "OMP_THREADS: " << OMP_THREADS << endl;
    signal(SIGINT, signal_callback_handler);
    std::string corpus;
    std::string checkpoint;
    int opt;
    uint epochs = 30;
    DATATYPE dropout = 0.2;
    DATATYPE lr = 0.005;
    bool tiny = false;
    while ((opt = getopt(argc, argv, "f:c:e:d:l:t:")) != -1) {
        switch (opt) {
            case 'f':
                corpus = optarg;
                break;
            case 'c':
                checkpoint = optarg;
                break;
            case 'e':
                epochs = atoi(optarg);
                break;
            case 'd':
                dropout = atof(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 't':
                tiny = atoi(optarg) == 1;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }

    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }
    std::cout << "epochs : " << epochs << std::endl;
    std::cout << "dropout : " << dropout << std::endl;
    std::cout << "lr : " << lr << std::endl;
    std::cout << "tiny : " << tiny << std::endl;
    train(corpus, checkpoint, epochs, dropout, lr, tiny);

    return 0;
}