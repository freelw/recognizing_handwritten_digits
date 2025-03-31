#include <iostream>
#include "layernorm.h"
#include "attention.h"
#include "posencoding.h"
#include "addnorm.h"
#include "ffn.h"
#include "test.h"

using namespace std;


int main() {
    // test_layernorm();
    // test_softmax();
    // test_attention_without_mask();
    // test_attention_with_mask();
    // cout << "------ test_mh_attention_without_mask0 ------" << endl;
    // test_mh_attention_without_mask0();
    // cout << "------ test_mh_attention_without_mask0 end ------" << endl;

    // cout << "------ test_mh_attention_without_mask1 ------" << endl;
    // test_mh_attention_without_mask1();
    // cout << "------ test_mh_attention_without_mask1 end ------" << endl;
    // cout << "------ test_attention_to_cp_with_mha ------" << endl;
    // test_attention_to_cp_with_mha();
    // cout << "------ test_attention_to_cp_with_mha end ------" << endl;
    // test_lazy_liner();
    // test_mh_attention_with_mask();
    // test_pos_encoding();
    // test_addnorm();
    // test_ffn();
    // test_mh_attention_without_mask1();
    // test_mh_attention_without_mask2();
    // test_encoder();

    // test_mh_attention_with_2d_mask();
    test_decoder();
    return 0;
}