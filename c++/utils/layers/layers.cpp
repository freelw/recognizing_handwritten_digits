#include "layers.h"

#include <cmath>
#include <vector>
#include <assert.h>
#include <random>
#include <chrono>
#include <iostream>

Liner::Liner(uint i, uint o, DATATYPE sigma, bool rand) : input_num(i), output_num(o) {
    weigt = new Parameters(Shape(o, i));
    bias = new Parameters(Shape(o, 1));

    // double stddev = sqrt(2./(input_num + output_num))*sqrt(2);
    // double stddev = sigma;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count();
    
    std::default_random_engine generator_w(seed1);
    std::default_random_engine generator_b(seed2);
    std::normal_distribution<double> distribution_w(0.0, sigma);
    std::normal_distribution<double> distribution_b(0.0, sigma);

    auto w = weigt->get_weight();
    auto b = bias->get_weight();
    for (uint i = 0; i < output_num; ++ i) {
        for (uint j = 0; j < input_num; ++ j) {
            (*w)[i][j] = rand ? distribution_w(generator_w) : 0.1;
        }
    }

    for (uint i = 0; i < output_num; ++ i) {
        (*b)[i][0] = rand ? distribution_b(generator_b) : 0.1;
    }
}

Liner::~Liner() {
    delete weigt;
    delete bias;   
}

Matrix *Liner::forward(Context *ctx, Matrix *input) {
    assert(input->getShape().rowCnt == input_num);
    LinerContext *ln_ctx = (LinerContext *)ctx;
    ln_ctx->input = input;
    auto w = weigt->get_weight();
    auto b = bias->get_weight();
    Matrix *res = w->at(*input);
    
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        for (uint i = 0; i < output_num; ++ i) {
            (*res)[i][j] += (*b)[i][0];
        }
    }
    return res;
}

Matrix *Liner::backward(Context *ctx, Matrix *grad) {
    assert(grad->getShape().rowCnt == output_num);
    LinerContext *ln_ctx = (LinerContext *)ctx;
    auto w = weigt->get_weight();
    Matrix *res_grad = w->transpose()->at(*grad);
    Matrix *bias_grad = grad->sum(1);
    bias->set_grad(bias_grad);
    Matrix *weight_grad = grad->at(*(ln_ctx->input->transpose()));
    weigt->set_grad(weight_grad);
    bias_grad->checkShape(*(bias->get_grad()));
    weight_grad->checkShape(*(weigt->get_grad()));
    return res_grad;
}

void Liner::zero_grad() {
    weigt->zero_grad();
    bias->zero_grad();
}

Context *Liner::init() {
    return new LinerContext();
}

void Liner::release(Context *ctx) {
    LinerContext *ln_ctx = (LinerContext *)ctx;
    delete ln_ctx;
}

std::vector<Parameters*> Liner::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(weigt);
    res.push_back(bias);
    return res;
}

Matrix *Relu::forward(Context *ctx, Matrix *input) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    rl_ctx->input = input;
    auto shape = input->getShape();
    Matrix *res = allocTmpMatrix(shape);
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res)[i][j] = value > 0 ? value : 0;
        }
    }
    return res;
}

Matrix *Relu::backward(Context *ctx, Matrix *grad) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    auto &input = rl_ctx->input;
    input->checkShape(*grad);
    Matrix *res_grad = allocTmpMatrix(grad->getShape());
    auto shape = grad->getShape();
    for (uint i = 0; i < shape.rowCnt; ++ i) {
        for (uint j = 0; j < shape.colCnt; ++ j) {
            auto &value = (*input)[i][j];
            (*res_grad)[i][j] = value > 0 ? (*grad)[i][j] : 0;
        }
    }
    return res_grad;
}

Context *Relu::init() {
    return new ReluContext();
}

void Relu::release(Context * ctx) {
    ReluContext *rl_ctx = (ReluContext*)ctx;
    delete rl_ctx;
}

Matrix *CrossEntropyLoss::forward(Context * ctx, Matrix *input) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    ce_ctx->input = input;
    assert(input->getShape().colCnt == labels.size());
    Matrix *loss = allocTmpMatrix(Shape(1,1));
    DATATYPE loss_value = 0;
    for (uint j = 0; j < input->getShape().colCnt; ++ j) {
        DATATYPE max = (*input)[0][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            auto e = (*input)[i][j];
            if (max < e) {
                max = e;
            }
        }
        DATATYPE sum = 0;
        auto target = labels[j];
        DATATYPE zt = (*input)[target][j];
        for (uint i = 0; i < input->getShape().rowCnt; ++ i) {
            DATATYPE e = (*input)[i][j];
            e = std::exp(e-max);
            sum += e;
        }
        CrosEntropyInfo p;
        p.sum = sum;
        p.max = max;
        ce_ctx->info.push_back(p);
        loss_value += -(zt - max - log(sum));
    }
    (*loss)[0][0] = loss_value/labels.size();
    return loss;
}

Matrix *CrossEntropyLoss::backward(Context *ctx, Matrix *) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    assert(ce_ctx->info.size() == labels.size());
    Matrix *grad = allocTmpMatrix(Shape(ce_ctx->input->getShape()));
    auto batch_size = labels.size();
    for (uint i = 0; i < batch_size; ++ i) {
        DATATYPE sum = ce_ctx->info[i].sum;
        DATATYPE max = ce_ctx->info[i].max;
        auto target = labels[i];
        for (uint j = 0; j < ce_ctx->input->getShape().rowCnt; ++j) {
            if (j == target) {
                continue;
            }
            auto &_grad = (*grad)[j][i];
            _grad = std::exp((*ce_ctx->input)[j][i] - max) / sum / batch_size;
        }
        (*grad)[target][i] = (std::exp((*ce_ctx->input)[target][i] - max) / sum - 1) / batch_size;
    }
    return grad;
}

Context *CrossEntropyLoss::init() {
    return new CrossEntropyLossContext();
}

void CrossEntropyLoss::release(Context *ctx) {
    CrossEntropyLossContext *ce_ctx = (CrossEntropyLossContext*)ctx;
    delete ce_ctx;
}

Rnn::Rnn(uint i, uint h, DATATYPE _sigma, bool _rand)
    : input_num(i), hidden_num(h), sigma(_sigma), rand(_rand) {
    if (!rand) {
        std::cerr << "Warning: using fixed weight for Rnn" << std::endl;
    }
    wxh = new Parameters(Shape(hidden_num, input_num));
    whh = new Parameters(Shape(hidden_num, hidden_num));
    bh = new Parameters(Shape(hidden_num, 1));
    
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed2 = std::chrono::system_clock::now().time_since_epoch().count();
    unsigned seed3 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w1(seed1);
    std::default_random_engine generator_w2(seed2);
    std::default_random_engine generator_b(seed3);
    std::normal_distribution<DATATYPE> distribution_w1(0.0, sigma);
    std::normal_distribution<DATATYPE> distribution_w2(0.0, sigma);
    std::normal_distribution<DATATYPE> distribution_b(0.0, sigma);

    auto wxhw = wxh->get_weight();
    auto whhw = whh->get_weight();
    auto bhw = bh->get_weight();
    
    for (uint i = 0; i < hidden_num; ++ i) {
        for (uint j = 0; j < input_num; ++ j) {
            if (rand) {
                (*wxhw)[i][j] = distribution_w1(generator_w1);
            } else {
                (*wxhw)[i][j] = 0.1;
            }
        }
    }

    for (uint i = 0; i < hidden_num; ++ i) {
        if (rand) {
            (*bhw)[i][0] = distribution_b(generator_b);
        } else {
            (*bhw)[i][0] = 0.1;
        }
    }

    for (uint i = 0; i < hidden_num; ++ i) {
        for (uint j= 0; j < hidden_num; ++ j) {
            if (rand) {
                (*whhw)[i][j] = distribution_w2(generator_w2);
            } else {
                (*whhw)[i][j] = 0.1;
            }
        }
    }
}

Rnn::~Rnn() {
    delete wxh;
    delete whh;
    delete bh;
}

RnnRes Rnn::forward(Context *_ctx, const std::vector<Matrix *> &inputs, Matrix *hidden, Matrix *) {
    RnnContext *ctx = (RnnContext *)_ctx;
    assert(ctx->inputs.size() == 0);
    assert(ctx->hiddens.size() == 0);
    assert(ctx->states.size() == 0);
    assert(inputs.size() >= 1);
    ctx->inputs = inputs;
    uint batch_size = inputs[0]->getShape().colCnt;
    assert(batch_size == 1);

    // assert(!hidden);
    if (!hidden) {
        hidden = allocTmpMatrix(Shape(hidden_num, batch_size));
    }
    hidden->checkShape(Shape(hidden_num, batch_size));
    ctx->hiddens.push_back(hidden);

    RnnRes res;
    res.states.reserve(inputs.size());
    for (auto x : inputs) {
        Matrix *state = *(wxh->get_weight()->at(*x)) + *(whh->get_weight()->at(*hidden));
        Shape shape = state->getShape();
        state->checkShape(Shape(hidden_num, batch_size));
        for (uint i = 0; i < shape.rowCnt; ++ i) {
            for (uint j = 0; j < shape.colCnt; ++ j) {
                (*state)[i][j] += (*bh->get_weight())[i][0];
            }
        }
        hidden = state->tanh();
        hidden->checkShape(Shape(hidden_num, batch_size));
        res.states.push_back(hidden);
        ctx->hiddens.push_back(hidden);
        ctx->states.push_back(state);
    }
    return res;
}

Matrix *Rnn::backward(
    Context *_ctx,
    const std::vector<Matrix *> &grad_hiddens_vec) {
    RnnContext *ctx = (RnnContext *)_ctx;
    assert(ctx->inputs.size() + 1 == ctx->hiddens.size());
    assert(ctx->states.size() + 1 == ctx->hiddens.size());
    Matrix *grad = grad_hiddens_vec[grad_hiddens_vec.size()-1];
    grad->checkShape(Shape(hidden_num, 1));
    for (int i = ctx->inputs.size() - 1; i >= 0; -- i) {
        auto x = ctx->inputs[i];
        auto htminus1 = ctx->hiddens[i];
        auto state = ctx->states[i];
        grad = (*state->tanh_prime()) * *grad;
        bh->inc_grad(grad);
        Matrix *wxh_grad = grad->at(*(x->transpose()));
        wxh->inc_grad(wxh_grad);
        Matrix *whh_grad = grad->at(*(htminus1->transpose()));
        whh->inc_grad(whh_grad);
        grad = whh->get_weight()->transpose()->at(*grad);
        if (i >= 1) {
            *grad += *(grad_hiddens_vec[i-1]);
        }
    }
    grad->checkShape(Shape(hidden_num, 1));
    return grad;
}

Context *Rnn::init() {
    return new RnnContext();
}

void Rnn::release(Context *_ctx) {
    RnnContext *ctx = (RnnContext *)_ctx;
    delete ctx;
}

std::vector<Parameters*> Rnn::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(wxh);
    res.push_back(whh);
    res.push_back(bh);
    return res;
}

void Rnn::zero_grad() {
    wxh->zero_grad();
    whh->zero_grad();
    bh->zero_grad();
}

void init_weight(Matrix *weight, DATATYPE sigma) {
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator_w(seed1);
    std::normal_distribution<DATATYPE> distribution_w(0.0, sigma);
    for (uint i = 0; i < weight->getShape().rowCnt; ++ i) {
        for (uint j = 0; j < weight->getShape().colCnt; ++ j) {
            (*weight)[i][j] = distribution_w(generator_w);
        }
    }
}

LSTM::LSTM(uint i, uint h, DATATYPE _sigma, bool _rand) {
    input_num = i;
    hidden_num = h;
    sigma = _sigma;
    rand = _rand;
    if (!rand) {
        std::cerr << "Warning: using fixed weight for LSTM" << std::endl;
    }
    wxi = new Parameters(Shape(hidden_num, input_num));
    whi = new Parameters(Shape(hidden_num, hidden_num));
    wxf = new Parameters(Shape(hidden_num, input_num));
    whf = new Parameters(Shape(hidden_num, hidden_num));
    wxo = new Parameters(Shape(hidden_num, input_num));
    who = new Parameters(Shape(hidden_num, hidden_num));
    wxc = new Parameters(Shape(hidden_num, input_num));
    whc = new Parameters(Shape(hidden_num, hidden_num));
    bi = new Parameters(Shape(hidden_num, 1));
    bf = new Parameters(Shape(hidden_num, 1));
    bo = new Parameters(Shape(hidden_num, 1));
    bc = new Parameters(Shape(hidden_num, 1));
    init_weight(wxi->get_weight(), sigma);
    init_weight(whi->get_weight(), sigma);
    init_weight(wxf->get_weight(), sigma);
    init_weight(whf->get_weight(), sigma);
    init_weight(wxo->get_weight(), sigma);
    init_weight(who->get_weight(), sigma);
    init_weight(wxc->get_weight(), sigma);
    init_weight(whc->get_weight(), sigma);
    init_weight(bi->get_weight(), sigma);
    init_weight(bf->get_weight(), sigma);
    init_weight(bo->get_weight(), sigma);
    init_weight(bc->get_weight(), sigma);
}

LSTM::~LSTM() {
    delete wxi;
    delete whi;
    delete wxf;
    delete whf;
    delete wxo;
    delete who;
    delete wxc;
    delete whc;
    delete bi;
    delete bf;
    delete bo;
    delete bc;
}

RnnRes LSTM::forward(Context *ctx, const std::vector<Matrix*> &inputs, Matrix *hidden, Matrix *cell) {
    LSTMContext *lstm_ctx = (LSTMContext *)ctx;
    assert(lstm_ctx->inputs.size() == 0);
    assert(lstm_ctx->hiddens.size() == 0);
    assert(lstm_ctx->cells.size() == 0);
    assert(lstm_ctx->cells_tanh.size() == 0);
    assert(inputs.size() >= 1);
    lstm_ctx->inputs = inputs;
    uint batch_size = inputs[0]->getShape().colCnt;
    assert(batch_size == 1);

    if (!hidden) {
        hidden = allocTmpMatrix(Shape(hidden_num, batch_size));
    }
    hidden->checkShape(Shape(hidden_num, batch_size));
    lstm_ctx->hiddens.push_back(hidden);

    if (!cell) {
        cell = allocTmpMatrix(Shape(hidden_num, batch_size));
    }
    cell->checkShape(Shape(hidden_num, batch_size));
    lstm_ctx->cells.push_back(cell);
    auto cell_tanh = cell->tanh();
    lstm_ctx->cells_tanh.push_back(cell_tanh);

    RnnRes res;
    res.states.reserve(inputs.size());
    for (auto x : inputs) {
        lstm_ctx->inputs.push_back(x);
        Matrix *i = (*(wxi->get_weight()->at(*x)) + *(whi->get_weight()->at(*hidden)))->expand_add(*(bi->get_weight()));
        Matrix *f = (*(wxf->get_weight()->at(*x)) + *(whf->get_weight()->at(*hidden)))->expand_add(*(bf->get_weight()));
        Matrix *o = (*(wxo->get_weight()->at(*x)) + *(who->get_weight()->at(*hidden)))->expand_add(*(bo->get_weight()));
        Matrix *c = (*(wxc->get_weight()->at(*x)) + *(whc->get_weight()->at(*hidden)))->expand_add(*(bc->get_weight()));
        lstm_ctx->i.push_back(i);
        lstm_ctx->f.push_back(f);
        lstm_ctx->o.push_back(o);
        lstm_ctx->c.push_back(c);
        i->checkShape(Shape(hidden_num, batch_size));
        f->checkShape(Shape(hidden_num, batch_size));
        o->checkShape(Shape(hidden_num, batch_size));
        c->checkShape(Shape(hidden_num, batch_size));
        Matrix *f_sigmoid = sigmoid(*f);
        Matrix *i_sigmoid = sigmoid(*i);
        Matrix *o_sigmoid = sigmoid(*o);
        Matrix *c_tanh = c->tanh();
        lstm_ctx->f_sigmoid.push_back(f_sigmoid);
        lstm_ctx->i_sigmoid.push_back(i_sigmoid);
        lstm_ctx->o_sigmoid.push_back(o_sigmoid);
        lstm_ctx->c_tanh.push_back(c_tanh);
        f_sigmoid->checkShape(Shape(hidden_num, batch_size));
        i_sigmoid->checkShape(Shape(hidden_num, batch_size));
        o_sigmoid->checkShape(Shape(hidden_num, batch_size));
        c_tanh->checkShape(Shape(hidden_num, batch_size));
        cell = *(*f_sigmoid * *cell) + *(*i_sigmoid * *c_tanh);
        cell->checkShape(Shape(hidden_num, batch_size));
        lstm_ctx->cells.push_back(cell);
        auto cell_tanh = cell->tanh();
        lstm_ctx->cells_tanh.push_back(cell_tanh);
        hidden = *o_sigmoid * *cell_tanh;
        hidden->checkShape(Shape(hidden_num, batch_size));
        lstm_ctx->hiddens.push_back(hidden);
        res.states.push_back(hidden);
    }
    return res;
}

Matrix *LSTM::backward(
    Context *ctx,
    const std::vector<Matrix *> &grad_hiddens_vec) {
    
    LSTMContext *lstm_ctx = (LSTMContext *)ctx;
    assert(grad_hiddens_vec.size() == lstm_ctx->inputs.size());
    assert(lstm_ctx->inputs.size() + 1 == lstm_ctx->hiddens.size());
    assert(lstm_ctx->inputs.size() + 1 == lstm_ctx->cells.size());
    assert(lstm_ctx->inputs.size() + 1 == lstm_ctx->cells_tanh.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->o.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->o_sigmoid.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->i.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->i_sigmoid.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->f.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->f_sigmoid.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->c.size());
    assert(lstm_ctx->inputs.size() == lstm_ctx->c_tanh.size());

    auto grad_hidden = grad_hiddens_vec[grad_hiddens_vec.size()-1];
    Matrix *grad_cell = allocTmpMatrix(Shape(hidden_num, 1));
    
    for (int ii = lstm_ctx->inputs.size() - 1; ii >= 0; -- ii) {
        auto x = lstm_ctx->inputs[ii];
        auto hminus1 = lstm_ctx->hiddens[ii];
        auto cminus1 = lstm_ctx->cells[ii];
        auto cell = lstm_ctx->cells[ii+1];
        auto cell_tanh = lstm_ctx->cells_tanh[ii+1];
        auto o_sigmoid = lstm_ctx->o_sigmoid[ii];
        auto o = lstm_ctx->o[ii];
        auto grad_o_sigmoid = *grad_hidden * *cell_tanh;
        auto grad_cell_tanh = *grad_hidden * *o_sigmoid;
        auto _grad_cell = *grad_cell_tanh * *cell->tanh_prime();
        auto grad_o = *grad_o_sigmoid * *sigmoid_prime(*o);
        *grad_cell += *_grad_cell;
        auto f = lstm_ctx->f[ii];
        auto f_sigmoid = lstm_ctx->f_sigmoid[ii];
        auto i = lstm_ctx->i[ii];
        auto i_sigmoid = lstm_ctx->i_sigmoid[ii];
        auto c = lstm_ctx->c[ii];
        auto c_tanh = lstm_ctx->c_tanh[ii];
        auto grad_f_sigmoid = *grad_cell * *cminus1;
        auto grad_i_sigmoid = *grad_cell * *c_tanh;
        auto grad_c_tanh = *grad_cell * *i_sigmoid;
        auto grad_f = *grad_f_sigmoid * *sigmoid_prime(*f);
        auto grad_i = *grad_i_sigmoid * *sigmoid_prime(*i);
        auto grad_c = *grad_c_tanh * *c->tanh_prime();
        grad_cell = *grad_cell * *f_sigmoid;
        bi->inc_grad(grad_i);
        bf->inc_grad(grad_f);
        bo->inc_grad(grad_o);
        bc->inc_grad(grad_c);
        Matrix *wxi_grad = grad_i->at(*(x->transpose()));
        wxi->inc_grad(wxi_grad);
        Matrix *whi_grad = grad_i->at(*(hminus1->transpose()));
        whi->inc_grad(whi_grad);

        Matrix *wxf_grad = grad_f->at(*(x->transpose()));
        wxf->inc_grad(wxf_grad);
        Matrix *whf_grad = grad_f->at(*(hminus1->transpose()));
        whf->inc_grad(whf_grad);

        Matrix *wxo_grad = grad_o->at(*(x->transpose()));
        wxo->inc_grad(wxo_grad);
        Matrix *who_grad = grad_o->at(*(hminus1->transpose()));
        who->inc_grad(who_grad);

        Matrix *wxc_grad = grad_c->at(*(x->transpose()));
        wxc->inc_grad(wxc_grad);
        Matrix *whc_grad = grad_c->at(*(hminus1->transpose()));
        whc->inc_grad(whc_grad);

        grad_hidden = whi->get_weight()->transpose()->at(*grad_i);
        *grad_hidden += *(whf->get_weight()->transpose()->at(*grad_f));
        *grad_hidden += *(who->get_weight()->transpose()->at(*grad_o));
        *grad_hidden += *(whc->get_weight()->transpose()->at(*grad_c));

        if (ii >= 1) {
            *grad_hidden += *(grad_hiddens_vec[ii-1]);
        }
    }
    return grad_hidden;
}

Context *LSTM::init() {
    return new LSTMContext();
}

void LSTM::release(Context *ctx) {
    LSTMContext *lstm_ctx = (LSTMContext *)ctx;
    delete lstm_ctx;
}

std::vector<Parameters*> LSTM::get_parameters() {
    std::vector<Parameters*> res;
    res.push_back(wxi);
    res.push_back(whi);
    res.push_back(wxf);
    res.push_back(whf);
    res.push_back(wxo);
    res.push_back(who);
    res.push_back(wxc);
    res.push_back(whc);
    res.push_back(bi);
    res.push_back(bf);
    res.push_back(bo);
    res.push_back(bc);
    return res;
}

void LSTM::zero_grad() {
    wxi->zero_grad();
    whi->zero_grad();
    wxf->zero_grad();
    whf->zero_grad();
    wxo->zero_grad();
    who->zero_grad();
    wxc->zero_grad();
    whc->zero_grad();
    bi->zero_grad();
    bf->zero_grad();
    bo->zero_grad();
    bc->zero_grad();
}

