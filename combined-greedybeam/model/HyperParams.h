#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"
#include <unordered_set>

struct HyperParams {
    Alphabet ner_labels; //o:0; {b,m,e,s}-agent (1,2,3,4); {b,m,e,s}-target (5,6,7,8); {b,m,e,s}-dse (9,10,11,12)
    Alphabet rel_labels; //noRel:0; AGENT-DSE, 1; TARGET-DSE, 2
    int ner_noprefix_num;
    unordered_map<string, int> word_stat;
    int maxlength;
    int action_num;
    dtype delta;
    int beam;
    int batch;

    dtype nnRegular; // for optimization
    dtype adaAlpha;  // for optimization
    dtype adaEps; // for optimization
    dtype dropProb;

    int word_dim;
    int word_lstm_dim;

    int ner_state_concat_dim;
    int rel_state_concat_dim;
    int state_hidden_dim;

    int lstm_layer;

  public:
    HyperParams() {
        beam = 1; // TODO:
        maxlength = max_step_size;
        bAssigned = false;
    }

    void setRequared(Options &opt) {
        //please specify dictionary outside

        bAssigned = true;
        beam = opt.beam;
        delta = opt.delta;
        batch = opt.batchSize;

        nnRegular = opt.regParameter;
        adaAlpha = opt.adaAlpha;
        adaEps = opt.adaEps;
        dropProb = opt.dropProb;

        word_dim = opt.wordEmbSize;
        word_lstm_dim = opt.wordRNNHiddenSize;

        ner_state_concat_dim = 4 * word_lstm_dim;
        rel_state_concat_dim = 10 * word_lstm_dim;

        state_hidden_dim = opt.state_hidden_dim; //TODO:

        lstm_layer = (opt.lstm_layer > 0) ?  opt.lstm_layer : 0;

    }

    void clear() {
        bAssigned = false;
    }

    bool bValid() {
        return bAssigned;
    }


  public:

    void print() {
        std::cout << "batch = " << batch << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "beam = " << beam << std::endl;
        std::cout << "ner_noprefix_num = " << ner_noprefix_num << std::endl;

        std::cout << "adaEps = " << adaEps << std::endl;
        std::cout << "adaAlpha = " << adaAlpha << std::endl;
        std::cout << "nnRegular = " << nnRegular << std::endl;
        std::cout << "dropProb = " << dropProb << std::endl;

        std::cout << "word_dim = " << word_dim << std::endl;
        std::cout << "word_lstm_dim = " << word_lstm_dim << std::endl;

        std::cout << "ner_state_concat_dim = " << ner_state_concat_dim << std::endl;
        std::cout << "rel_state_concat_dim = " << rel_state_concat_dim << std::endl;
        std::cout << "state_hidden_dim = " << state_hidden_dim << std::endl;
        std::cout << "lstm_layer = " << lstm_layer << std::endl;

    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */
