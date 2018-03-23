#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"
#include <unordered_set>

struct HyperParams {
    Alphabet ner_labels;
    Alphabet rel_labels;
    unordered_map<string, unordered_set<int> > rel_dir;
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
    int word_ext_dim;

    int word_hidden_dim;
    int word_lstm_dim;

    int ner_dim;


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
        word_ext_dim = opt.wordExtEmbSize;

        word_hidden_dim = opt.wordHiddenSize;
        word_lstm_dim = opt.wordRNNHiddenSize;

        ner_dim = opt.nerEmbSize;


        ner_state_concat_dim = 4 * word_lstm_dim + ner_dim;
        rel_state_concat_dim = 10 * word_lstm_dim + 2 * ner_dim;

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
        std::cout << "word_ext_dim = " << word_ext_dim << std::endl;
        std::cout << "word_hidden_dim = " << word_hidden_dim << std::endl;
        std::cout << "word_lstm_dim = " << word_lstm_dim << std::endl;

        std::cout << "ner_dim = " << ner_dim << std::endl;

        std::cout << "ner_state_concat_dim = " << ner_state_concat_dim << std::endl;
        std::cout << "rel_state_concat_dim = " << rel_state_concat_dim << std::endl;
        std::cout << "state_hidden_dim = " << state_hidden_dim << std::endl;
        std::cout << "lstm_layer = " << lstm_layer << std::endl;

    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */
