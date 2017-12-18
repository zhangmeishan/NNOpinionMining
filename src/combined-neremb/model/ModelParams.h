#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {
  public:
    //neural parameters
    Alphabet embeded_words; // words
    LookupTable word_table; // should be initialized outside
    Alphabet embeded_ext_words;
    LookupTable word_ext_table;

    Alphabet embeded_ners;
    LookupTable ner_table;

    BiParams word_tanh_conv;
    LSTM1Params word_left_lstm; //left lstm
    LSTM1Params word_right_lstm; //right lstm

    vector<LSTM1Params> word_left_lstm_deeper; //left lstm
    vector<LSTM1Params> word_right_lstm_deeper; //right lstm

    Alphabet embeded_ner_actions;
    ActionParams scored_action_ner_table;
    Alphabet embeded_rel_actions;
    ActionParams scored_action_rel_table;

  public:
    bool initial(HyperParams &opts) {
        word_tanh_conv.initial(opts.word_hidden_dim, opts.word_dim, opts.word_ext_dim, true);
        word_left_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim); //left lstm
        word_right_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim); //right lstm

        word_left_lstm_deeper.resize(opts.lstm_layer);
        word_right_lstm_deeper.resize(opts.lstm_layer);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
            word_right_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
        }

        scored_action_ner_table.initial(&embeded_ner_actions, opts.ner_state_concat_dim);
        scored_action_rel_table.initial(&embeded_rel_actions, opts.rel_state_concat_dim);

        return true;
    }


    void exportModelParams(ModelUpdate &ada) {
        //neural features
        word_table.exportAdaParams(ada);
        //word_ext_table.exportAdaParams(ada);
        ner_table.exportAdaParams(ada);

        word_tanh_conv.exportAdaParams(ada);
        word_left_lstm.exportAdaParams(ada);
        word_right_lstm.exportAdaParams(ada);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].exportAdaParams(ada);
            word_right_lstm_deeper[idx].exportAdaParams(ada);
        }

        scored_action_ner_table.exportAdaParams(ada);
        scored_action_rel_table.exportAdaParams(ada);
    }

    // will add it later
    void saveModel() {

    }

    void loadModel(const string &inFile) {

    }

};

#endif /* SRC_ModelParams_H_ */
