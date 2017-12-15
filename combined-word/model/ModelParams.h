#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {
  public:
    //neural parameters
    Alphabet embeded_words; // words
    LookupTable word_table; // should be initialized outside

    LSTM1Params word_left_lstm; //left lstm
    LSTM1Params word_right_lstm; //right lstm

    vector<LSTM1Params> word_left_lstm_deeper; //left lstm
    vector<LSTM1Params> word_right_lstm_deeper; //right lstm

    UniParams ner_state_hidden;
    UniParams rel_state_hidden;

    Alphabet embeded_actions;
    LookupTable scored_action_table;

  public:
    bool initial(HyperParams &opts) {
        word_left_lstm.initial(opts.word_lstm_dim, opts.word_dim); //left lstm
        word_right_lstm.initial(opts.word_lstm_dim, opts.word_dim); //right lstm

        word_left_lstm_deeper.resize(opts.lstm_layer);
        word_right_lstm_deeper.resize(opts.lstm_layer);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
            word_right_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
        }


        ner_state_hidden.initial(opts.state_hidden_dim, opts.ner_state_concat_dim, false);
        rel_state_hidden.initial(opts.state_hidden_dim, opts.rel_state_concat_dim, false);
        scored_action_table.initial(&embeded_actions, opts.state_hidden_dim, true);

        return true;
    }


    void exportModelParams(ModelUpdate &ada) {
        //neural features
        word_table.exportAdaParams(ada);

        word_left_lstm.exportAdaParams(ada);
        word_right_lstm.exportAdaParams(ada);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].exportAdaParams(ada);
            word_right_lstm_deeper[idx].exportAdaParams(ada);
        }

        ner_state_hidden.exportAdaParams(ada);
        rel_state_hidden.exportAdaParams(ada);
        scored_action_table.exportAdaParams(ada);
    }

    // will add it later
    void saveModel() {

    }

    void loadModel(const string &inFile) {

    }

};

#endif /* SRC_ModelParams_H_ */
