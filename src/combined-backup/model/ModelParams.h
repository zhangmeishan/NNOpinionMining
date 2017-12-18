#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

  public:
    //neural parameters
    Alphabet embeded_chars; // chars
    LookupTable char_table; // should be initialized outside
    Alphabet embeded_words; // words
    LookupTable word_table; // should be initialized outside
    Alphabet embeded_ext_words;
    LookupTable word_ext_table;
    Alphabet embeded_tags; // tags
    LookupTable tag_table; // should be initialized outside
    Alphabet embeded_ners;
    LookupTable ner_table; // should be initialized outside
    Alphabet embeded_actions;

    UniParams char_tanh_conv;
    UniParams word_tanh_conv;

    LSTM1Params word_left_lstm; //left lstm
    LSTM1Params word_right_lstm; //right lstm

    vector<LSTM1Params> word_left_lstm_deeper; //left lstm
    vector<LSTM1Params> word_right_lstm_deeper; //right lstm

    LSTM1Params ner_lstm;

    UniParams ner_state_hidden;
    UniParams rel_state_hidden;
    LookupTable scored_action_table;

  public:
    bool initial(HyperParams &opts) {
        char_tanh_conv.initial(opts.char_hidden_dim, opts.char_represent_dim, true);

        word_tanh_conv.initial(opts.word_hidden_dim, opts.word_represent_dim, true);
        word_left_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim); //left lstm
        word_right_lstm.initial(opts.word_lstm_dim, opts.word_hidden_dim); //right lstm

        word_left_lstm_deeper.resize(opts.lstm_layer);
        word_right_lstm_deeper.resize(opts.lstm_layer);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
            word_right_lstm_deeper[idx].initial(opts.word_lstm_dim, 2 * opts.word_lstm_dim);
        }


        ner_lstm.initial(opts.ner_lstm_dim, opts.ner_dim);

        ner_state_hidden.initial(opts.state_hidden_dim, opts.ner_state_concat_dim, true);
        rel_state_hidden.initial(opts.state_hidden_dim, opts.rel_state_concat_dim, true);
        scored_action_table.initial(&embeded_actions, opts.state_hidden_dim, true);
        scored_action_table.E.val.random(0.01);

        return true;
    }


    void exportModelParams(ModelUpdate &ada) {
        //neural features
        char_table.exportAdaParams(ada);
        word_table.exportAdaParams(ada);
        //word_ext_table.exportAdaParams(ada);
        tag_table.exportAdaParams(ada);
        ner_table.exportAdaParams(ada);

        char_tanh_conv.exportAdaParams(ada);
        word_tanh_conv.exportAdaParams(ada);

        word_left_lstm.exportAdaParams(ada);
        word_right_lstm.exportAdaParams(ada);

        for (int idx = 0; idx < word_left_lstm_deeper.size(); idx++) {
            word_left_lstm_deeper[idx].exportAdaParams(ada);
            word_right_lstm_deeper[idx].exportAdaParams(ada);
        }

        ner_lstm.exportAdaParams(ada);

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
