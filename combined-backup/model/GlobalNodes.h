#ifndef SRC_GlobalNodes_H_
#define SRC_GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
    //sequential LSTM
    vector<vector<LookupNode> > char_inputs;
    vector<WindowBuilder> char_windows;
    vector<vector<UniNode> > char_convs;
    vector<MaxPoolNode> char_represents;

    vector<LookupNode> word_inputs;
    vector<LookupNode> word_ext_inputs;
    vector<LookupNode> tag_inputs;

    vector<ConcatNode> word_represents;
    vector<UniNode> word_tanh_conv;

    LSTM1Builder word_left_lstm;
    LSTM1Builder word_right_lstm;

    int additional_layer_num;

    vector<vector<ConcatNode> > word_deeper_represents;
    vector<LSTM1Builder> word_left_lstm_deeper;
    vector<LSTM1Builder> word_right_lstm_deeper;

  public:
    inline void resize(int max_length, int max_clength, int layer_num) {
        resizeVec(char_inputs, max_length, max_clength);
        char_windows.resize(max_length);
        resizeVec(char_convs, max_length, max_clength);
        char_represents.resize(max_length);
        for (int idx = 0; idx < max_length; idx++) {
            char_windows[idx].resize(max_clength);
            char_represents[idx].setParam(max_clength);
        }

        word_inputs.resize(max_length);
        word_ext_inputs.resize(max_length);
        tag_inputs.resize(max_length);
        word_represents.resize(max_length);
        word_tanh_conv.resize(max_length);
        word_left_lstm.resize(max_length);
        word_right_lstm.resize(max_length);

        additional_layer_num = layer_num;
        resizeVec(word_deeper_represents, additional_layer_num, max_length);
        word_left_lstm_deeper.resize(additional_layer_num);
        word_right_lstm_deeper.resize(additional_layer_num);

        for (int idx = 0; idx < additional_layer_num; idx++) {
            word_left_lstm_deeper[idx].resize(max_length);
            word_right_lstm_deeper[idx].resize(max_length);
        }


    }

  public:
    inline void initial(ModelParams &params, HyperParams &hyparams) {
        int length = word_inputs.size();
        for (int idx = 0; idx < length; idx++) {
            word_inputs[idx].setParam(&params.word_table);
            word_ext_inputs[idx].setParam(&params.word_ext_table);
            tag_inputs[idx].setParam(&params.tag_table);
            word_tanh_conv[idx].setParam(&params.word_tanh_conv); //TODO:
            for (int idy = 0; idy < char_inputs[idx].size(); idy++) {
                char_inputs[idx][idy].setParam(&params.char_table);
                char_convs[idx][idy].setParam(&params.char_tanh_conv);
            }
        }

        word_left_lstm.init(&params.word_left_lstm, hyparams.dropProb, true);
        word_right_lstm.init(&params.word_right_lstm, hyparams.dropProb, false);

        for (int idx = 0; idx < additional_layer_num; idx++) {
            for (int idy = 0; idy < length; idy++) {
                word_deeper_represents[idx][idy].init(2 * hyparams.word_lstm_dim, -1);
            }
            word_left_lstm_deeper[idx].init(&params.word_left_lstm_deeper[idx], hyparams.dropProb, true);
            word_right_lstm_deeper[idx].init(&params.word_right_lstm_deeper[idx], hyparams.dropProb, false);
        }

        for (int idx = 0; idx < length; idx++) {
            for (int idy = 0; idy < char_inputs[idx].size(); idy++) {
                char_inputs[idx][idy].init(hyparams.char_dim, hyparams.dropProb);
                char_windows[idx].init(hyparams.char_dim, hyparams.char_context);
                char_convs[idx][idy].init(hyparams.char_hidden_dim, hyparams.dropProb);
            }
            char_represents[idx].init(hyparams.char_hidden_dim, hyparams.dropProb);
            word_inputs[idx].init(hyparams.word_dim, hyparams.dropProb);
            word_ext_inputs[idx].init(hyparams.word_ext_dim, hyparams.dropProb);
            tag_inputs[idx].init(hyparams.tag_dim, hyparams.dropProb);
            word_represents[idx].init(hyparams.word_represent_dim, -1);
            word_tanh_conv[idx].init(hyparams.word_hidden_dim, hyparams.dropProb);
        }
    }


  public:
    inline void forward(Graph* cg, const Instance& inst, HyperParams* hyparams) {
        int word_size = inst.words.size();
        string currWord, currPos;
        for (int idx = 0; idx < word_size; idx++) {
            currWord = normalize_to_lower(inst.words[idx]);
            currPos = inst.tags[idx];

            int char_size = inst.chars[idx].size();
            if (char_size > char_inputs[idx].size()) {
                char_size = char_inputs[idx].size();
            }
            for (int idy = 0; idy < char_size; idy++) {
                char_inputs[idx][idy].forward(cg, inst.chars[idx][idy]);
            }
            char_windows[idx].forward(cg, getPNodes(char_inputs[idx], char_size));
            for (int idy = 0; idy < char_size; idy++) {
                char_convs[idx][idy].forward(cg, &(char_windows[idx]._outputs[idy]));
            }
            char_represents[idx].forward(cg, getPNodes(char_convs[idx], char_size));

            word_ext_inputs[idx].forward(cg, currWord);


            // Unknown word strategy: STOCHASTIC REPLACEMENT

            int c = hyparams->word_stat[currWord];
            dtype rand_drop = rand() / double(RAND_MAX);
            if (cg->train && c <= 1 && rand_drop < 0.5) {
                currWord = unknownkey;
            }


            word_inputs[idx].forward(cg, currWord);
            tag_inputs[idx].forward(cg, currPos);
            word_represents[idx].forward(cg, &word_inputs[idx], &word_ext_inputs[idx], &tag_inputs[idx], &(char_represents[idx]));
        }

        for (int idx = 0; idx < word_size; idx++) {
            word_tanh_conv[idx].forward(cg, &(word_represents[idx]));
        }
        word_left_lstm.forward(cg, getPNodes(word_tanh_conv, word_size));
        word_right_lstm.forward(cg, getPNodes(word_tanh_conv, word_size));

        for (int idx = 0; idx < additional_layer_num; idx++) {
            for (int idy = 0; idy < word_size; idy++) {
                if (idx == 0) word_deeper_represents[idx][idy].forward(cg, &(word_left_lstm._hiddens[idy]), &(word_right_lstm._hiddens[idy]));
                else word_deeper_represents[idx][idy].forward(cg, &(word_left_lstm_deeper[idx - 1]._hiddens[idy]), &(word_right_lstm_deeper[idx - 1]._hiddens[idy]));
            }

            word_left_lstm_deeper[idx].forward(cg, getPNodes(word_deeper_represents[idx], word_size));
            word_right_lstm_deeper[idx].forward(cg, getPNodes(word_deeper_represents[idx], word_size));
        }
    }

};

#endif /* SRC_GlobalNodes_H_ */
