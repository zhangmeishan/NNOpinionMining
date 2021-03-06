#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"

struct ActionedNodes {
    PSubNode left_lstm_entity;
    PSubNode right_lstm_entity;
    ConcatNode ner_state_represent;


    PSubNode left_lstm_middle;
    PSubNode left_lstm_end;
    PSubNode left_lstm_pointi;
    PSubNode left_lstm_pointj;

    PSubNode right_lstm_middle;
    PSubNode right_lstm_start;
    PSubNode right_lstm_pointi;
    PSubNode right_lstm_pointj;
    ConcatNode rel_state_represent;

    vector<LookupNode> current_action_ner_input;
    vector<LookupNode> current_action_rel_input;
    vector<PDotNode> action_score;
    vector<PAddNode> outputs;

    BucketNode bucket_word;
    BucketNode bucket_state;
    HyperParams *opt;

  public:
    inline void initial(ModelParams &params, HyperParams &hyparams) {
        opt = &hyparams;

        left_lstm_entity.init(hyparams.word_lstm_dim, -1);
        right_lstm_entity.init(hyparams.word_lstm_dim, -1);
        ner_state_represent.init(hyparams.ner_state_concat_dim, -1);

        left_lstm_middle.init(hyparams.word_lstm_dim, -1);
        left_lstm_end.init(hyparams.word_lstm_dim, -1);
        left_lstm_pointi.init(hyparams.word_lstm_dim, -1);
        left_lstm_pointj.init(hyparams.word_lstm_dim, -1);

        right_lstm_middle.init(hyparams.word_lstm_dim, -1);
        right_lstm_start.init(hyparams.word_lstm_dim, -1);
        right_lstm_pointi.init(hyparams.word_lstm_dim, -1);
        right_lstm_pointj.init(hyparams.word_lstm_dim, -1);
        rel_state_represent.init(hyparams.rel_state_concat_dim, -1);

        current_action_ner_input.resize(hyparams.action_num);
        current_action_rel_input.resize(hyparams.action_num);
        action_score.resize(hyparams.action_num);
        outputs.resize(hyparams.action_num);

        //neural features
        for (int idx = 0; idx < hyparams.action_num; idx++) {
            current_action_ner_input[idx].setParam(&(params.scored_action_ner_table));
            current_action_ner_input[idx].init(hyparams.ner_state_concat_dim, -1);

            current_action_rel_input[idx].setParam(&(params.scored_action_rel_table));
            current_action_rel_input[idx].init(hyparams.rel_state_concat_dim, -1);

            action_score[idx].init(1, -1);
            outputs[idx].init(1, -1);
        }

        bucket_word.init(hyparams.word_lstm_dim, -1);
        bucket_state.init(hyparams.rel_state_concat_dim, -1);
    }


  public:
    inline void forward(Graph *cg, const vector<CAction> &actions, const AtomFeatures &atomFeat, PNode prevStateNode) {
        vector<PNode> sumNodes;
        CAction ac;
        int ac_num;
        int position;
        vector<PNode> states, pools_left, pools_middle, pools_right;
        ac_num = actions.size();

        bucket_word.forward(cg, 0);
        bucket_state.forward(cg, 0);
        PNode pseudo_word = &(bucket_word);
        PNode pseudo_state = &(bucket_state);


        if (!atomFeat.bRel) {
            states.clear();

            PNode  p_word_context = &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_next_position]);
            states.push_back(p_word_context);
            p_word_context = &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_next_position]);
            states.push_back(p_word_context);

            /* for (int context = 1; context <= opt->word_context; context++) {
                position = atomFeat.ner_next_position + context;
                p_word_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_left_lstm->_hiddens[position]) : pseudo_word;
                states.push_back(p_word_context);
                p_word_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_right_lstm->_hiddens[position]) : pseudo_word;
                states.push_back(p_word_context);

                position = atomFeat.ner_next_position - context;
                p_word_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_left_lstm->_hiddens[position]) : pseudo_word;
                states.push_back(p_word_context);
                p_word_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_right_lstm->_hiddens[position]) : pseudo_word;
                states.push_back(p_word_context);
            } */

            //entity-level
            PNode left_lstm_node_end = (atomFeat.ner_last_end >= 0) ? &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_last_end]) : pseudo_word;
            PNode left_lstm_node_start = (atomFeat.ner_last_start > 0) ? &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_last_start - 1]) : pseudo_word;
            left_lstm_entity.forward(cg, left_lstm_node_end, left_lstm_node_start);
            states.push_back(&left_lstm_entity);

            PNode right_lstm_node_end = (atomFeat.ner_last_end >= 0) ? &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_last_end]) : pseudo_word;
            PNode right_lstm_node_start = (atomFeat.ner_last_start > 0) ? &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_last_start - 1]) : pseudo_word;
            right_lstm_entity.forward(cg, right_lstm_node_end, right_lstm_node_start);
            states.push_back(&right_lstm_entity);

            ner_state_represent.forward(cg, states);

        } else if (atomFeat.rel_must_o == 0) {
            int i = atomFeat.rel_i;
            int start_i = atomFeat.rel_i_start;
            int j = atomFeat.rel_j;
            int start_j = atomFeat.rel_j_start;

            if (start_i >= 0 && i >= start_i && start_j > i && j >= start_j && j < atomFeat.word_size) {

            } else {
                std::cout << "" << std::endl;
            }


            states.clear();

            PNode left_lstm_left = (start_i >= 1) ? &(atomFeat.p_word_left_lstm->_hiddens[start_i - 1]) : pseudo_word;
            states.push_back(left_lstm_left);

            PNode left_lstm_node_i = &(atomFeat.p_word_left_lstm->_hiddens[i]);
            left_lstm_pointi.forward(cg, left_lstm_node_i, left_lstm_left);
            states.push_back(&left_lstm_pointi);

            left_lstm_middle.forward(cg, &(atomFeat.p_word_left_lstm->_hiddens[start_j - 1]), left_lstm_node_i);
            states.push_back(&left_lstm_middle);

            PNode left_lstm_node_j = &(atomFeat.p_word_left_lstm->_hiddens[j]);
            left_lstm_pointj.forward(cg, left_lstm_node_j, &(atomFeat.p_word_left_lstm->_hiddens[start_j - 1]));
            states.push_back(&left_lstm_pointj);

            left_lstm_end.forward(cg, &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.word_size - 1]), left_lstm_node_j);
            states.push_back(&left_lstm_end);


            PNode right_lstm_right = (j < atomFeat.word_size - 1) ? &(atomFeat.p_word_right_lstm->_hiddens[j + 1]) : pseudo_word;
            states.push_back(right_lstm_right);

            PNode right_lstm_node_j = &(atomFeat.p_word_right_lstm->_hiddens[j]);
            right_lstm_pointj.forward(cg, right_lstm_node_j, right_lstm_right);
            states.push_back(&right_lstm_pointj);

            right_lstm_middle.forward(cg, &(atomFeat.p_word_right_lstm->_hiddens[i + 1]), right_lstm_node_j);
            states.push_back(&right_lstm_middle);

            PNode right_lstm_node_i = &(atomFeat.p_word_right_lstm->_hiddens[i]);
            right_lstm_pointi.forward(cg, right_lstm_node_i, &(atomFeat.p_word_right_lstm->_hiddens[i + 1]));
            states.push_back(&right_lstm_pointi);

            right_lstm_start.forward(cg, &(atomFeat.p_word_right_lstm->_hiddens[0]), right_lstm_node_i);
            states.push_back(&right_lstm_start);

            rel_state_represent.forward(cg, states);
        } else {
            //nothing do to
        }




        for (int idx = 0; idx < ac_num; idx++) {
            ac.set(actions[idx]);
            sumNodes.clear();

            string action_name = ac.str(opt);

            if (!atomFeat.bRel) {
                current_action_ner_input[idx].forward(cg, action_name);
                action_score[idx].forward(cg, &current_action_ner_input[idx], &ner_state_represent);
            } else if (atomFeat.rel_must_o == 0) {
                current_action_rel_input[idx].forward(cg, action_name);
                action_score[idx].forward(cg, &current_action_rel_input[idx], &rel_state_represent);
            } else {
                current_action_rel_input[idx].forward(cg, action_name);
                action_score[idx].forward(cg, &current_action_rel_input[idx], pseudo_state);
            }
            sumNodes.push_back(&action_score[idx]);

            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            outputs[idx].forward(cg, sumNodes);
        }
    }
};


#endif /* SRC_ActionedNodes_H_ */
