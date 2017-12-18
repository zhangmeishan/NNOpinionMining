#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"

struct ActionedNodes {
    AvgPoolNode left_lstm_entity;
    AvgPoolNode right_lstm_entity;
    ConcatNode ner_state_represent;
    LookupNode last_entity;


    AvgPoolNode left_lstm_left;
    AvgPoolNode left_lstm_entity1;
    AvgPoolNode left_lstm_middle;
    AvgPoolNode left_lstm_entity2;
    AvgPoolNode left_lstm_right;

    AvgPoolNode right_lstm_left;
    AvgPoolNode right_lstm_entity1;
    AvgPoolNode right_lstm_middle;
    AvgPoolNode right_lstm_entity2;
    AvgPoolNode right_lstm_right;

    LookupNode relation_entity1;
    LookupNode relation_entity2;

    ConcatNode rel_state_represent;

    vector<ActionNode> current_action_ner_input;
    vector<ActionNode> current_action_rel_input;

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

        left_lstm_left.init(hyparams.word_lstm_dim, -1);
        left_lstm_entity1.init(hyparams.word_lstm_dim, -1);
        left_lstm_middle.init(hyparams.word_lstm_dim, -1);
        left_lstm_entity2.init(hyparams.word_lstm_dim, -1);
        left_lstm_right.init(hyparams.word_lstm_dim, -1);

        right_lstm_left.init(hyparams.word_lstm_dim, -1);
        right_lstm_entity1.init(hyparams.word_lstm_dim, -1);
        right_lstm_middle.init(hyparams.word_lstm_dim, -1);
        right_lstm_entity2.init(hyparams.word_lstm_dim, -1);
        right_lstm_right.init(hyparams.word_lstm_dim, -1);

        rel_state_represent.init(hyparams.rel_state_concat_dim, -1);

        current_action_ner_input.resize(hyparams.action_num);
        current_action_rel_input.resize(hyparams.action_num);
        outputs.resize(hyparams.action_num);

        last_entity.setParam(&(params.ner_table));
        last_entity.init(hyparams.ner_dim, hyparams.dropProb);
        relation_entity1.setParam(&(params.ner_table));
        relation_entity1.init(hyparams.ner_dim, hyparams.dropProb);
        relation_entity2.setParam(&(params.ner_table));
        relation_entity2.init(hyparams.ner_dim, hyparams.dropProb);

        //neural features
        for (int idx = 0; idx < hyparams.action_num; idx++) {
            current_action_ner_input[idx].setParam(&(params.scored_action_ner_table));
            current_action_ner_input[idx].init(1, -1);

            current_action_rel_input[idx].setParam(&(params.scored_action_rel_table));
            current_action_rel_input[idx].init(1, -1);

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

            if (atomFeat.ner_last_start >= 0 && atomFeat.ner_last_end >= 0 && atomFeat.ner_last_end >= atomFeat.ner_last_start) {
                int length = atomFeat.ner_last_end - atomFeat.ner_last_start + 1;
                left_lstm_entity.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, atomFeat.ner_last_start, length));
                right_lstm_entity.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, atomFeat.ner_last_start, length));
                states.push_back(&left_lstm_entity);
                states.push_back(&right_lstm_entity);
            } else {
                states.push_back(pseudo_word);
                states.push_back(pseudo_word);
            }

            last_entity.forward(cg, atomFeat.ner_last_label);
            states.push_back(&last_entity);

            ner_state_represent.forward(cg, states);

        } else if (atomFeat.rel_must_o == 0) {
            int i = atomFeat.rel_i;
            int start_i = atomFeat.rel_i_start;
            int j = atomFeat.rel_j;
            int start_j = atomFeat.rel_j_start;

            states.clear();
            if (start_i >= 0 && i >= start_i && start_j > i && j >= start_j && j < atomFeat.word_size) {
                if (start_i > 0) {
                    left_lstm_left.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, 0, start_i));
                    right_lstm_left.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, 0, start_i));
                    states.push_back(&left_lstm_left);
                    states.push_back(&right_lstm_left);
                } else {
                    states.push_back(pseudo_word);
                    states.push_back(pseudo_word);
                }

                int entity1_length = i - start_i + 1;
                if (entity1_length > 0) {
                    left_lstm_entity1.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, start_i, entity1_length));
                    right_lstm_entity1.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, start_i, entity1_length));
                    states.push_back(&left_lstm_entity1);
                    states.push_back(&right_lstm_entity1);
                } else {
                    states.push_back(pseudo_word);
                    states.push_back(pseudo_word);
                }

                int middle_length = start_j - i - 1;
                if (middle_length > 0) {
                    left_lstm_middle.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, i + 1, middle_length));
                    right_lstm_middle.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, i + 1, middle_length));
                    states.push_back(&left_lstm_middle);
                    states.push_back(&right_lstm_middle);
                } else {
                    states.push_back(pseudo_word);
                    states.push_back(pseudo_word);
                }


                int entity2_length = j - start_j + 1;
                if (entity1_length > 0) {
                    left_lstm_entity2.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, start_j, entity2_length));
                    right_lstm_entity2.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, start_j, entity2_length));
                    states.push_back(&left_lstm_entity2);
                    states.push_back(&right_lstm_entity2);
                } else {
                    states.push_back(pseudo_word);
                    states.push_back(pseudo_word);
                }

                int right_length = atomFeat.word_size - j - 1;
                if (right_length > 0) {
                    left_lstm_right.forward(cg, getPNodes(atomFeat.p_word_left_lstm->_hiddens, j + 1, right_length));
                    right_lstm_right.forward(cg, getPNodes(atomFeat.p_word_right_lstm->_hiddens, j + 1, right_length));
                    states.push_back(&left_lstm_right);
                    states.push_back(&right_lstm_right);
                } else {
                    states.push_back(pseudo_word);
                    states.push_back(pseudo_word);
                }

                relation_entity1.forward(cg, atomFeat.ner1_label);
                relation_entity2.forward(cg, atomFeat.ner2_label);
                states.push_back(&relation_entity1);
                states.push_back(&relation_entity2);

            } else {
                std::cout << "relation feature extraction error" << std::endl;
                std::cout << "start_i: " << start_i << std::endl;
                std::cout << "i: " << i << std::endl;
                std::cout << "start_j: " << start_j << std::endl;
                std::cout << "j: " << j << std::endl;
                std::cout << "word size: " << atomFeat.word_size << std::endl;
            }


            rel_state_represent.forward(cg, states);
        } else {
            //nothing do to
        }




        for (int idx = 0; idx < ac_num; idx++) {
            ac.set(actions[idx]);
            sumNodes.clear();

            string action_name = ac.str(opt);

            if (!atomFeat.bRel) {
                current_action_ner_input[idx].forward(cg, action_name, &ner_state_represent);
                sumNodes.push_back(&current_action_ner_input[idx]);
            } else if (atomFeat.rel_must_o == 0) {
                current_action_rel_input[idx].forward(cg, action_name, &rel_state_represent);
                sumNodes.push_back(&current_action_rel_input[idx]);
            } else {
                current_action_rel_input[idx].forward(cg, action_name, pseudo_state);
                sumNodes.push_back(&current_action_rel_input[idx]);
            }


            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            outputs[idx].forward(cg, sumNodes);
        }
    }
};


#endif /* SRC_ActionedNodes_H_ */
