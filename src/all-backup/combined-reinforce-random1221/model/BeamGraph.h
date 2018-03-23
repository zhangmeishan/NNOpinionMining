#ifndef SRC_BeamGraph_H_
#define SRC_BeamGraph_H_

#include "ModelParams.h"
#include "State.h"


// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
class BeamGraphBuilder {

  public:
    GlobalNodes globalNodes;
    // node instances
    CStateItem start;
    vector<vector<CStateItem> > states;
    vector<vector<COutput> > outputs;

  private:
    ModelParams *pModel;
    HyperParams *pOpts;

    // node pointers
  public:
    BeamGraphBuilder() {
        clear();
    }

    ~BeamGraphBuilder() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void initial(ModelParams& model, HyperParams& opts) {
        std::cout << "state size: " << sizeof(CStateItem) << std::endl;
        std::cout << "action node size: " << sizeof(ActionedNodes) << std::endl;
        globalNodes.resize(max_token_size, max_word_length, opts.lstm_layer);
        states.resize(opts.maxlength + 1);

        globalNodes.initial(model, opts);
        for (int idx = 0; idx < states.size(); idx++) {
            states[idx].resize(opts.beam);
            for (int idy = 0; idy < states[idx].size(); idy++) {
                states[idx][idy].initial(model, opts);
            }
        }
        start.clear();
        start.initial(model, opts);

        pModel = &model;
        pOpts = &opts;
    }

    inline void clear() {
        //beams.clear();
        clearVec(outputs);
        clearVec(states);
        pModel = NULL;
        pOpts = NULL;
    }

  public:
    inline void encode(Graph* pcg, Instance& inst) {
        globalNodes.forward(pcg, inst, pOpts);
    }

  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void decode(Graph* pcg, Instance& inst, bool nerOnly, const vector<CAction>* goldAC = NULL) {
        //first step, clear node values
        clearVec(outputs);

        //second step, build graph
        vector<CStateItem*> lastStates;
        CStateItem* pGenerator;
        int step, offset;
        vector<vector<CAction> > actions; // actions to apply for a candidate
        CScoredState scored_action; // used rank actions
        COutput output;
        bool correct_action_scored;
        bool correct_in_beam;
        CAction answer, action;
        vector<COutput> per_step_output;
        vector<CScoredState> scored_states;
        vector<CScoredState> beam_states;

        CStateItem* pGoldGenerator;
        PNode pGoldScore;

        actions.resize(pOpts->beam);

        lastStates.clear();
        start.setInput(inst);
        lastStates.push_back(&start);

        step = 0;
        while (true) {
            //prepare for the next
            pGoldGenerator = NULL;
            pGoldScore = NULL;
            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                if (pcg->train && pGenerator->_bGold) pGoldGenerator = pGenerator;
                pGenerator->prepare(pOpts, pModel, &globalNodes);
            }

            if (pcg->train && pGoldGenerator == NULL) {
                std::cout << "gold state is doppped, bug exists" << std::endl;
            }


            answer.clear();
            per_step_output.clear();
            correct_action_scored = false;
            if (pcg->train && step < goldAC->size()) answer = (*goldAC)[step];

            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                pGenerator->getCandidateActions(actions[idx], pOpts, pModel, false);
                if (pcg->train && nerOnly && pGenerator->allow_rel()) {
                    actions[idx].clear();
                    if (pGenerator->_bGold)actions[idx].push_back(answer);
                    else {
                        action.set(CAction::REL, 0);
                        actions[idx].push_back(action);
                    }
                }
                pGenerator->computeNextScore(pcg, actions[idx], true);
            }

            pcg->compute(); //must compute here, or we can not obtain the scores
            scored_states.clear();
            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                scored_action.item = pGenerator;
                output.curState = pGenerator;
                for (int idy = 0; idy < actions[idx].size(); ++idy) {
                    scored_action.ac.set(actions[idx][idy]); //TODO:
                    output.ac.set(actions[idx][idy]); //TODO:
                    if (pGenerator->_bGold && actions[idx][idy] == answer) {
                        scored_action.bGold = true;
                        correct_action_scored = true;
                        output.bGold = true;
                    } else {
                        scored_action.bGold = false;
                        output.bGold = false;
                    }
                    if (pcg->train && actions[idx][idy] != answer)pGenerator->_nextscores.outputs[idy].val[0] += pOpts->delta;
                    scored_action.score = pGenerator->_nextscores.outputs[idy].val[0];
                    scored_action.position = idy;
                    output.in = &(pGenerator->_nextscores.outputs[idy]);
                    if (output.bGold) pGoldScore = output.in;
                    scored_states.push_back(scored_action);
                    per_step_output.push_back(output);
                }
            }

            outputs.push_back(per_step_output);

            if (pcg->train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
                std::cout << answer.str(pOpts) << std::endl;
                for (int idx = 0; idx < lastStates.size(); idx++) {
                    pGenerator = lastStates[idx];
                    if (pGenerator->_bGold) {
                        pGenerator->getCandidateActions(actions[idx], pOpts, pModel, false);
                        for (int idy = 0; idy < actions[idx].size(); ++idy) {
                            std::cout << actions[idx][idy].str(pOpts) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                return;
            }

            reduce_and_sort(scored_states, beam_states, pcg->train);
            offset = beam_states.size();
            if (offset == 0) { // judge correctiveness
                std::cout << "error, reach no output here, please find why" << std::endl;
                std::cout << "" << std::endl;
                return;
            }

            bool bGoldAdded = false;
            for (int idx = 0; idx < offset; idx++) {
                pGenerator = beam_states[idx].item;
                action.set(beam_states[idx].ac);
                pGenerator->move(&(states[step][idx]), action);
                states[step][idx]._bGold = beam_states[idx].bGold;
                if (states[step][idx]._bGold) bGoldAdded = true;
                states[step][idx]._score = &(pGenerator->_nextscores.outputs[beam_states[idx].position]);
            }

            //last element
            if (pcg->train && !bGoldAdded) {
                std::cout << "strange here, please check, gold state must be in here" << std::endl;
            }

            bool bAllEnd = true;
            for (int idx = 0; idx < offset; idx++) {
                if (!states[step][idx]._bEnd) {
                    bAllEnd = false;
                    break;
                }
            }

            if (bAllEnd) {
                break;
            }


//for next step
            lastStates.clear();
            correct_in_beam = false;
            for (int idx = 0; idx < offset; idx++) {
                lastStates.push_back(&(states[step][idx]));
                if (lastStates[idx]->_bGold) {
                    correct_in_beam = true;
                }
            }

            if (pcg->train && !correct_in_beam) {
                break;
            }

            step++;
        }

        //check
        if (pcg->train) {
            vector<CAction> testGoldAcs;
            pGoldGenerator = NULL;
            for (int idx = 0; idx < offset; idx++) {
                if (states[step][idx]._bGold) {
                    pGoldGenerator = &states[step][idx];
                    break;
                }
            }

            if (pGoldGenerator == NULL) {
                std::cout << "strange gold standard state has been filtered" << std::endl;
            }

            while (pGoldGenerator && !pGoldGenerator->_bStart) {
                if (!pGoldGenerator->_lastAction.isNone()) {
                    testGoldAcs.insert(testGoldAcs.begin(), pGoldGenerator->_lastAction);
                }
                pGoldGenerator = pGoldGenerator->_prevState;
                if (!pGoldGenerator->_bGold) {
                    std::cout << "strange traces for gold state" << std::endl;
                }
            }

            if (goldAC->size() != testGoldAcs.size()) {
                std::cout << "strange traces for gold actions" << std::endl;
            }

            for (int idx = 0; idx < testGoldAcs.size(); idx++) {
                if (testGoldAcs[idx] != (*goldAC)[idx]) {
                    std::cout << "Different Action: " << testGoldAcs[idx].str(pOpts) << " " << (*goldAC)[idx].str(pOpts) << std::endl;
                }
            }
        }

        return;
    }


    inline void reduce_and_sort(const vector<CScoredState>& scored_states, vector<CScoredState>& beam_states, bool bGoldIn) {
        beam_states.clear();

        int num = scored_states.size();
        if (num == 0) {
            std::cout << "candidates could not be smaller than one" << std::endl;
            return;
        }

        dtype max_value = scored_states[0].score;
        for (int idx = 1; idx < num; idx++) {
            if (scored_states[idx].score > max_value) {
                max_value = scored_states[idx].score;
            }
        }

        vector<dtype> probs;
        probs.resize(num);
        dtype sum = 0;
        for (int idx = 0; idx < num; idx++) {
            probs[idx] = exp(scored_states[idx].score - max_value);
            if (probs[idx] < 1e-8)probs[idx] = 1e-8;
            sum += probs[idx];
        }

        for (int idx = 0; idx < num; idx++) {
            probs[idx] = probs[idx] / sum;
        }


        vector<LabelScore> idxscores;
        idxscores.resize(num);
        int goldIndex = -1;
        for (int idx = 0; idx < num; idx++) {
            dtype randv = rand() / dtype(RAND_MAX);
            idxscores[idx].labelId = idx;
            idxscores[idx].score = probs[idx] * randv;

            if (bGoldIn && scored_states[idx].bGold) {
                idxscores[idx].score = 2.0;
                goldIndex = idx;
            }
        }

        std::sort(idxscores.begin(), idxscores.end(), label_compare);


        for (int idx = 0; idx < pOpts->beam && idx < num; idx++) {
            int stateId = idxscores[idx].labelId;
            beam_states.push_back(scored_states[stateId]);
        }

        if (!beam_states[0].bGold) {
            std::cout << "check here, goldIndex = " << goldIndex << std::endl;
            for (int idx = 0; idx < num; idx++) {
                std::cout << idxscores[idx].labelId << " " << idxscores[idx].score << std::endl;
            }

            std::cout << "max_value" << max_value << ", sum = " << sum << std::endl;
            for (int idx = 0; idx < num; idx++) {
                std::cout << "probs[" << idx << "] = " << probs[idx] << std::endl;
            }

            std::cout << "orginal scores" << std::endl;
            for (int idx = 0; idx < num; idx++) {
                std::cout << "scored_states[" << idx << "] = " << scored_states[idx].score << std::endl;
            }
        }

        std::sort(beam_states.begin(), beam_states.end(), state_compare);


    }

};

#endif /* SRC_BeamGraph_H_ */