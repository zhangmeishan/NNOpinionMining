#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "GreedyGraph.h"

class Driver {
  public:
    Driver() {
        _batch = 0;
        _clip = 10.0;
    }

    ~Driver() {
        _batch = 0;
        _clip = 10.0;
        _greedy_builders.clear();
    }

  public:
    Graph _cg;  // build neural graphs
    vector<Graph> _decode_cgs;
    vector<GreedyGraphBuilder> _greedy_builders;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    ModelUpdate _ada;  // model update

    int _batch;
    dtype _clip;
    dtype _anneal; //percentage of max-entropy

  public:
    inline void initial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
            return;
        }
        if (!_modelparams.initial(_hyperparams)) {
            std::cout << "model parameter initialization Error, Please check!" << std::endl;
            return;
        }
        _hyperparams.print();

        _greedy_builders.resize(_hyperparams.batch);
        _decode_cgs.resize(_hyperparams.batch);

        for (int idx = 0; idx < _hyperparams.batch; idx++) {
            _greedy_builders[idx].initial(_modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
        _batch = 0;
        _anneal = _hyperparams.anneal;
    }

    inline void setDropFactor(dtype drop_factor) {
        _cg.setDropFactor(drop_factor);
        for (int idx = 0; idx < _decode_cgs.size(); idx++) {
            _decode_cgs[idx].setDropFactor(drop_factor);
        }
    }


  public:
    dtype train(std::vector<Instance > &sentences, const vector<vector<CAction> > &goldACs, bool nerOnly) {
        _eval.reset();
        dtype cost = 0.0;
        int num = sentences.size();

        if (num > _greedy_builders.size()) {
            std::cout << "input example number is larger than predefined batch number" << std::endl;
            return -1;
        }

        _cg.clearValue(true);
        for (int idx = 0; idx < num; idx++) {
            _greedy_builders[idx].encode(&_cg, sentences[idx]);
        }
        _cg.compute();

        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < num; idx++) {
            _decode_cgs[idx].clearValue(true);
            _greedy_builders[idx].decode(&(_decode_cgs[idx]), (sentences[idx]), nerOnly, &(goldACs[idx]));

            int upper_step = nerOnly ? sentences[idx].size() : _greedy_builders[idx].outputs.size();
            //reward_computation_oracle(_greedy_builders[idx], (sentences[idx]), upper_step);
            cost += loss_google(_greedy_builders[idx], upper_step, num);

            _decode_cgs[idx].backward();
        }

        _cg.backward();


        return cost;
    }

    void decode(vector<Instance> &sentences, vector<CResult> &results) {
        int num = sentences.size();
        if (num > _greedy_builders.size()) {
            std::cout << "input example number is larger than predefined batch number" << std::endl;
            return;
        }
        _cg.clearValue();
        for (int idx = 0; idx < num; idx++) {
            _greedy_builders[idx].encode(&_cg, sentences[idx]);
        }
        _cg.compute();

        results.resize(num);
        //#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < num; idx++) {
            _decode_cgs[idx].clearValue();
            _greedy_builders[idx].decode(&(_decode_cgs[idx]), sentences[idx], false);
            int step = _greedy_builders[idx].outputs.size();
            _greedy_builders[idx].states[step - 1].getResults(results[idx], _hyperparams);
        }
    }

    void updateModel() {
        if (_batch <= 0) return;
        if (_ada._params.empty()) {
            _modelparams.exportModelParams(_ada);
        }
        //_ada.rescaleGrad(1.0 / _batch);
        //_ada.update(10);
        _ada.updateAdam(_clip);
        _batch = 0;
    }


    void writeModel();

    void loadModel();

  private:
    dtype loss_google_raml(GreedyGraphBuilder& builder, int upper_step, int num) {
        int maxstep = builder.outputs.size();
        if (maxstep == 0) return 1.0;
        if (upper_step > 0 && maxstep > upper_step) maxstep = upper_step;

        PNode pBestNode = NULL;
        PNode pGoldNode = NULL;
        PNode pCurNode;
        dtype sum, max, sumReward, maxReward;
        int curcount;
        int maxRewardIndex, goldIndex;
        vector<dtype> scores;
        vector<dtype> probs, gprobs;
        vector<dtype> scoreRewards;
        dtype cost = 0.0;
        dtype eps = 1e-8;

        for (int step = 0; step < maxstep; step++) {
            curcount = builder.outputs[step].size();
            _eval.overall_label_count++;
            if (curcount == 1) {
                _eval.correct_label_count++;
                continue;
            }

            goldIndex = -1;
            maxRewardIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (builder.outputs[step][idx].bGold) {
                    pGoldNode = pCurNode;
                    goldIndex = idx;
                }
                if (maxRewardIndex == -1 || builder.outputs[step][idx].reward > builder.outputs[step][maxRewardIndex].reward) {
                    maxRewardIndex = idx;
                }
            }

            if (goldIndex == -1 || maxRewardIndex != goldIndex) {
                std::cout << "impossible:  goldIndex = " << goldIndex << ", maxRewardIndex = " << maxRewardIndex << std::endl;
                std::cout << "reward:  gold = " << builder.outputs[step][goldIndex].reward << ", maxReward = " << builder.outputs[step][maxRewardIndex].reward << std::endl;
            }

            max = pBestNode->val[0];
            maxReward = builder.outputs[step][maxRewardIndex].reward;
            sum = 0.0;
            sumReward = 0.0;
            scores.resize(curcount);
            scoreRewards.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                if (scores[idx] < eps) scores[idx] = eps;
                sum += scores[idx];

                scoreRewards[idx] = exp((builder.outputs[step][idx].reward - maxReward) * _anneal);
                if (scoreRewards[idx] < eps) scoreRewards[idx] = eps;
                sumReward += scoreRewards[idx];
            }

            probs.resize(curcount);
            gprobs.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                probs[idx] = scores[idx] / sum;
                gprobs[idx] = scoreRewards[idx] / sumReward;
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                pCurNode->loss[0] += (probs[idx] - gprobs[idx]) / num;
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;

            cost += -log(probs[goldIndex]);

            if (std::isinf(cost) || std::isnan(cost)) {
                std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
            }

            _batch++;
        }

        return cost;
    }

    dtype loss_google(GreedyGraphBuilder& builder, int upper_step, int num) {
        int maxstep = builder.outputs.size();
        if (maxstep == 0) return 1.0;
        if (upper_step > 0 && maxstep > upper_step) maxstep = upper_step;
        PNode pBestNode = NULL;
        PNode pGoldNode = NULL;
        PNode pCurNode;
        dtype sum, max;
        int curcount, goldIndex;
        vector<dtype> scores;
        dtype cost = 0.0;

        for (int step = 0; step < maxstep; step++) {
            curcount = builder.outputs[step].size();
            _eval.overall_label_count++;
            if (curcount == 1) {
                _eval.correct_label_count++;
                continue;
            }
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (builder.outputs[step][idx].bGold) {
                    pGoldNode = pCurNode;
                    goldIndex = idx;
                }
            }

            if (goldIndex == -1) {
                std::cout << "impossible" << std::endl;
            }
            pGoldNode->loss[0] = -1.0 / num;

            max = pBestNode->val[0];
            sum = 0.0;
            scores.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                pCurNode->loss[0] += scores[idx] / (sum * num);
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;

            cost += -log(scores[goldIndex] / sum);

            if (std::isinf(cost) || std::isnan(cost)) {
                std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
            }

            _batch++;
        }

        return cost;
    }

    void reward_computation_oracle(GreedyGraphBuilder& builder, Instance& inst, int max_step) {
        vector<CStateItem> state(_hyperparams.maxlength + 1);
        int last_step = builder.outputs.size() - 1;
        if (last_step > max_step - 1) last_step = max_step - 1;
        int step = 0;
        CResult output;
        CAction nextac;
        CStateItem* pGenerator;
        NewMetric ner, rel;
        for (int idstep = 0; idstep <= last_step; idstep++) {
            int curcount = builder.outputs[idstep].size();
            for (int idx = 0; idx < curcount; idx++) {
                pGenerator = builder.outputs[idstep][idx].curState;
                nextac = builder.outputs[idstep][idx].ac;
                step = 0;
                pGenerator->move(&state[step], nextac);

                while (!state[step]._bEnd) {
                    state[step].getOrcaleAction(_hyperparams, inst.result, nextac);
                    state[step].move(&state[step + 1], nextac);
                    step++;
                }

                state[step].getResults(output, _hyperparams);

                ner.reset();
                rel.reset();
                inst.evaluatePropNER(output, ner);
                inst.evaluatePropREL(output, rel);

                builder.outputs[idstep][idx].reward = ner.getAccuracy() * rel.getAccuracy();

                if (builder.outputs[idstep][idx].bGold && abs(builder.outputs[idstep][idx].reward - 1.0) > 1e-5) {
                    std::cout << "check reward computation" << std::endl;

                    ner.print();
                    rel.print();
                    std::cout << idx << ", reward: " << builder.outputs[idstep][idx].reward << std::endl;

                    pGenerator = builder.outputs[idstep][idx].curState;
                    nextac = builder.outputs[idstep][idx].ac;
                    step = 0;
                    pGenerator->move(&state[step], nextac);

                    while (!state[step]._bEnd) {
                        state[step].getOrcaleAction(_hyperparams, inst.result, nextac);
                        state[step].move(&state[step + 1], nextac);
                        step++;
                    }

                    state[step].getResults(output, _hyperparams);

                    ner.reset();
                    rel.reset();
                    inst.evaluatePropNER(output, ner);
                    inst.evaluatePropREL(output, rel);
                }
            }
        }
    }

    void reward_computation_onlygold(GreedyGraphBuilder& builder, Instance& inst, int max_step) {
        vector<CStateItem> state(_hyperparams.maxlength + 1);
        int last_step = builder.outputs.size() - 1;
        if (last_step > max_step - 1) last_step = max_step - 1;

        for (int idstep = 0; idstep <= last_step; idstep++) {
            int curcount = builder.outputs[idstep].size();
            for (int idx = 0; idx < curcount; idx++) {
                bool bGold = builder.outputs[idstep][idx].bGold;
                builder.outputs[idstep][idx].reward = bGold ? 1 : 0;
            }
        }
    }



  public:
    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

    inline void setClip(dtype clip) {
        _clip = clip;
    }

};

#endif /* SRC_Driver_H_ */
