#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "BeamGraph.h"
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
        _beam_builders.clear();
        _greedy_builders.clear();
    }

  public:
    Graph _cg;  // build neural graphs
    vector<Graph> _decode_cgs;
    vector<BeamGraphBuilder> _beam_builders;
    vector<GreedyGraphBuilder> _greedy_builders;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    ModelUpdate _ada;  // model update

    int _batch;
    bool _useBeam;
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

        _beam_builders.resize(_hyperparams.batch);
        _greedy_builders.resize(_hyperparams.batch);
        _decode_cgs.resize(_hyperparams.batch);
        dtype beam_drop = -1.0;
        dtype greedy_drop = _hyperparams.dropProb;
        for (int idx = 0; idx < _hyperparams.batch; idx++) {
            _hyperparams.dropProb = beam_drop;
            _beam_builders[idx].initial(_modelparams, _hyperparams);
            _hyperparams.dropProb = greedy_drop;
            _greedy_builders[idx].initial(_modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
        _batch = 0;
        _useBeam = false;
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
        if(_useBeam) {
            if (num > _beam_builders.size()) {
                std::cout << "input example number is larger than predefined batch number" << std::endl;
                return -1;
            }

            _cg.clearValue(true);
            for (int idx = 0; idx < num; idx++) {
                _beam_builders[idx].encode(&_cg, sentences[idx]);
            }
            _cg.compute();

            // #pragma omp parallel for schedule(static,1)
            for (int idx = 0; idx < num; idx++) {
                _decode_cgs[idx].clearValue(true);
                _beam_builders[idx].decode(&(_decode_cgs[idx]), (sentences[idx]), nerOnly, &(goldACs[idx]));

                int upper_step = nerOnly ? sentences[idx].size() : _beam_builders[idx].outputs.size();
                //compute reward
                reward_computation(_beam_builders[idx], (sentences[idx]), upper_step - 1, nerOnly);
                //reward_computation_onlygold(_beam_builders[idx], (sentences[idx]), upper_step - 1, nerOnly);
                cost += loss_google(_beam_builders[idx], upper_step, num);

                _decode_cgs[idx].backward();
            }

            _cg.backward();
        } else {
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
                cost += loss_google(_greedy_builders[idx], upper_step, num);

                _decode_cgs[idx].backward();
            }

            _cg.backward();
        }

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

    dtype loss_google(BeamGraphBuilder& builder, int upper_step, int num) {
        int maxstep = builder.outputs.size();
        if (maxstep == 0) return 1.0;
        if (upper_step > 0 && maxstep > upper_step) maxstep = upper_step;
        PNode pBestNode = NULL;
        PNode pGoldNode = NULL;
        PNode pCurNode;
        dtype sum, max, norm;
        int curcount, goldIndex;
        vector<dtype> scores;
        vector<dtype> probs;
        vector<dtype> wrewards;
        dtype cost = 0.0;
        dtype eps = 1e-8;

        for (int step = maxstep - 1; step < maxstep; step++) {
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

            max = pBestNode->val[0] * _anneal;
            sum = 0.0;
            scores.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] * _anneal - max);
                if (scores[idx] < eps) scores[idx] = eps;
                sum += scores[idx];
            }

            probs.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                probs[idx] = scores[idx] / sum;
            }

            norm = 0.0;
            wrewards.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                if (builder.outputs[step][idx].reward < -1e-20) {
                    std::cout << "reward must be larger than zero" << std::endl;
                    builder.outputs[step][idx].reward = 0;
                }
                wrewards[idx] = builder.outputs[step][idx].reward * probs[idx];
                norm += wrewards[idx];
            }

            if (std::isinf(norm) || std::isnan(norm) || norm < 1e-20) {
                std::cout << "strange norm, please check" << std::endl;
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = builder.outputs[step][idx].in;
                pCurNode->loss[0] += (_anneal / num) * (probs[idx] - wrewards[idx] / norm);
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


    void reward_computation(BeamGraphBuilder& builder, Instance& inst, int last_step, bool nerOnly) {
        int curcount = builder.outputs[last_step].size();
        for (int idx = 0; idx < curcount; idx++) {
            CStateItem* pGenerator = builder.outputs[last_step][idx].curState;
            CAction nextac = builder.outputs[last_step][idx].ac;
            builder.outputs[last_step][idx].reward = pGenerator->rewardByAction(inst, nextac, _hyperparams, nerOnly);
        }
    }

    void reward_computation_onlygold(BeamGraphBuilder& builder, Instance& inst, int last_step, bool nerOnly) {
        int curcount = builder.outputs[last_step].size();
        for (int idx = 0; idx < curcount; idx++) {
            bool bGold = builder.outputs[last_step][idx].bGold;
            builder.outputs[last_step][idx].reward = bGold ? 1 : 0;
        }
    }

    dtype loss_google_old(BeamGraphBuilder& builder, int upper_step, int num) {
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

        for (int step = maxstep - 1; step < maxstep; step++) {
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

  public:
    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

    //useBeam = true, beam searcher
    inline void setGraph(bool useBeam) {
        _useBeam = useBeam;
    }

    inline void setClip(dtype clip) {
        _clip = clip;
    }

};

#endif /* SRC_Driver_H_ */
