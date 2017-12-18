#ifndef STATE_H_
#define STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"
#include "Instance.h"
#include "GlobalNodes.h"

class CStateItem {
  public:
    short _label;
    short _current_i; //
    short _current_j; //
    short _step;
    short _labels[max_token_size][max_token_size];
    short _entities[max_token_size + 1];
    short _headed_entities[max_token_size + 1];
    short _heads[max_token_size + 1];
    short _entity_size;
    short _head_size;
    short _current_entity_index;
    short _current_head_index;

    CStateItem *_prevState;
    Instance *_inst;
    int _word_size;

    CAction _lastAction;
    PNode _score;

    // features
    ActionedNodes _nextscores;  // features current used
    AtomFeatures _atomFeat;  //features will be used for future


  public:
    bool _bStart; // whether it is a start state
    bool _bGold; // for train
    bool _bEnd; // whether it is an end state


  public:
    CStateItem() {
        clear();
    }


    virtual ~CStateItem() {
        clear();
    }

    void initial(ModelParams &params, HyperParams &hyparams) {
        _nextscores.initial(params, hyparams);
    }

    void setInput(Instance& inst) {
        _inst = &inst;
        _word_size = _inst->size();
    }

    void clear() {
        _current_i = -1;
        _current_j = -1;
        _step = 0;
        _label = invalid_label;

        for (int idx = 0; idx < max_token_size; idx++) {
            for (int idy = 0; idy < max_token_size; idy++) {
                _labels[idx][idy] = 0;
            }
            _entities[idx] = -1;
            _heads[idx] = -1;
            _headed_entities[idx] = -1;
        }
        _entities[max_token_size] = -1;
        _heads[max_token_size] = -1;
        _headed_entities[max_token_size] = -1;

        _entity_size = 0;
        _head_size = 0;
        _current_entity_index = -1;
        _current_head_index = -1;

        _prevState = 0;
        _lastAction.clear();

        _inst = 0;
        _word_size = 0;

        _score = NULL;
        _bStart = true;
        _bGold = true;
        _bEnd = false;
    }



  protected:
    inline void copyProperty2Next(CStateItem *next) {
        next->_current_i = _current_i;
        next->_current_j = _current_j;

        for (int idx = 0; idx < max_token_size; idx++) {
            memcpy(next->_labels[idx], _labels[idx], sizeof(short) * (max_token_size));
        }

        next->_entity_size = _entity_size;
        for (int idx = 0; idx < _entity_size; idx++) {
            next->_entities[idx] = _entities[idx];
            next->_headed_entities[idx] = _headed_entities[idx];
        }
        next->_entities[_entity_size] = -1;
        next->_headed_entities[_entity_size] = -1;

        next->_head_size = _head_size;
        for (int idx = 0; idx < _head_size; idx++) {
            next->_heads[idx] = _heads[idx];
        }
        next->_heads[_head_size] = -1;

        next->_current_entity_index = _current_entity_index;
        next->_current_head_index = _current_head_index;

        // do not need modification any more
        next->_inst = _inst;
        next->_word_size = _word_size;
        next->_step = _step + 1;
        next->_prevState = this;
    }

    // conditions
  public:
    bool allow_ner() const {
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;

        if (next_i == next_j && next_j < _word_size && next_j >= 0) {
            return true;
        }
        return false;
    }

    bool allow_rel() const {
        /*
        if (_current_j == _current_i) return false;
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;

        if (next_j == _word_size) {
        	next_i = 0;
            next_j = next_i + _current_j - _current_i + 1;
        }

        if (next_j > next_i && next_j < _word_size && next_i >= 0) {
            return true;
        }
        return false;
        */
        if (_current_entity_index >= 0) return true;
        return false;
    }


  public:
    // please check the specified index has been annotated with ner label first
    short getNERId(const int& i) const {
        if (i >= _word_size || i < 0) {
            return -1;
        }

        return _labels[i][i];
    }

    // please check the specified index has been annotated with ner label first
    short getSpanStart(const int& i) const {
        if (i >= _word_size || i < 0) {
            return -1;
        }

        if (i == 0 || _labels[i][i] % 4 == 0) {
            return i;
        }

        int j = i - 1;
        while (j > 0) {
            if (_labels[j][j] % 4 == 1) {
                break;
            }
            j--;
        }

        return j;
    }

    //actions
  public:
    void ner(CStateItem *next, short ner_id) {
        if (!allow_ner()) {
            std::cout << "assign ner error" << std::endl;
            return;
        }

        copyProperty2Next(next);
        next->_label = ner_id;
        next->_current_j = _current_j + 1;
        next->_current_i = _current_i + 1;

        if (ner_id == 3 || ner_id == 4 || ner_id == 7 || ner_id == 8) {
            next->_entities[_entity_size] = next->_current_j;
            next->_headed_entities[_entity_size] = 0;
            next->_entities[_entity_size + 1] = -1;
            next->_headed_entities[_entity_size + 1] = -1;
            next->_entity_size = _entity_size + 1;
        } else if (ner_id == 11 || ner_id == 12) {
            next->_heads[_head_size] = next->_current_j;
            next->_heads[_head_size + 1] = -1;
            next->_head_size = _head_size + 1;
        }

        //next step starts opinion relation
        if (next->_current_j == _word_size - 1) {
            if (next->_entity_size > 0 && next->_head_size > 0) {
                next->_current_entity_index = 0;
            } else {
                next->_bEnd = true;
            }
        }

        next->_labels[next->_current_i][next->_current_j] = ner_id;


        next->_lastAction.set(CAction::NER, ner_id); //TODO:
    }

    void rel(CStateItem *next, short rel_id) {
        if (!allow_rel()) {
            std::cout << "assign relation error" << std::endl;
            return;
        }

        copyProperty2Next(next);

        if (_current_head_index == _head_size - 1) {
            next->_current_entity_index = _current_entity_index + 1;
            next->_current_head_index = 0;
        } else {
            next->_current_head_index = _current_head_index + 1;
        }

        short temp_i = _entities[next->_current_entity_index];
        short temp_j = _heads[next->_current_head_index];

        next->_label = rel_id;
        next->_current_j = temp_j > temp_i ? temp_j : temp_i;
        next->_current_i = temp_j > temp_i ? temp_i : temp_j;

        if (rel_id != 0) {
            next->_headed_entities[next->_current_entity_index] = _headed_entities[next->_current_entity_index] + 1;
        }

        if (next->_current_entity_index == _entity_size - 1 && next->_current_head_index == _head_size - 1) {
            next->_bEnd = true;
            next->_current_entity_index = -1;
            next->_current_head_index = -1;
        }

        next->_labels[next->_current_i][next->_current_j] = rel_id;

        next->_lastAction.set(CAction::REL, rel_id); //TODO:
    }

    void idle(CStateItem *next) {
        if (!_bEnd) {
            std::cout << "is not terminated, error" << std::endl;
            return;
        }
        copyProperty2Next(next);

        next->_bEnd = true;
        next->_lastAction.setNoAction(); //TODO:
    }

    //move, orcale
  public:
    void move(CStateItem *next, const CAction &ac) {
        next->_bStart = false;
        next->_bEnd = false;
        next->_bGold = false;
        if (ac.isNER()) {
            ner(next, ac._label);
        } else if (ac.isREL()) {
            rel(next, ac._label); //TODO:
        } else if (_bEnd && ac.isNone()) {
            idle(next); //TODO:
        } else {
            std::cout << "error action" << std::endl;
        }

    }

    //partial results
    void getResults(CResult &result, HyperParams &opts) const {
        result.clear();
        const CStateItem* curr = this;
        result.words = &(_inst->words);
        result.tags = &(_inst->tags);
        result.heads = &(_inst->heads);
        result.labels = &(_inst->labels);
        result.allocate(_word_size);

        for (int idx = 0; idx < _word_size; idx++) {
            result.relations[idx][idx] = opts.ner_labels.from_id(_labels[idx][idx]);
            for (int idy = idx + 1; idy < _word_size; idy++) {
                short labelId = _labels[idx][idy];
                if (labelId > 0) {
                    result.relations[idx][idy] = opts.rel_labels.from_id(labelId);
                    result.directions[idx][idy] = 1;
                } else if (labelId < 0) {
                    result.relations[idx][idy] = opts.rel_labels.from_id(-labelId);
                    result.directions[idx][idy] = 2;
                } else {
                    result.relations[idx][idy] = opts.rel_labels.from_id(0);
                    result.directions[idx][idy] = 0;
                }
            }
        }
    }

    // TODO:
    void getGoldAction(HyperParams &opts, const CResult &result, CAction &ac) const {
        if (allow_ner()) {
            short next_j = _current_j + 1;
            ac.set(CAction::NER, opts.ner_labels.from_string(result.relations[next_j][next_j]));
            return;
        }

        if (allow_rel()) {
            short next_i, next_j;
            short next_entity_index, next_head_index;
            next_entity_index = _current_entity_index;
            next_head_index = _current_head_index;
            if (_current_head_index == _head_size - 1) {
                next_entity_index = _current_entity_index + 1;
                next_head_index = 0;
            } else {
                next_head_index = _current_head_index + 1;
            }

            short temp_i = _entities[next_entity_index];
            short temp_j = _heads[next_head_index];

            next_j = temp_j > temp_i ? temp_j : temp_i;
            next_i = temp_j > temp_i ? temp_i : temp_j;

            int rel_labelId = opts.rel_labels.from_string(result.relations[next_i][next_j]);
            if (rel_labelId == 0 || result.directions[next_i][next_j] == 1) {
            } else if (result.directions[next_i][next_j] == 2) {
                rel_labelId = -rel_labelId;
            } else {
                std::cout << "get gold relation error" << std::endl;
            }
            ac.set(CAction::REL, rel_labelId);
            return;
        }

        ac.setNoAction();
        return;
    }
    //
    //	// we did not judge whether history actions are match with current state.
    void getGoldAction(const CStateItem* goldState, CAction& ac) const {
        if (_step > goldState->_step || _step < 0) {
            ac.set(CAction::NO_ACTION, -1);
            return;
        }
        const CStateItem *prevState = goldState->_prevState;
        CAction curAction = goldState->_lastAction;
        while (_step < prevState->_step) {
            curAction = prevState->_lastAction;
            prevState = prevState->_prevState;
        }
        return ac.set(curAction._code, curAction._label);
    }

    void getCandidateActions(vector<CAction> &actions, HyperParams* opts, ModelParams* params, bool bTrain) const {
        actions.clear();
        CAction ac;

        if (_bEnd) {
            //std::cout << "terminated, error" << std::endl;
            ac.setNoAction();
            actions.push_back(ac);
            return;
        }

        if (allow_ner()) {
            int modvalue = _lastAction._label % 4;
            if (modvalue == 0 || modvalue == 3) { //o, e-xx, s-xx
                ac.set(CAction::NER, 0);
                if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                for (int i = 0; i < opts->ner_noprefix_num; i++) {
                    ac.set(CAction::NER, 4 * i + 1);  //b-xx
                    if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                    ac.set(CAction::NER, 4 * i + 4);  //s-xx
                    if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                }
            } else if (modvalue == 1) { //b-xx
                ac.set(CAction::NER, _lastAction._label + 1);  //m-xx
                if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                ac.set(CAction::NER, _lastAction._label + 2);  //e-xx
                if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
            } else { // m-xx
                ac.set(CAction::NER, _lastAction._label);  //m-xx
                if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                ac.set(CAction::NER, _lastAction._label + 1);  //e-xx
                if (params->embeded_ner_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
            }

            //gold
            //ac.set(CAction::NER, opts->ner_labels.from_string(_inst->result.ners[_current_i]));
            //actions.push_back(ac);
            return;
        }

        if (allow_rel()) {
            short next_i, next_j;
            short next_entity_index, next_head_index;
            next_entity_index = _current_entity_index;
            next_head_index = _current_head_index;
            if (_current_head_index == _head_size - 1) {
                next_entity_index = _current_entity_index + 1;
                next_head_index = 0;
            } else {
                next_head_index = _current_head_index + 1;
            }

            short temp_i = _entities[next_entity_index];
            short temp_j = _heads[next_head_index];

            next_j = temp_j > temp_i ? temp_j : temp_i;
            next_i = temp_j > temp_i ? temp_i : temp_j;

            short label_i = getNERId(next_i);
            short label_j = getNERId(next_j);

            int modvalue_i = label_i % 4;
            int modvalue_j = label_j % 4;


            if (label_i > 0 && label_j > 0
                    && (modvalue_i == 0 || modvalue_i == 3)
                    && (modvalue_j == 0 || modvalue_j == 3)) {
                string ner_label_i = opts->ner_labels.from_id(label_i);
                ner_label_i = cleanLabel(ner_label_i);

                string ner_label_j = opts->ner_labels.from_id(label_j);
                ner_label_j = cleanLabel(ner_label_j);


                string rel = ner_label_i + "-" + ner_label_j;
                int rel_id = opts->rel_labels.from_string(rel);
                if (rel_id != -1) {
                    ac.set(CAction::REL, rel_id);
                    actions.push_back(ac);
                }

                rel = ner_label_j + "-" + ner_label_i;
                rel_id = opts->rel_labels.from_string(rel);
                if (rel_id != -1) {
                    ac.set(CAction::REL, -rel_id);
                    actions.push_back(ac);
                }

                if (next_head_index == _head_size - 1 && _headed_entities[next_entity_index] == 0) {
                    //std::cout << "last dse" << std::endl;
                    //be sure that each agent/target has one dse
                } else {
                    ac.set(CAction::REL, 0);
                    actions.push_back(ac);
                }

                /*
                for (int idx = 0; idx < opts->rel_labels.size(); idx++) {
                  if (opts->rel_dir[opts->rel_labels.from_id(idx)].find(1) != opts->rel_dir[opts->rel_labels.from_id(idx)].end()) {
                    ac.set(CAction::REL, idx);
                    actions.push_back(ac);
                  }
                  if (opts->rel_dir[opts->rel_labels.from_id(idx)].find(-1) != opts->rel_dir[opts->rel_labels.from_id(idx)].end()) {
                    ac.set(CAction::REL, -idx);
                    actions.push_back(ac);
                  }
                }*/

            } else {
                if (actions.size() != 1) {
                    std::cout << "impossible conditions: getCandidateActions" << std::endl;
                }
                ac.set(CAction::REL, 0);
                actions.push_back(ac);
            }


        }

    }

    //TODO: debug
    inline std::string str(HyperParams* opts) const {
        stringstream curoutstr;
        curoutstr << "score: " << _score->val[0] << " ";

        curoutstr << "actions:";
        vector<string> allacs;

        const CStateItem * curState;
        curState = this;
        while (!curState->_bStart) {
            allacs.insert(allacs.begin(), curState->_lastAction.str(opts));
            curState = curState->_prevState;
        }
        for (int idx = 0; idx < allacs.size(); idx++) {
            curoutstr << " " << allacs[idx];
        }
        return curoutstr.str();
    }


  public:

    inline void computeNextScore(Graph *cg, const vector<CAction>& acs, bool useBeam) {
        if (_bStart || !useBeam) {
            _nextscores.forward(cg, acs, _atomFeat, NULL);
        } else {
            _nextscores.forward(cg, acs, _atomFeat, _score);
        }
    }

    inline void prepare(HyperParams* hyper_params, ModelParams* model_params, GlobalNodes* global_nodes) {
        _atomFeat.clear();
        if (_bEnd) {
            _atomFeat.bEnd = true;
            return;
        }
        _atomFeat.bEnd = false;
        if (allow_ner()) {
            short next_j = _current_j + 1;
            _atomFeat.ner_next_position = next_j;
            _atomFeat.ner_last_end = _current_j;
            _atomFeat.ner_last_start = getSpanStart(_atomFeat.ner_last_end);
            short label_j = getNERId(_current_j);
            _atomFeat.ner_last_label = (label_j >= 0) ? hyper_params->ner_labels.from_id(label_j) : nullkey;
            _atomFeat.bRel = false;
        } else if(allow_rel()) {
            short next_i, next_j;
            short next_entity_index, next_head_index;
            next_entity_index = _current_entity_index;
            next_head_index = _current_head_index;
            if (_current_head_index == _head_size - 1) {
                next_entity_index = _current_entity_index + 1;
                next_head_index = 0;
            } else {
                next_head_index = _current_head_index + 1;
            }

            short temp_i = _entities[next_entity_index];
            short temp_j = _heads[next_head_index];

            next_j = temp_j > temp_i ? temp_j : temp_i;
            next_i = temp_j > temp_i ? temp_i : temp_j;

            _atomFeat.rel_i = next_i;
            _atomFeat.rel_j = next_j;

            _atomFeat.rel_i_start = getSpanStart(_atomFeat.rel_i);
            _atomFeat.rel_j_start = getSpanStart(_atomFeat.rel_j);

            short label_i = getNERId(_atomFeat.rel_i);
            short label_j = getNERId(_atomFeat.rel_j);

            _atomFeat.rel_j_nerlabel = hyper_params->ner_labels.from_id(label_j);

            int modvalue_i = label_i % 4;
            int modvalue_j = label_j % 4;

            if (label_i > 0 && label_j > 0
                    && (modvalue_i == 0 || modvalue_i == 3)
                    && (modvalue_j == 0 || modvalue_j == 3)) {
                _atomFeat.rel_must_o = 0;
            } else {
                std::cout << "impossible conditions: prepare" << std::endl;
                _atomFeat.rel_must_o = 1;
            }
            _atomFeat.bRel = true;
        } else {
            std::cout << "error for next step!" << std::endl;
        }

        _atomFeat.word_size = _word_size;
        if (global_nodes == NULL) {
            _atomFeat.p_word_left_lstm = _atomFeat.p_word_right_lstm = NULL;
        } else {
            int layer_num = global_nodes->additional_layer_num;
            _atomFeat.p_word_left_lstm = (layer_num > 0) ? &(global_nodes->word_left_lstm_deeper[layer_num - 1]) : &(global_nodes->word_left_lstm);
            _atomFeat.p_word_right_lstm = (layer_num > 0) ? &(global_nodes->word_right_lstm_deeper[layer_num - 1]) : &(global_nodes->word_right_lstm);
        }
    }

    //only support last step before finished
    inline dtype rewardByAction(Instance& inst, const CAction& ac, HyperParams &opts, bool nerOnly) {
        CResult output;
        if (ac.isNER()) {
            short next_j = _current_j + 1;
            short temp = _labels[next_j][next_j];
            _labels[next_j][next_j] = ac._label;
            getResults(output, opts);
            _labels[next_j][next_j] = temp;
            int new_head_size = _head_size;
            if (ac._label == 11 || ac._label == 12) new_head_size++;
            if (_entity_size > 0 && new_head_size == 0) {
                return 0;
            }
        } else if (ac.isREL()) {
            short next_i, next_j;
            short next_entity_index, next_head_index;
            next_entity_index = _current_entity_index;
            next_head_index = _current_head_index;
            if (_current_head_index == _head_size - 1) {
                next_entity_index = _current_entity_index + 1;
                next_head_index = 0;
            } else {
                next_head_index = _current_head_index + 1;
            }

            short temp_i = _entities[next_entity_index];
            short temp_j = _heads[next_head_index];

            next_j = temp_j > temp_i ? temp_j : temp_i;
            next_i = temp_j > temp_i ? temp_i : temp_j;

            short temp = _labels[next_i][next_j];
            _labels[next_i][next_j] = ac._label;
            getResults(output, opts);
            _labels[next_i][next_j] = temp;

        } else {
            if (!_bEnd) {
                std::cout << "error to compute the reward value" << std::endl;
            }
            if (_entity_size > 0 && _head_size == 0) {
                return 0;
            }
            getResults(output, opts);
        }

        Metric ner, rel;
        inst.evaluate(output, ner, rel);

        dtype reward = nerOnly ? ner.getAccuracy() : ner.getAccuracy() * rel.getAccuracy();

        return reward;
    }

};

class CScoredState {
  public:
    CStateItem *item;
    CAction ac;
    dtype score;
    bool bGold;
    int position;

  public:
    CScoredState() : item(0), score(0), ac(0, -1), bGold(0), position(-1) {
    }

    CScoredState(const CScoredState &other) : item(other.item), score(other.score), ac(other.ac), bGold(other.bGold),
        position(other.position) {

    }

  public:
    bool operator<(const CScoredState &a1) const {
        return score < a1.score;
    }

    bool operator>(const CScoredState &a1) const {
        return score > a1.score;
    }

    bool operator<=(const CScoredState &a1) const {
        return score <= a1.score;
    }

    bool operator>=(const CScoredState &a1) const {
        return score >= a1.score;
    }
};

class CScoredState_Compare {
  public:
    int operator()(const CScoredState &o1, const CScoredState &o2) const {
        if (o1.score < o2.score)
            return -1;
        else if (o1.score > o2.score)
            return 1;
        else
            return 0;
    }
};


struct COutput {
    PNode in;
    CStateItem *curState;
    CAction ac;
    dtype reward;
    bool bGold;

    COutput() : in(NULL), curState(NULL), ac(), reward(0), bGold(false) {
    }

    COutput(const COutput &other) : in(other.in), curState(other.curState), ac(other.ac), reward(other.reward), bGold(other.bGold) {
    }
};

#endif /* STATE_H_ */