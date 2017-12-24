#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include "N3LDG.h"
#include "Metric.h"
#include "NewMetric.h"
#include "Result.h"

class Instance {
  public:
    Instance() {
    }

    Instance(const Instance &other) {
        copyValuesFrom(other);
    }

    ~Instance() {
    }

  public:

    int size() const {
        return words.size();
    }


    void clear() {
        words.clear();
        tags.clear();
        heads.clear();
        labels.clear();
        result.clear();
        clearVec(chars);
        clearVec(syn_feats);
    }

    void allocate(const int &size) {
        if (words.size() != size) {
            words.resize(size);
        }
        if (tags.size() != size) {
            tags.resize(size);
        }
        if (heads.size() != size) {
            heads.resize(size);
        }
        if (labels.size() != size) {
            labels.resize(size);
        }

        syn_feats.resize(size);
        for (int idx = 0; idx < size; idx++) {
            syn_feats[idx].clear();
        }

        result.allocate(size);
    }


    void copyValuesFrom(const Instance &anInstance) {
        allocate(anInstance.size());
        for (int i = 0; i < anInstance.size(); i++) {
            words[i] = anInstance.words[i];
        }

        for (int i = 0; i < anInstance.size(); i++) {
            tags[i] = anInstance.tags[i];
        }

        for (int i = 0; i < anInstance.size(); i++) {
            heads[i] = anInstance.heads[i];
        }

        for (int i = 0; i < anInstance.size(); i++) {
            labels[i] = anInstance.labels[i];
        }

        result.copyValuesFrom(anInstance.result, &words, &tags, &heads, &labels);

        getChars();

        for (int i = 0; i < anInstance.size(); i++) {
            syn_feats[i].clear();
            for (int j = 0; j < anInstance.syn_feats[i].size(); j++) {
                syn_feats[i].push_back(anInstance.syn_feats[i][j]);
            }
        }

    }


    void evaluate(CResult &other, Metric &nerEval, Metric &relEval) {
        unordered_set<string>::iterator iter;

        unordered_set<string> gold_entities, pred_entities;
        result.extractNERs(gold_entities);
        other.extractNERs(pred_entities);

        nerEval.overall_label_count += gold_entities.size();
        nerEval.predicated_label_count += pred_entities.size();
        for (iter = pred_entities.begin(); iter != pred_entities.end(); iter++) {
            if (gold_entities.find(*iter) != gold_entities.end()) {
                nerEval.correct_label_count++;
            }
        }

        unordered_set<string> gold_entity_relations, pred_entity_relations;
        result.extractRelations(gold_entity_relations);
        other.extractRelations(pred_entity_relations);

        relEval.overall_label_count += gold_entity_relations.size();
        relEval.predicated_label_count += pred_entity_relations.size();
        for (iter = pred_entity_relations.begin(); iter != pred_entity_relations.end(); iter++) {
            if (gold_entity_relations.find(*iter) != gold_entity_relations.end()) {
                relEval.correct_label_count++;
            }
        }

    }


    void evaluateBinyNER(CResult &other, unordered_map<string, Metric>& nersEval) {

        unordered_map<string, unordered_set<string> > gold_entities, pred_entities;
        result.extractNERs(gold_entities);
        other.extractNERs(pred_entities);

        unordered_map<string, unordered_set<string> >::iterator map_iter;
        unordered_set<string>::iterator iter1;
        unordered_set<string>::iterator iter2;

        for (map_iter = gold_entities.begin(); map_iter != gold_entities.end(); map_iter++) {
            if (nersEval.find(map_iter->first) == nersEval.end()) {
                nersEval[map_iter->first] = Metric();
            }
            nersEval[map_iter->first].overall_label_count += map_iter->second.size();
        }

        for (map_iter = pred_entities.begin(); map_iter != pred_entities.end(); map_iter++) {
            if (nersEval.find(map_iter->first) == nersEval.end()) {
                nersEval[map_iter->first] = Metric();
            }

            nersEval[map_iter->first].predicated_label_count += map_iter->second.size();

            if (gold_entities.find(map_iter->first) != gold_entities.end()) {
                for (iter1 = map_iter->second.begin(); iter1 != map_iter->second.end(); iter1++) {
                    Span sp1(*iter1);
                    for (iter2 = gold_entities[map_iter->first].begin(); iter2 != gold_entities[map_iter->first].end(); iter2++) {
                        Span sp2(*iter2);
                        if (sp1.bBinary(sp2)) {
                            nersEval[map_iter->first].correct_label_count++;
                        }
                    }
                }
            }
        }

    }

    void evaluatePropNER(CResult &other, unordered_map<string, NewMetric>& nersEval) {
        unordered_map<string, unordered_set<string> > gold_entities, pred_entities;
        result.extractNERs(gold_entities);
        other.extractNERs(pred_entities);

        unordered_map<string, unordered_set<string> >::iterator map_iter;
        unordered_set<string>::iterator iter1;
        unordered_set<string>::iterator iter2;

        for (map_iter = gold_entities.begin(); map_iter != gold_entities.end(); map_iter++) {
            if (nersEval.find(map_iter->first) == nersEval.end()) {
                nersEval[map_iter->first] = NewMetric();
            }
            nersEval[map_iter->first].overall_label_count += map_iter->second.size();
        }

        for (map_iter = pred_entities.begin(); map_iter != pred_entities.end(); map_iter++) {
            if (nersEval.find(map_iter->first) == nersEval.end()) {
                nersEval[map_iter->first] = NewMetric();
            }

            nersEval[map_iter->first].predicated_label_count += map_iter->second.size();

            if (gold_entities.find(map_iter->first) != gold_entities.end()) {
                for (iter1 = map_iter->second.begin(); iter1 != map_iter->second.end(); iter1++) {
                    Span sp1(*iter1);
                    for (iter2 = gold_entities[map_iter->first].begin(); iter2 != gold_entities[map_iter->first].end(); iter2++) {
                        Span sp2(*iter2);
                        nersEval[map_iter->first].correct_label_count += sp1.matchProp(sp2);
                    }
                }
            }
        }
    }


    void evaluateBinyREL(CResult &other, unordered_map<string, Metric>& relsEval) {
        unordered_set<string>::iterator iter;

        unordered_map<string, unordered_set<string> > gold_entities, pred_entities;
        result.extractRelations(gold_entities);
        other.extractRelations(pred_entities);

        unordered_map<string, unordered_set<string> >::iterator map_iter;
        unordered_set<string>::iterator iter1;
        unordered_set<string>::iterator iter2;

        for (map_iter = gold_entities.begin(); map_iter != gold_entities.end(); map_iter++) {
            if (relsEval.find(map_iter->first) == relsEval.end()) {
                relsEval[map_iter->first] = Metric();
            }
            relsEval[map_iter->first].overall_label_count += map_iter->second.size();
        }

        for (map_iter = pred_entities.begin(); map_iter != pred_entities.end(); map_iter++) {
            if (relsEval.find(map_iter->first) == relsEval.end()) {
                relsEval[map_iter->first] = Metric();
            }

            relsEval[map_iter->first].predicated_label_count += map_iter->second.size();

            if (gold_entities.find(map_iter->first) != gold_entities.end()) {
                for (iter1 = map_iter->second.begin(); iter1 != map_iter->second.end(); iter1++) {
                    SpanPair sp1(*iter1);
                    for (iter2 = gold_entities[map_iter->first].begin(); iter2 != gold_entities[map_iter->first].end(); iter2++) {
                        SpanPair sp2(*iter2);
                        if (sp1.bBinary(sp2)) {
                            relsEval[map_iter->first].correct_label_count++;
                        }
                    }
                }
            }
        }
    }

    //reward computation
    void evaluatePropNER(CResult &other, NewMetric& nersEval) {
        unordered_map<string, unordered_set<string> > gold_entities, pred_entities;
        result.extractNERs(gold_entities);
        other.extractNERs(pred_entities);

        unordered_map<string, unordered_set<string> >::iterator map_iter;
        unordered_set<string>::iterator iter1;
        unordered_set<string>::iterator iter2;

        for (map_iter = gold_entities.begin(); map_iter != gold_entities.end(); map_iter++) {
            nersEval.overall_label_count += map_iter->second.size();
        }

        for (map_iter = pred_entities.begin(); map_iter != pred_entities.end(); map_iter++) {
            nersEval.predicated_label_count += map_iter->second.size();


            if (gold_entities.find(map_iter->first) != gold_entities.end()) {
                for (iter1 = map_iter->second.begin(); iter1 != map_iter->second.end(); iter1++) {
                    Span sp1(*iter1);
                    dtype max_score = 0;
                    for (iter2 = gold_entities[map_iter->first].begin(); iter2 != gold_entities[map_iter->first].end(); iter2++) {
                        Span sp2(*iter2);
                        dtype cur_score = sp1.matchProp(sp2);
                        if (cur_score > max_score) max_score = cur_score;
                    }
                    if (max_score > 1 + 1e-8) {
                        std::cout << "error prop evaluation" << std::endl;
                    }
                    nersEval.correct_label_count += max_score;
                }
            }


        }
    }

    //reward computation
    void evaluatePropREL(CResult &other, NewMetric& relsEval) {
        unordered_set<string>::iterator iter;

        unordered_map<string, unordered_set<string> > gold_entities, pred_entities;
        result.extractRelations(gold_entities);
        other.extractRelations(pred_entities);

        unordered_map<string, unordered_set<string> >::iterator map_iter;
        unordered_set<string>::iterator iter1;
        unordered_set<string>::iterator iter2;

        for (map_iter = gold_entities.begin(); map_iter != gold_entities.end(); map_iter++) {
            relsEval.overall_label_count += map_iter->second.size();
        }

        for (map_iter = pred_entities.begin(); map_iter != pred_entities.end(); map_iter++) {
            relsEval.predicated_label_count += map_iter->second.size();

            if (gold_entities.find(map_iter->first) != gold_entities.end()) {
                for (iter1 = map_iter->second.begin(); iter1 != map_iter->second.end(); iter1++) {
                    SpanPair sp1(*iter1);
                    dtype max_score = 0;
                    for (iter2 = gold_entities[map_iter->first].begin(); iter2 != gold_entities[map_iter->first].end(); iter2++) {
                        SpanPair sp2(*iter2);
                        dtype cur_score = sp1.matchProp(sp2);
                        if (cur_score > max_score) max_score = cur_score;
                    }
                    relsEval.correct_label_count += max_score;
                }
            }


        }
    }


  public:
    vector<string> words;
    vector<vector<string> > chars;
    vector<string> tags;
    vector<string> labels;
    vector<int> heads;
    CResult result;

    //extenral features
    vector<vector<PNode> > syn_feats;

  public:
    inline void getChars() {
        int size = words.size();
        chars.resize(size);
        for (int idx = 0; idx < size; idx++) {
            getCharactersFromUTF8String(words[idx], chars[idx]);
        }

    }

};

#endif
