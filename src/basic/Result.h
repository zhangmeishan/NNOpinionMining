#ifndef BASIC_CRESULT_H_
#define BASIC_CRESULT_H_

#include <string>
#include <vector>
#include <fstream>
#include "N3LDG.h"
#include "Alphabet.h"
#include "Utf.h"

struct Span {
    int start;
    int end;
  public :
    Span() : start(-1), end(-1) {

    }

    Span(int x, int y) : start(x), end(y) {

    }

    Span(const string& strIn) {
        read(strIn);
    }

  public:
    inline std::string str() const {
        stringstream ss;
        ss << "(" << start << ", " << end << ")";
        return ss.str();
    }

    inline void read(const string& strIn) {
        int idx = strIn.find_first_of(",");
        int start1 = 1;
        int end1 = idx - 1;
        int start2 = idx + 2;
        int end2 = strIn.length() - 2;
        start = atoi(strIn.substr(start1, end1 - start1 + 1).c_str());
        end = atoi(strIn.substr(start2, end2 - start2 + 1).c_str());
        if (start > end) {
            std::cout << "span read error" << std::endl;
        }
    }

    inline bool bBinary(const Span &other) const {
        if (end < other.start || start > other.end) {
            return false;
        }

        return true;
    }

    inline int length() const {
        return end - start + 1;
    }

    inline int matchNum(const Span &other) const {
        if (end < other.start || start > other.end) {
            return 0;
        }

        int correct = 0;
        for (int idx = other.start; idx <= other.end; idx++) {
            if (idx >= start && idx <= end) {
                correct++;
            }
        }

        return correct;
    }

    inline dtype matchProp(const Span &other) const {
        if (end < other.start || start > other.end) {
            return 0;
        }

        int correct = 0;
        for (int idx = other.start; idx <= other.end; idx++) {
            if (idx >= start && idx <= end) {
                correct++;
            }
        }

        return correct * 2.0 / (length() + other.length());
    }

  public:
    bool operator==(const Span &a1) const {
        return (start == a1.start) && (end == a1.end);
    }

    bool operator!=(const Span &a1) const {
        return (start != a1.start) || (end != a1.end);
    }
};


struct SpanPair {
    Span s1;
    Span s2;
  public:
    SpanPair() : s1(), s2() {

    }

    SpanPair(Span x, Span y) : s1(x), s2(y) {

    }

    SpanPair(const string& strIn) {
        read(strIn);
    }

  public:
    inline std::string str() const {
        stringstream ss;
        ss << s1.str() << "\t" << s2.str();
        return ss.str();
    }

    inline void read(const string& strIn) {
        int idx = strIn.find_first_of("\t");
        s1.read(strIn.substr(0, idx));
        s2.read(strIn.substr(idx + 1, strIn.length() - idx));
    }

    inline bool bBinary(const SpanPair &other) const  {
        if (s1.bBinary(other.s1) && s2.bBinary(other.s2)) {
            return true;
        }
        return false;
    }

    inline dtype matchProp(const SpanPair &other) const {
        return s1.matchProp(other.s1) * s2.matchProp(other.s2);
    }



  public:
    bool operator==(const SpanPair &a1) const {
        return (s1 == a1.s1) && (s2 == a1.s2);
    }

    bool operator!=(const SpanPair &a1) const {
        return (s1 != a1.s1) || (s2 != a1.s2);
    }
};



class CResult {
  public:
    vector<vector<string> > relations;
    //0, no relation && ner; 1, left-to-right; 2, right-to-left; -1, invalid cells
    vector<vector<int> > directions;


    const vector<string> *words;
    const vector<string> *tags;
    const vector<int> *heads;
    const vector<string> *labels;


  public:
    inline void clear() {
        words = NULL;
        tags = NULL;
        heads = NULL;
        labels = NULL;

        clearVec(relations);
        clearVec(directions);
    }

    inline void allocate(const int &size) {

        relations.resize(size);
        directions.resize(size);
        for (int idx = 0; idx < size; idx++) {
            relations[idx].resize(size);
            directions[idx].resize(size);
            for (int idy = 0; idy < size; idy++) {
                if(idx == idy) relations[idx][idy] = "o";
                else if (idx < idy) {
                    relations[idx][idy] = "noRel";
                } else {
                    relations[idx][idy] = "#";
                }
                if (idx <= idy) {
                    directions[idx][idy] = 0;
                } else {
                    directions[idx][idy] = -1;
                }
            }
        }

    }

    inline void extractNERs(unordered_map<string, unordered_set<string> >& entities) {
        static int idx, idy, endpos;
        entities.clear();
        int size = relations.size();
        idx = 0;
        while (idx < size) {
            if (is_start_label(relations[idx][idx])) {
                idy = idx;
                endpos = -1;
                while (idy < size) {
                    if (!is_continue_label(relations[idy][idy], relations[idx][idx], idy - idx)) {
                        endpos = idy - 1;
                        break;
                    }
                    endpos = idy;
                    idy++;
                }
                Span sp(idx, endpos);
                entities[cleanLabel(relations[idx][idx])].insert(sp.str());
                idx = endpos;
            }
            idx++;
        }
    }

    inline void extractRelations(unordered_map<string, unordered_set<string> >& entity_relations) {
        static int idx, idy, endpos;
        unordered_map<int, Span> marked_entities;
        idx = 0;
        int size = relations.size();
        while (idx < size) {
            if (is_start_label(relations[idx][idx])) {
                idy = idx;
                endpos = -1;
                while (idy < size) {
                    if (!is_continue_label(relations[idy][idy], relations[idx][idx], idy - idx)) {
                        endpos = idy - 1;
                        break;
                    }
                    endpos = idy;
                    idy++;
                }
                Span sp(idx, endpos);
                marked_entities[endpos] = sp;
                idx = endpos;
            }
            idx++;
        }
        entity_relations.clear();
        for (int i = 0; i < size - 1; ++i) {
            for (int j = i + 1; j < size; j++) {
                if (directions[i][j] == 1) {
                    entity_relations[relations[i][j]].insert(marked_entities[i].str() + "\t" + marked_entities[j].str());
                }
                if (directions[i][j] == 2) {
                    entity_relations[relations[i][j]].insert(marked_entities[j].str() + "\t" + marked_entities[i].str());
                }

            }
        }
    }

    inline void extractNERs(unordered_set<string>& entities) {
        static int idx, idy, endpos;
        entities.clear();
        int size = relations.size();
        idx = 0;
        while (idx < size) {
            if (is_start_label(relations[idx][idx])) {
                idy = idx;
                endpos = -1;
                while (idy < size) {
                    if (!is_continue_label(relations[idy][idy], relations[idx][idx], idy - idx)) {
                        endpos = idy - 1;
                        break;
                    }
                    endpos = idy;
                    idy++;
                }
                stringstream ss;
                ss << "[" << idx << "," << endpos << "]";
                entities.insert(cleanLabel(relations[idx][idx]) + ss.str());
                idx = endpos;
            }
            idx++;
        }
    }

    inline void extractRelations(unordered_set<string>& entity_relations) {
        static int idx, idy, endpos;
        unordered_map<int, string> marked_entities;
        idx = 0;
        int size = relations.size();
        while (idx < size) {
            if (is_start_label(relations[idx][idx])) {
                idy = idx;
                endpos = -1;
                while (idy < size) {
                    if (!is_continue_label(relations[idy][idy], relations[idx][idx], idy - idx)) {
                        endpos = idy - 1;
                        break;
                    }
                    endpos = idy;
                    idy++;
                }
                stringstream ss;
                ss << "[" << idx << "," << endpos << "]";
                marked_entities[endpos] = ss.str();
                idx = endpos;
            }
            idx++;
        }
        entity_relations.clear();
        for (int i = 0; i < size - 1; ++i) {
            for (int j = i + 1; j < size; j++) {
                if (directions[i][j] > 0) {
                    stringstream ss;
                    ss << "(" << marked_entities[i] << "," << marked_entities[j] << ")=[" << directions[i][j] << "]" << relations[i][j] << std::endl;
                    entity_relations.insert(ss.str());
                }
            }
        }
    }

    inline int size() const {
        return relations.size();
    }

    inline void copyValuesFrom(const CResult &result) {
        static int size;
        size = result.size();
        allocate(size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                relations[i][j] = result.relations[i][j];
                directions[i][j] = result.directions[i][j];
            }
        }

        words = result.words;
        tags = result.tags;
        heads = result.heads;
        labels = result.labels;
    }

    inline void copyValuesFrom(const CResult &result, const vector<string> *pwords,
                               const vector<string> *ptags, const vector<int> *pheads, const vector<string> *plabels) {
        static int size;
        size = result.size();
        allocate(size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                relations[i][j] = result.relations[i][j];
                directions[i][j] = result.directions[i][j];
            }
        }
        words = pwords;
        tags = ptags;
        heads = pheads;
        labels = plabels;
    }


    inline std::string str() const {
        stringstream ss;
        int size = relations.size();
        for (int i = 0; i < size; ++i) {
            ss << "token " << (*words)[i] << " " << (*tags)[i] << " " << (*heads)[i] << " " << (*labels)[i] << " " << relations[i][i] << std::endl;
        }
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; j++) {
                if (directions[i][j] == 1) {
                    ss << "rel " << i << " " << j << " " << 1 << " " << relations[i][j] << std::endl;
                }
                if (directions[i][j] == 2) {
                    ss << "rel " << i << " " << j << " " << -1 << " " << relations[i][j] << std::endl;
                }
            }
        }

        return ss.str();
    }

};


#endif
