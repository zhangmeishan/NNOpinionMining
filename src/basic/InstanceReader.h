#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include "Utf.h"
#include <sstream>

class InstanceReader : public Reader {
  public:
    InstanceReader() {
    }

    ~InstanceReader() {
    }

    Instance *getNext() {
        m_instance.clear();
        static string strLine;
        static vector<string> vecLine;
        vecLine.clear();
        while (1) {
            if (!my_getline(m_inf, strLine)) {
                break;
            }
            if (strLine.empty())
                break;
            vecLine.push_back(strLine);
        }

        static vector<string> charInfo;
        static vector<string> tmpInfo;
        static int count, parent_id;
        count = 0;
        for (int i = 0; i < vecLine.size(); i++) {
            split_bychar(vecLine[i], charInfo, ' ');
            if (charInfo[0] == "token") {
                count++;
            }
        }
        if (count == 1) {
            //std::cout << "token error: token size is only one!" << std::endl;
            return getNext();
        }
        m_instance.allocate(count);

        char previous_mark = 'o';
        for (int i = 0; i < count; i++) {
            split_bychar(vecLine[i], charInfo, ' ');
            if (charInfo.size() != 6) {
                std::cout << "token error" << std::endl;
                return getNext();
            }
            m_instance.words[i] = charInfo[1];
            m_instance.tags[i] = charInfo[2];
            m_instance.heads[i] = atoi(charInfo[3].c_str());
            m_instance.labels[i] = charInfo[4];
            if (previous_mark == 'o' || previous_mark == 's' || previous_mark == 'e') {
                if (charInfo[5][0] == 'm' || charInfo[5][0] == 'e') {
                    std::cout << "ner label error" << std::endl;
                    return getNext();
                }
            } else if (previous_mark == 'b') {
                if (charInfo[5][0] == 'b' || charInfo[5][0] == 's' || charInfo[5][0] == 'o') {
                    std::cout << "ner label error" << std::endl;
                    return getNext();
                }
            } else if (previous_mark == 'm') {
                if (charInfo[5][0] == 'b' || charInfo[5][0] == 's' || charInfo[5][0] == 'o') {
                    std::cout << "ner label error" << std::endl;
                    return getNext();
                }
            }

            m_instance.result.relations[i][i] = charInfo[5];
            previous_mark = charInfo[5][0];
        }

        m_instance.result.words = &m_instance.words;
        m_instance.result.tags = &m_instance.tags;
        m_instance.result.heads = &m_instance.heads;
        m_instance.result.labels = &m_instance.labels;

        for (int i = count; i < vecLine.size(); i++) {
            split_bychar(vecLine[i], charInfo, ' ');
            if (charInfo[0] != "rel" || charInfo.size() != 5) {
                std::cout << "relation error" << std::endl;
                return getNext();
            }
            if (charInfo[4] == "noRel") {
                continue;
            }
            int arg1 = atoi(charInfo[1].c_str());
            int arg2 = atoi(charInfo[2].c_str());
            if (arg1 >= arg2 || arg1 < 0 || arg2 >= count) {
                std::cout << "relation error" << std::endl;
                return getNext();
            }
            int direction = atoi(charInfo[3].c_str());
            if (direction == 1) {
                m_instance.result.directions[arg1][arg2] = 1;
            } else  if (direction == -1) {
                m_instance.result.directions[arg1][arg2] = 2;
            } else {
                std::cout << "relation direction error" << std::endl;
                return getNext();
            }
            m_instance.result.relations[arg1][arg2] = charInfo[4];
        }
        m_instance.getChars();
        return &m_instance;
    }
};

#endif

