/*
 * NewMetric.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_NEWMETRIC_H_
#define SRC_NEWMETRIC_H_


using namespace std;

struct NewMetric {
  public:
    int overall_label_count;
    dtype correct_label_count;
    int predicated_label_count;

  public:
    NewMetric() {
        overall_label_count = 0;
        correct_label_count = 0;
        predicated_label_count = 0;
    }

    ~NewMetric() {}

    void reset() {
        overall_label_count = 0;
        correct_label_count = 0;
        predicated_label_count = 0;
    }

    void set(const NewMetric& other) {
        overall_label_count = other.overall_label_count;
        correct_label_count = other.correct_label_count;
        predicated_label_count = other.predicated_label_count;
    }


    double getAccuracy() {
        if (overall_label_count + predicated_label_count == 0) return 1.0;
        if (predicated_label_count == 0) {
            return correct_label_count*1.0 / overall_label_count;
        } else {
            return correct_label_count*2.0 / (overall_label_count + predicated_label_count);
        }
    }


    void print() {
        if (predicated_label_count == 0) {
            std::cout << "Accuracy:\tP=" << correct_label_count << "/" << overall_label_count
                      << "=" << correct_label_count*1.0 / overall_label_count << std::endl;
        } else {
            std::cout << "Recall:\tP=" << correct_label_count << "/" << overall_label_count << "=" << correct_label_count*1.0 / overall_label_count
                      << ", " << "Accuracy:\tP=" << correct_label_count << "/" << predicated_label_count << "=" << correct_label_count*1.0 / predicated_label_count
                      << ", " << "Fmeasure:\t" << correct_label_count*2.0 / (overall_label_count + predicated_label_count) << std::endl;
        }
    }

};

#endif /* SRC_EXAMPLE_H_ */

