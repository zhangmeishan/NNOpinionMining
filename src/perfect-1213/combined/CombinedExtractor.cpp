#include "CombinedExtractor.h"
#include <chrono>
#include <omp.h>
#include <set>

#include "Argument_helper.h"

Extractor::Extractor() {
    srand(0);
}


Extractor::~Extractor() {
}

int Extractor::createAlphabet(vector<Instance> &vecInsts) {
    cout << "Creating Alphabet..." << endl;

    int totalInstance = vecInsts.size();

    unordered_map<string, int> word_stat;
    unordered_map<string, int> rel_stat;

    nerlabels.clear();
    relations.clear();
    assert(totalInstance > 0);
    for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance &instance = vecInsts[numInstance];
        int word_size = instance.words.size();
        for (int idx = 0; idx < word_size; idx++) {
            string curWord = instance.words[idx];
            m_driver._hyperparams.word_stat[curWord]++;
            word_stat[curWord]++;

            string curner = instance.result.relations[idx][idx];
            if (is_start_label(curner)) {
                nerlabels.insert(cleanLabel(curner));
            }
        }

        for (int idx = 0; idx < word_size; idx++) {
            for (int idy = 0; idy < word_size; idy++) {
                int direction = instance.result.directions[idx][idy];
                if (direction == 1) {
                    rel_stat[instance.result.relations[idx][idy]]++;
                    relations.insert(instance.result.relations[idx][idy]);
                    m_driver._hyperparams.rel_dir[instance.result.relations[idx][idy]].insert(1);
                }
                if (direction == 2) {
                    rel_stat[instance.result.relations[idx][idy]]++;
                    relations.insert(instance.result.relations[idx][idy]);
                    m_driver._hyperparams.rel_dir[instance.result.relations[idx][idy]].insert(-1);
                }
            }
        }
    }

    word_stat[unknownkey] = m_options.wordCutOff + 1;
    m_driver._modelparams.embeded_words.initial(word_stat, m_options.wordCutOff);


    // TODO:
    m_driver._hyperparams.ner_labels.clear();
    m_driver._hyperparams.ner_labels.from_string("o");
    unordered_map<string, int>::const_iterator iter;
	unordered_set<string>::const_iterator siter;
    for (siter = nerlabels.begin(); siter != nerlabels.end(); siter++) {
        m_driver._hyperparams.ner_labels.from_string("b-" + *siter);
        m_driver._hyperparams.ner_labels.from_string("m-" + *siter);
        m_driver._hyperparams.ner_labels.from_string("e-" + *siter);
        m_driver._hyperparams.ner_labels.from_string("s-" + *siter);
    }
    m_driver._hyperparams.ner_labels.set_fixed_flag(true);
    m_driver._hyperparams.ner_noprefix_num = nerlabels.size();

    m_driver._hyperparams.rel_labels.clear();
    m_driver._hyperparams.rel_labels.from_string("noRel");
    m_driver._hyperparams.rel_dir["noRel"].insert(1);
    for (iter = rel_stat.begin(); iter != rel_stat.end(); iter++) {
        m_driver._hyperparams.rel_labels.from_string(iter->first);
    }
    m_driver._hyperparams.rel_labels.set_fixed_flag(true);

	int ner_count = m_driver._hyperparams.ner_labels.size();
    int rel_count = m_driver._hyperparams.rel_labels.size();

    m_driver._hyperparams.action_num = ner_count > 2 * rel_count ? ner_count : 2 * rel_count;

    unordered_map<string, int> action_ner_stat, action_rel_stat;
    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
    CResult output;
    CAction answer;
    Metric ner, rel, rel_punc;
    ner.reset();
    rel.reset();
    rel_punc.reset();

    int stepNum;

    for (int numInstance = 0; numInstance < totalInstance; numInstance++) {
        Instance &instance = vecInsts[numInstance];
        stepNum = 0;
        state[stepNum].clear();
        state[stepNum].setInput(instance);
        while (!state[stepNum].IsTerminated()) {
            state[stepNum].getGoldAction(m_driver._hyperparams, instance.result, answer);
            //std::cout << answer.str(&(m_driver._hyperparams)) << " ";

			if (answer.isNER()) {
				action_ner_stat[answer.str(&(m_driver._hyperparams))]++;
			}
			else if (answer.isREL()) {
				action_rel_stat[answer.str(&(m_driver._hyperparams))]++;
			}
            //      TODO: state? answer(gold action)?
            //state[stepNum].prepare(&m_driver._hyperparams, NULL, NULL);
            state[stepNum].move(&(state[stepNum + 1]), answer);
            stepNum++;
        }
        //
        state[stepNum].getResults(output, m_driver._hyperparams);
        //    std::cout << endl;
        //    std::cout << output.str();
        ////
        instance.evaluate(output, ner, rel); //TODO: 不唯一? //FIXME:

        if (!ner.bIdentical() || !rel.bIdentical()) {
            std::cout << "error state conversion!" << std::endl;
            exit(0);
        }


        if ((numInstance + 1) % m_options.verboseIter == 0) {
            std::cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                std::cout << std::endl;
            std::cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    m_driver._modelparams.embeded_ner_actions.initial(action_ner_stat, 0);
	m_driver._modelparams.embeded_rel_actions.initial(action_rel_stat, 0);

    return 0;
}

void Extractor::getGoldActions(vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions) {
    vecActions.clear();

    vector<CAction> acs;
    bool bFindGold;
    Metric ner, rel, rel_punc;
    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
    CResult output;
    CAction answer;
    ner.reset();
    rel.reset();
    rel_punc.reset();
    static int numInstance, stepNum;
    vecActions.resize(vecInsts.size());
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        Instance &instance = vecInsts[numInstance];

        stepNum = 0;
        state[stepNum].clear();
        state[stepNum].setInput(instance);
        while (!state[stepNum].IsTerminated()) {
            state[stepNum].getGoldAction(m_driver._hyperparams, instance.result, answer);
            //std::cout << answer.str(&(m_driver._hyperparams)) << " ";
            state[stepNum].getCandidateActions(acs, &m_driver._hyperparams, &m_driver._modelparams);

            bFindGold = false;
            for (int idz = 0; idz < acs.size(); idz++) {
                if (acs[idz] == answer) {
                    bFindGold = true;
                    break;
                }
            }
            if (!bFindGold) {
                state[stepNum].getCandidateActions(acs, &m_driver._hyperparams, &m_driver._modelparams);
                std::cout << "gold action has been filtered" << std::endl;
                exit(0);
            }

            vecActions[numInstance].push_back(answer);
            state[stepNum].move(&state[stepNum + 1], answer);
            stepNum++;
        }

        state[stepNum].getResults(output, m_driver._hyperparams);
        //FIXME: 内存错误
        //    std::cout << endl;
        //    std::cout << output.str();
        instance.evaluate(output, ner, rel);

        if (!ner.bIdentical() || !rel.bIdentical()) {
            std::cout << "error state conversion!" << std::endl;
            exit(0);
        }

        if ((numInstance + 1) % m_options.verboseIter == 0) {
            cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                cout << std::endl;
            cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
}

void Extractor::train(const string &trainFile, const string &devFile, const string &testFile, const string &modelFile,
                      const string &optionFile) {
    if (optionFile != "")
        m_options.load(optionFile);

    m_options.showOptions();
    vector<Instance> trainInsts, devInsts, testInsts;
    m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
    if (devFile != "")
        m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
    if (testFile != "")
        m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

    vector<vector<Instance> > otherInsts(m_options.testFiles.size());
    for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
        m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
    }

    createAlphabet(trainInsts);

	vector<vector<CAction> > trainInstGoldactions, devInstGoldactions, testInstGoldactions;
	getGoldActions(trainInsts, trainInstGoldactions);
	//getGoldActions(devInsts, devInstGoldactions);
	//getGoldActions(testInsts, testInstGoldactions);

    bool succeed = m_driver._modelparams.word_table.initial(&m_driver._modelparams.embeded_words, m_options.wordEmbFile, true, m_options.wordEmbNormalize);
	if (succeed) {
		m_options.wordEmbSize = m_driver._modelparams.word_table.nDim;
	}
	else {
		m_driver._modelparams.word_table.initial(&m_driver._modelparams.embeded_words, m_options.wordEmbSize, true);
	}

    m_driver._hyperparams.setRequared(m_options);
    m_driver.initial();

    double bestFmeasure = -1;

    int inputSize = trainInsts.size();

    std::vector<int> indexes;
    for (int i = 0; i < inputSize; ++i)
        indexes.push_back(i);

    Metric eval;
    Metric dev_ner, dev_rel;
    Metric test_ner, test_rel;

    unordered_map<string, Metric> dev_ners, dev_rels, dev_prop_ners;
    unordered_map<string, Metric> test_ners, test_rels, test_prop_ners;
    unordered_set<string>::iterator iter_set;
    for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
        dev_ners[*iter_set] = Metric();
        test_ners[*iter_set] = Metric();
        dev_prop_ners[*iter_set] = Metric();
        test_prop_ners[*iter_set] = Metric();
    }

    for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
        dev_rels[*iter_set] = Metric();
        test_rels[*iter_set] = Metric();
    }

    int maxIter = m_options.maxIter;
    int oneIterMaxRound = (inputSize + m_options.batchSize - 1) / m_options.batchSize;
    std::cout << "\nmaxIter = " << maxIter << std::endl;
    int devNum = devInsts.size(), testNum = testInsts.size();

    vector<CResult > decodeInstResults;
    bool bCurIterBetter;
    vector<Instance> subInstances;
    vector<vector<CAction> > subInstGoldActions;
    NRVec<bool> decays;
    decays.resize(maxIter);
    decays = false;
    /* decays[5] = true;
    decays[15] = true;
    decays[30] = true;
    decays[50] = true;
    decays[75] = true; */
    int maxNERIter = m_options.maxNERIter;
    int startBeam = m_options.startBeam;
    m_driver.setGraph(false);
    m_driver.setClip(m_options.clip);
    for (int iter = 0; iter < maxIter; ++iter) {
        //if (decays[iter]) {
		dtype adaAlpha = m_options.adaAlpha / (1 + m_options.decay * iter);
        m_driver.setUpdateParameters(m_options.regParameter, adaAlpha, m_options.adaEps);
		std::cout << "\nadaAlpha = " << m_driver ._ada._alpha << std::endl;
        //}
        if (startBeam >= 0 && iter >= startBeam) {
            m_driver.setGraph(true);
        }

		if(m_options.reach_drop > 0)m_driver.setDropFactor(iter * 1.0 / m_options.reach_drop);

        std::cout << "##### Iteration " << iter << std::endl;
        srand(iter);
        bool bEvaluate = false;

        if (m_options.batchSize == 1) {
            auto t_start_train = std::chrono::high_resolution_clock::now();
        eval.reset();
        bEvaluate = true;
            random_shuffle(indexes.begin(), indexes.end());
            std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
        for (int idy = 0; idy < inputSize; idy++) {
            subInstances.clear();
            subInstGoldActions.clear();
            subInstances.push_back(trainInsts[indexes[idy]]);
            subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
            double cost = m_driver.train(subInstances, subInstGoldActions, iter < maxNERIter);

            eval.overall_label_count += m_driver._eval.overall_label_count;
            eval.correct_label_count += m_driver._eval.correct_label_count;

            if ((idy + 1) % (m_options.verboseIter) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
                //if (m_driver._batch >= m_options.batchSize) {
                //    m_driver.updateModel();
                //}
                m_driver.updateModel();
            }
            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        } else {
            eval.reset();
            auto t_start_train = std::chrono::high_resolution_clock::now();
            bEvaluate = true;
            for (int idk = 0; idk < (inputSize + m_options.batchSize - 1) / m_options.batchSize; idk++) {
                random_shuffle(indexes.begin(), indexes.end());
                subInstances.clear();
                subInstGoldActions.clear();
                for (int idy = 0; idy < m_options.batchSize; idy++) {
                    subInstances.push_back(trainInsts[indexes[idy]]);
                    subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
                }
                double cost = m_driver.train(subInstances, subInstGoldActions, iter < maxNERIter);

                eval.overall_label_count += m_driver._eval.overall_label_count;
                eval.correct_label_count += m_driver._eval.correct_label_count;

                if ((idk + 1) % (m_options.verboseIter) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    std::cout << "current: " << idk + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
                }

                m_driver.updateModel();
            }

            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        }

        if (bEvaluate && devNum > 0) {
            auto t_start_dev = std::chrono::high_resolution_clock::now();
            std::cout << "Dev start." << std::endl;
            bCurIterBetter = false;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            dev_ner.reset();
            dev_rel.reset();

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                dev_ners[*iter_set].reset();
                dev_prop_ners[*iter_set].reset();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                dev_rels[*iter_set].reset();
            }

            predict(devInsts, decodeInstResults);

            for (int idx = 0; idx < devInsts.size(); idx++) {
                devInsts[idx].evaluate(decodeInstResults[idx], dev_ner, dev_rel);
                devInsts[idx].evaluateBinyNER(decodeInstResults[idx], dev_ners);
                devInsts[idx].evaluatePropNER(decodeInstResults[idx], dev_prop_ners);
                devInsts[idx].evaluateBinyREL(decodeInstResults[idx], dev_rels);
            }
            auto t_end_dev = std::chrono::high_resolution_clock::now();
            std::cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
            std::cout << "dev:" << std::endl;
            dev_ner.print();
            dev_rel.print();

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " ";
                dev_ners[*iter_set].print();
            }

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " PROP ";
                dev_prop_ners[*iter_set].print();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                std::cout << *iter_set << " ";
                dev_rels[*iter_set].print();
            }

            if (!m_options.outBest.empty() && dev_rel.getAccuracy() > bestFmeasure) {
                m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
                bCurIterBetter = true;
            }
        }

        if (testNum > 0) {
            clock_t time_start = clock();
            auto t_start_test = std::chrono::high_resolution_clock::now();
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset();
            test_rel.reset();

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                test_ners[*iter_set].reset();
                test_prop_ners[*iter_set].reset();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                test_rels[*iter_set].reset();
            }

            predict(testInsts, decodeInstResults);
            for (int idx = 0; idx < testInsts.size(); idx++) {
                testInsts[idx].evaluate(decodeInstResults[idx], test_ner, test_rel);
                testInsts[idx].evaluateBinyNER(decodeInstResults[idx], test_ners);
                testInsts[idx].evaluatePropNER(decodeInstResults[idx], test_prop_ners);
                testInsts[idx].evaluateBinyREL(decodeInstResults[idx], test_rels);
            }
            auto t_end_test = std::chrono::high_resolution_clock::now();
            std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
            std::cout << "test:" << std::endl;
            test_ner.print();
            test_rel.print();

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " ";
                test_ners[*iter_set].print();
            }

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " PROP ";
                test_prop_ners[*iter_set].print();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                std::cout << *iter_set << " ";
                test_rels[*iter_set].print();
            }


            if (!m_options.outBest.empty() && bCurIterBetter) {
                m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
            }
        }

        for (int idx = 0; idx < otherInsts.size(); idx++) {
            auto t_start_other = std::chrono::high_resolution_clock::now();
            std::cout << "processing " << m_options.testFiles[idx] << std::endl;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset();
            test_rel.reset();
            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                test_ners[*iter_set].reset();
                test_prop_ners[*iter_set].reset();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                test_rels[*iter_set].reset();
            }

            predict(otherInsts[idx], decodeInstResults);
            for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
                otherInsts[idx][idy].evaluate(decodeInstResults[idy], test_ner, test_rel);
                otherInsts[idx][idy].evaluateBinyNER(decodeInstResults[idy], test_ners);
                otherInsts[idx][idy].evaluatePropNER(decodeInstResults[idy], test_prop_ners);
                otherInsts[idx][idy].evaluateBinyREL(decodeInstResults[idy], test_rels);
            }
            auto t_end_other = std::chrono::high_resolution_clock::now();
            std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_other - t_start_other).count() << std::endl;
            std::cout << "test:" << std::endl;
            test_ner.print();
            test_rel.print();

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " ";
                test_ners[*iter_set].print();
            }

            for (iter_set = nerlabels.begin(); iter_set != nerlabels.end(); iter_set++) {
                std::cout << *iter_set << " PROP ";
                test_prop_ners[*iter_set].print();
            }

            for (iter_set = relations.begin(); iter_set != relations.end(); iter_set++) {
                std::cout << *iter_set << " ";
                test_rels[*iter_set].print();
            }

            if (!m_options.outBest.empty() && bCurIterBetter) {
                m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
            }
        }


        if (m_options.saveIntermediate && dev_rel.getAccuracy() > bestFmeasure) {
            std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
            bestFmeasure = dev_rel.getAccuracy();
            writeModelFile(modelFile);
        }

    }
}

void Extractor::predict(vector<Instance>& inputs, vector<CResult> &outputs) {
    int sentNum = inputs.size();
    if (sentNum <= 0) return;
    vector<Instance> batch_sentences;
    vector<CResult> batch_outputs;
    outputs.resize(sentNum);
    int sent_count = 0;
    for (int idx = 0; idx < sentNum; idx++) {
        batch_sentences.push_back(inputs[idx]);
        if (batch_sentences.size() == m_options.batchSize || idx == sentNum - 1) {
            m_driver.decode(batch_sentences, batch_outputs);
            batch_sentences.clear();
            for (int idy = 0; idy < batch_outputs.size(); idy++) {
                outputs[sent_count].copyValuesFrom(batch_outputs[idy]);
                outputs[sent_count].words = &(inputs[sent_count].words);
                outputs[sent_count].tags = &(inputs[sent_count].tags);
                outputs[sent_count].heads = &(inputs[sent_count].heads);
                outputs[sent_count].labels = &(inputs[sent_count].labels);
                sent_count++;
            }
        }
    }

    if (outputs.size() != sentNum) {
        std::cout << "decoded number not match" << std::endl;
    }

}

void Extractor::test(const string &testFile, const string &outputFile, const string &modelFile) {

}


void Extractor::loadModelFile(const string &inputModelFile) {

}

void Extractor::writeModelFile(const string &outputModelFile) {

}

int main(int argc, char *argv[]) {
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
    std::string wordEmbFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    dsr::Argument_helper ah;
    int memsize = 0;
    int threads = 1;

    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training",
                        trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training",
                        devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
                        "testing corpus to train a model or input file to test a model, optional when training and must when testing",
                        testFile);
    ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
    ah.new_named_string("word", "wordEmbFile", "named_string",
                        "pretrained word embedding file to train a model, optional when training", wordEmbFile);
    ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training",
                        optionFile);
    ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
    ah.new_named_int("mem", "memsize", "named_int", "memory allocated for tensor nodes", memsize);
    ah.new_named_int("th", "thread", "named_int", "number of threads for openmp", threads);

    ah.process(argc, argv);

    //  omp_set_num_threads(threads);
    //  Eigen::setNbThreads(threads);
    //  mkl_set_num_threads(4);
    //  mkl_set_dynamic(false);
    //  omp_set_nested(false);
    //  omp_set_dynamic(false);
    Extractor extractor;
    if (bTrain) {
        extractor.train(trainFile, devFile, testFile, modelFile, optionFile);
    } else {
        extractor.test(testFile, outputFile, modelFile);
    }

    //test(argv);
    //ah.write_values(std::cout);

}
