import numpy as np;
import LinRegLearner as lrl;
import BagLearner as bl;
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose;
        self.learners = [];
    def author(self):
        return "welsakka3";
    def study_group(self):
        return "welsakka3";
    def add_evidence(self, data_x, data_y):
        for i in range(0, 20):
            self.learners.append(bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False));
            self.learners[i].add_evidence(data_x, data_y);
    def query(self, points):
        y_predict = np.empty((points.shape[0],0));
        for learner in self.learners:
            y_predict = np.append(y_predict, learner.query(points).reshape(-1, 1), axis=1);
        return np.array([np.mean(y_predict[i,:]) for i in range (0, y_predict.shape[0])]);