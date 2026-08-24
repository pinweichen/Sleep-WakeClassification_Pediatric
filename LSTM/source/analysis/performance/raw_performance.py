class RawPerformance(object):
    def __init__(self, true_labels, class_probabilities, subject=None, predicted_labels=None, feats=None,
                 model_state_dict=None, best_hyperparams=None, attention_weights=None):
        self.true_labels = true_labels
        self.class_probabilities = class_probabilities
        self.subject = subject
        self.predicted_labels = predicted_labels
        self.feature_importance = feats       # dict  {feature_name: importance_score}
        self.model_state_dict = model_state_dict  # torch state_dict for the best model this fold
        self.best_hyperparams = best_hyperparams  # dict  {param_name: value} from grid search

