import os


class ExpConfig:
    """
        expConfig is used to configure an experiment,
        setting up the elements: dataset, model, experiment setting, and evaluation metrics.
    """

    def __init__(
        self,
        dataset,  # the datasets
        setting,  # experiment setting to run the experiment
        model,  # the model to be testd
        dataset_test=None
    ):
        self.dataset = dataset
        self.setting = setting
        self.model = model
        self.dataset_test = dataset_test

    def run(self):
        """
            Run the experiment
        """
        # set up the experiment setting
        self.setting.setup(dataset=self.dataset, model=self.model, dataset_test=self.dataset_test)
        if self.setting.istrain or self.setting.continue_train:
            # run experiment setting
            self.setting.run()
        else:
            self.setting.test()
