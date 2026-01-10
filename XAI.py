from method1.model import method1
from method2.model import method2
from method3.model import method3
from method4.model import method4

from AI_models.CDCN import CDCN
from AI_models.CDCN import DGCNN


"""
This will be the main XAI module. Its function is simply to call the functions
implemented in the remaining directories [method1, ..., methodn]. When someone
wants to use this directory in the future, the idea is that they interact with
this class.

When initializing this class, the model must be loaded (if you think additional
elements should be added, we can discuss them as the implementation progresses).
Each function must receive the model and any additional arguments you consider
necessary to implement the method.

Each implemented method must save the results it generates to the specified
savepath, and it should only print results if the verbose flag is enabled.

*** Don't use emojis in prints or code; all the code has to be in English.

The ideal outcome is to implement at least XAI methods for CDC and DGCNN on the 
SEED and SEED-IV databases.
"""


class XAI_module():
    def __init__(self, model_path, savepath, verbose = True):
        # Load model
        self.model = self.load_model(model_path)
        # Path to save all results
        self.save_path = savepath
        # Verbose
        self.verbose = verbose

    def load_model(self, model_path):
        #Load model code
        # Create model
        pass
    
    def method1(self, args):
        method1(self.model, self.save_path, self.verbose, args)
    
    def method2(self, args):
        method2(self.model, self.save_path, self.verbose, args)
    
    def method3(self, args):
        method3(self.model, self.save_path, self.verbose, args)

    def method4(self, args):
        method4(self.model, self.save_path, self.verbose, args)