""" TODO """
import sys
import os
from gplpy.gggp.grammar import CFG, ProbabilisticModel
from gplpy.evo.evolution import Experiment, Setup, Problem, Evolution_EDA, Evolution_WX
from gplpy.evo.log import DBLogger
from gplpy.gggp.derivation import Derivation, WX, OnePointMutation
from gplpy.gggp.metaderivation import MetaDerivation, EDA


from bson.objectid import ObjectId
import tensorflow

gp_setups = {}

# SETUP EXAMPLES ###########################################################
WX_setup = Setup(name='WX', evolution=Evolution_WX, max_recursions = 250, probabilistic_model=ProbabilisticModel.uniform, crossover=WX, selection_rate=2, mutation=OnePointMutation, mutation_rate=0.05, immigration_rate=.15)
EDA_setup = Setup(name='EDA', evolution=Evolution_EDA, max_recursions = 250, crossover=EDA, selection_rate=0.5, exploration_rate=0., model_update_rate=.5, offspring_rate=1., immigration_rate=.15)
EDX_setup = Setup(name='EDX', evolution=Evolution_EDA, max_recursions = 250, crossover=EDA, selection_rate=0.5, exploration_rate=0.001, model_update_rate=.5, offspring_rate=.25, immigration_rate=.15)


class DFFNN(Problem):
    epochs = 10
    batch_size = 128

    @staticmethod
    def fitness(individual, args):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation

        topology = list(map(len, str(individual.derivation).replace(' ','').split("0")))
        input_size, num_classes, X_train, X_test, y_train, y_test = args

        model = Sequential()
        # First layer and hidden layer
        model.add(Dense(topology.pop(0), activation='relu', input_dim=input_size))
        # Hidden layers
        for layer_size in topology:
            model.add(Dense(layer_size, activation='relu'))
        # Output layer
        model.add(Dense(1 if num_classes==2 else num_classes, activation='sigmoid' if num_classes==2 else 'softmax'))

        # Setup optimizer
        model.compile(loss='binary_crossentropy'if num_classes==2 else 'categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            epochs=DFFNN.epochs,
                            batch_size=DFFNN.batch_size,
                            verbose=0,
                            validation_data=(X_test, y_test))

        score = model.evaluate(X_test, y_test, verbose=0)
        #print('Test loss:', score[0])
        #print('Test accuracy:', score[1])
        individual._fitness = score[0]
        individual.learning_iterations = len(history.epoch)
        individual.mature.set()

if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    ## IS 
    study = "DFFNN"
    study_id = None
    #study_id = ObjectId("590b99f0d140a535c9dfbe12")

    # Grammar initialization
    grammar_file = os.getcwd() + '/gr/' + study.replace(' ', '_') + '.gr'
    gr = CFG(grammar_file)

    # logger initialization
    # Set to True to log into mongodb
    logger = False
    if logger:
        logger = DBLogger(server='cluster0-21cbd.gcp.mongodb.net', user='gplpy_logger', password='q1e3w2r4', cluster=True)
        if study_id:
            logger.resume_study(study_id=study_id, grammar=grammar_file[5:])
        else:
            logger.new_study(study=study, grammar=grammar_file[5:])
        logger.createDeleteStudyRoutine()

    # Setup problem    
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.utils import to_categorical

    exp_name = "Cancer"
    X, y = datasets.load_breast_cancer(return_X_y=True)
    num_classes = len(set(y))
    input_size = X.shape[1]
    if num_classes > 2:
        y = to_categorical(y, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = y_train.reshape(y_train.size, 1)
    y_test = y_test.reshape(y_test.size, 1)
    args = (input_size, num_classes, X_train, X_test, y_train, y_test)

    # Run
    samples = 1
    ids =Experiment(study=study, experiment=exp_name, grammar=gr, problem=DFFNN, fitness_args=args,
                    setups=[EDX_setup, EDA_setup, WX_setup], logger=logger, samples=samples).run()
    
    if logger and logger.server is 'localhost':
        logger.plot_experiments_evolution()
        logger.plot_range_study()
        logger.report_statistics()