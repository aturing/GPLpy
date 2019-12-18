"""TODO"""
import matplotlib
matplotlib.use('Agg')

import os
import datetime
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from pandas import Series
from scipy import stats
import numpy as np


class DBLogger(object):
    """Class for log process into MongoDB"""

    def __init__(self, server='127.0.0.1', port='27017', user=None, password=None, cluster=False):
        """Constructor. Initialize variables for connection
           server -- (default localhost)
           port -- (default 27017)
           user -- (default None)
           password -- (default None)
        """
        self.server = server
        self.port = port
        self.user = user
        self.password = password
        self.cluster = cluster


        self.study_id = None
        self.experiment_id = None
        self.sample_id = None

        self.db = self.connect()

        self.study_path = None
        self.study_name = None
        self.experiment_name = None

        self.grammar = None
        # Blue: #0072bd, Orange:#d95319, Yellow:#edb120, Green:#77ac30, Light-blue:#4dbeee, Red:#a2142f, Purple:#7e2f8e
        self.plot_colors = ['#0072bd', '#d95319', '#edb120', '#77ac30', '#4dbeee', '#a2142f', '#7e2f8e']
        self.plot_markers = ['^', 'o', 's', '+', 'x']

    def connect(self):
        if self.cluster:
          db = MongoClient('mongodb+srv://' + self.user +':' + self.password+'@' + self.server + '/test?retryWrites=true&w=majority').log
        else:
            if self.user is None:
                db = MongoClient('mongodb://'+ self.server + ':' + self.port +'/test').log
            else:
                db = MongoClient('mongodb://'+ self.user +':' + self.password + '@' + self.server + ':' + self.port + '/test').log
        return db

    def new_study(self, study, grammar):
        self.grammar = grammar.split("/")[-1]
        self.study_id = self.db.study.insert_one(
            {'study': study, 'date': datetime.datetime.utcnow(), 'grammar': grammar}).inserted_id

    def resume_study(self, study_id, grammar):
        self.grammar = grammar.split("/")[-1]
        self.study_id = study_id

    def new_experiment(self, name, setup_name, initialization, crossover):
        self.experiment_name = name
        self.experiment_id = self.db.experiment.insert_one({'name': name,
                                                            'setup_name': setup_name,
                                                            'study_id': self.study_id,
                                                            'initialization': str(initialization),
                                                            'crossover': str(crossover),
                                                            'date': datetime.datetime.utcnow()}).inserted_id
        self.add_experiment_to_study()
        return self.experiment_id

    def new_sample(self, sample):
        self.sample_id = self.db.sample.insert_one({'study_id': self.study_id,
                                                    'experiment_id': self.experiment_id,
                                                    'sample': sample,
                                                    'date': datetime.datetime.utcnow()
                                                   }).inserted_id
        return self.sample_id

    def log_evolution(self, best_fitness, average_fitness, improvement):
        self.db.sample.update({'experiment_id': self.experiment_id, '_id': self.sample_id},
                              {'$push': {'best_fitness': best_fitness,
                                         'average_fitness': average_fitness,
                                         'improvement': improvement}})

    def add_experiment_to_study(self):
        self.db.study.update({'_id': self.study_id},
                             {'$push': {'experiments_id': self.experiment_id}})

    def log_experiment(self, fitness, iterations, learning_iterations=0):
        self.db.experiment.update({'_id': self.experiment_id},
                                    {'$push': {'samples_id': self.sample_id,
                                                'fitness': fitness,
                                                'iterations': iterations,
                                                'learning_iterations': learning_iterations}})                                       

    def save_experiment_statistics(self, experiment_id, it_stats, f_stats, l_it_stats):
        iterations_mean = it_stats.mean()
        iterations_deviation = it_stats.std(ddof=1)
        iterations_confidence_interval = stats.norm.interval(0.95, loc=iterations_mean, scale=iterations_deviation)
        fitness_mean = f_stats.mean()
        fitness_deviation = f_stats.std(ddof=1)
        fitness_confidence_interval = stats.norm.interval(0.95, loc=fitness_mean, scale=fitness_deviation)
        learning_iterations_mean = l_it_stats.mean()
        learning_iterations_deviation = l_it_stats.std(ddof=1)
        learning_iterations_confidence_interval = stats.norm.interval(0.95, loc=fitness_mean, scale=fitness_deviation)

        self.db.experiment.update({'_id': experiment_id},
                                  {'$set': {'statistics':
                                                {'iterations_mean': iterations_mean,
                                                 'iterations_deviation': iterations_deviation,
                                                 'iterations_confidence_interval': iterations_confidence_interval,
                                                 'iterations_descriptive':
                                                     str(it_stats.describe(percentiles=[.05, .25, 0.5, .75, .95])),
                                                 'fitness_mean': fitness_mean,
                                                 'fitness_deviation': fitness_deviation,
                                                 'fitness_confidence_interval': fitness_confidence_interval,
                                                 'fitness_descriptive':
                                                     str(f_stats.describe(
                                                         percentiles=[.05, .25, 0.5, .75, .95])),
                                                 'learning_iterations_mean': learning_iterations_mean,
                                                 'learning_iterations_deviation': learning_iterations_deviation,
                                                 'learning_iterations_interval': learning_iterations_confidence_interval,
                                                 'learning_iterations_descriptive':
                                                     str(l_it_stats.describe(
                                                         percentiles=[.05, .25, 0.5, .75, .95]))}}})

    def obtain_statistics(self, experiment_name=None):
        results = self.get_experiment_results(experiment_name)
        for experiment_set in results:
            setups = experiment_set['setups']
            while len(setups):
                s1 = setups.pop()
                s1_it_stats = Series(s1['iterations'])
                s1_f_stats = Series(s1['fitness'])
                s1_l_it_stats = Series(s1['learning_iterations'])
                self.save_experiment_statistics(s1['experiment_id'], s1_it_stats, s1_f_stats, s1_l_it_stats)

                for s2 in setups:
                    it_win = None
                    f_win = None
                    l_it_win = None
                    s2_it_stats = Series(s2['iterations'])
                    s2_f_stats = Series(s2['fitness'])
                    
                    s2_l_it_stats = Series(s2['learning_iterations'])

                    f_it_anova, p_it_anova = stats.f_oneway(s1['iterations'], s2['iterations'])
                    if p_it_anova < 0.05:
                        it_win = s1 if s1_it_stats.mean() < s2_it_stats.mean() else s2

                    f_f_anova, p_f_anova = stats.f_oneway(s1['fitness'], s2['fitness'])
                    if p_f_anova < 0.05:
                        f_win = s1 if s1_f_stats.mean() < s2_f_stats.mean() else s2
                    
                    f_l_it_anova, p_l_it_anova = stats.f_oneway(s1['learning_iterations'], s2['learning_iterations'])
                    if p_l_it_anova < 0.05:
                        l_it_win = s1 if s1_l_it_stats.mean() < s2_l_it_stats.mean() else s2

                    self.save_experiment_anova(experiment_set['_id'], s1, s2,
                                               p_it_anova, f_it_anova, it_win,
                                               p_f_anova, f_f_anova, f_win,
                                               p_l_it_anova, f_l_it_anova, l_it_win)

    def save_experiment_anova(self, experiment, e1, e2,
                              p_it_anova, f_it_anova, it_lower,
                              p_f_anova, f_f_anova, f_lower,
                              p_l_it_anova, f_l_it_anova, l_it_lower):
        iteration_lower = None
        fitness_lower = None
        if it_lower is not None:
            iteration_lower = {'experiment_id': it_lower['experiment_id'],
                               'setup_name': it_lower['setup_name']}

        if f_lower is not None:
            fitness_lower = {'experiment_id': f_lower['experiment_id'],
                             'setup_name': f_lower['setup_name']}

        if l_it_lower is not None:
            learning_iteration_lower = {'experiment_id': l_it_lower['experiment_id'],
                                        'setup_name': l_it_lower['setup_name']}

        self.db.anova.insert_one({'study_id': self.study_id,
                                  'experiment': experiment,
                                  'experiments': [{'experiment_id': e1['experiment_id'],
                                                   'setup_name': e1['setup_name']},
                                                  {'experiment_id': e2['experiment_id'],
                                                   'setup_name': e2['setup_name']}],
                                  'iterations': {'p': p_it_anova,
                                                 'f': f_it_anova,
                                                 'lower': iteration_lower},
                                  'fitness': {'p': p_f_anova,
                                              'f': f_f_anova,
                                              'lower': fitness_lower},
                                  'learning_iterations': {'p': p_l_it_anova,
                                                          'f': f_l_it_anova,
                                                          'lower': iteration_lower},})

    def get_experiments(self):
        return self.db.study.find_one({'_id': self.study_id})['experiments']

    def get_experiment(self, experiment_id):
        return self.db.experiment.find_one({'_id': experiment_id})

    def get_experiment_iterations(self):
        return self.db.experiment.find_one({'_id': self.experiment_id})['iterations']

    def get_experiment_fitness(self):
        return self.db.experiment.find_one({'_id': self.experiment_id})['fitness']
    
    def get_experiment_results(self, experiment_name=None):
        if experiment_name is not None:
            return self.db.experiment.aggregate([{'$match': {'study_id': self.study_id,
                                                             'name': experiment_name}},
                                                 {'$sort': {"setup_name": -1}},
                                                 {'$group': {'_id': '$name',
                                                             'setups': {'$push': {'experiment_id': '$_id',
                                                                                  'setup_name': '$setup_name',
                                                                                  'fitness': '$fitness',
                                                                                  'iterations': '$iterations',
                                                                                  'learning_iterations': '$learning_iterations'}}}}
                                                ])
        else:
            return self.db.experiment.aggregate([{'$match': {'study_id': self.study_id}},
                                                 {'$sort': {"setup_name": -1}},
                                                 {'$group': {'_id': '$name',
                                                             'setups': {'$push': {'experiment_id': '$_id',
                                                                                  'setup_name': '$setup_name',
                                                                                  'fitness': '$fitness',
                                                                                  'iterations': '$iterations',
                                                                                  'learning_iterations': '$learning_iterations'}}}}
                                                ])

    def report_statistics(self):
        if self.study_path is None:
            setup = self.db.study.find_one({'_id': self.study_id}, {'_id': 0, 'study': 1})
            if setup is None:
                raise Exception('There is no experiment with that ObjectID')

            self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
            self.study_name = setup['study']

        if os.path.basename(os.getcwd()) != self.study_path:
            if not os.path.isdir('./experiments/' + self.study_path):
                os.mkdir('./experiments/' + self.study_path)
            os.chdir('./experiments/' + self.study_path)

        # cursor = self.db.experiment.aggregate([
        #     {'$match': {'study_id': self.study_id}},
        #     {'$sort': {'experiment': 1}},
        #     {'$group': {'_id': {'crossover': '$crossover', 'initialization': '$initialization'},
        #                 'experiments': {'$push': '$experiment'},
        #                 'fitness_mean': {'$push': '$statistics.fitness_mean'},
        #                 'fitness_deviation': {'$push': '$statistics.fitness_deviation'},
        #                 'iteration_mean': {'$push': '$statistics.iteration_mean'},
        #                 'iteration_deviation': {'$push': '$statistics.iteration_deviation'}}}])

        cursor = self.db.experiment.aggregate([
            {'$match': {'study_id': self.study_id}},
            {'$sort': {'experiment': 1}},
            {'$group': {'_id': '$setup_name',
                        'experiments_name': {'$push': '$experiment'},
                        'experiments': {'$push': {'_id': '$_id',
                                                  'name': '$name',
                                                  'date': '$date',
                                                  'statistics': '$statistics'}}}}], allowDiskUse=True)
        if cursor is None:
            raise Exception('There is no logged data with that study id')

        experiments_name = None
        for setup in cursor:
            if not experiments_name:
                experiments_name = setup['experiments_name']

            with open('./statistics_' + setup['_id'] + '.txt', 'w+') as f:
                f.write('Descriptive Statistics')
                f.write('\n======================')
                f.write('\nStudy: ' + self.study_name + 'Id:' + str(self.study_id))
                f.write('\nGrammar: ' + self.grammar)
                f.write('\nSetup: ' + setup['_id'])
                f.write('\n')

                for e in setup['experiments']:
                    # Reporting descriptive statistics
                    f.write('\n----------------------------------------------------->' + str(e['name']) + '\n')
                    f.write('\nExperiment: ' + str(e['name']))
                    if 'statistics' in e:
                        f.write('\n Fitness descriptive statistics\n')
                        f.write(e['statistics']['fitness_descriptive'])
                        f.write('\n\n Iteration descriptive statistics\n')
                        f.write(e['statistics']['iterations_descriptive'])
                        f.write('\n\n Iteration descriptive statistics\n')
                        f.write(e['statistics']['learning_iterations_descriptive'])
                f.close()

        for experiment in experiments_name:
            with open('./anova_experiment_' + str(experiment) + '.txt', 'w+') as f:
                f.write('Anova')
                f.write('\n=====')
                f.write('\nStudy: ' + self.study_name + 'Id:' + str(self.study_id))
                f.write('\nGrammar: ' + self.grammar)
                f.write('\nExperiment: ' + str(experiment))

                anovas = self.db.anova.aggregate([{'$match': {'study_id': self.study_id,
                                                              'experiment': experiment}},
                                                  {'$project': {'_id': 0,
                                                                'experiments.setup_name': 1,
                                                                'iterations': 1,
                                                                'fitness': 1}}], allowDiskUse=True)

                if anovas is None:
                    f.close()
                    raise Exception('There is no logged data with that study id and experiments ids')

                for anova in anovas:
                    f.write('\n\n----------------------------------------------------->' +
                            anova['experiments'][0]['setup_name'] + ' vs ' + anova['experiments'][1]['setup_name']
                            )
                    if anova['fitness']['lower'] or anova['iterations']['lower']:
                        f.write(' *****')
                    f.write('\n\nSetup 1:' + anova['experiments'][0]['setup_name'])
                    f.write('\nSetup 2:' + anova['experiments'][1]['setup_name'])
                    f.write('\n-------')
                    f.write('\n\nIteration Anova')
                    f.write('\n---------------')
                    f.write('\n f: ' + str(anova['iterations']['f']) + '\tp: ' + str(anova['iterations']['p']))
                    if anova['iterations']['lower']:
                        f.write('\n' + anova['iterations']['lower']['setup_name'].upper() +
                                ' ITERATIONS ARES SIGNIFICANTLY LOWER ****')
                    f.write('\n\nFitness Anova')
                    f.write('\n---------------')
                    f.write('\n f: ' + str(anova['fitness']['f']) + '\tp: ' + str(anova['fitness']['p']))
                    if anova['fitness']['lower']:
                        f.write('\n****' + anova['fitness']['lower']['setup_name'].upper() +
                                ' FITNESS IS SIGNIFICANTLY LOWER ****')

                f.close()

    def plot_range_study(self):
        if self.study_path is None:
            setup = self.db.study.find_one({'_id': self.study_id}, {'_id': 0, 'study': 1})
            if setup is None:
                raise Exception('There is no exception with that ObjectID')

            self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
            self.study_name = setup['study']

        if os.path.basename(os.getcwd()) != self.study_path:
            if not os.path.isdir('./experiments/' + self.study_path):
                os.mkdir('./experiments/' + self.study_path)
            os.chdir('./experiments/' + self.study_path)

        # PLOTTING MEAN AND STANDARD DEVIATION EVOLUTION #######################################
        cursor = self.db.experiment.aggregate([
            {'$match': {'study_id': self.study_id}},
            {'$sort': {'experiment': 1}},
            {'$group': {'_id': '$setup_name',
                        'experiments_name': {'$push': '$name'},
                        'fitness': {'$push': '$fitness'},
                        'fitness_mean': {'$push': '$statistics.fitness_mean'},
                        'fitness_confidence_interval': {'$push': '$statistics.fitness_confidence_interval'},
                        'iterations': {'$push': '$iterations'},
                        'iterations_mean': {'$push': '$statistics.iterations_mean'},
                        'iterations_confidence_interval': {'$push': '$statistics.iterations_confidence_interval'},
                        'learning_iterations': {'$push': '$learning_iterations'},
                        'learning_iterations_mean': {'$push': '$statistics.learning_iterations_mean'},
                        'learning_iterations_confidence_interval': {'$push': '$statistics.learning_iterations_confidence_interval'}}},
            {'$sort': {'_id': 1 }}], allowDiskUse=True)

        if cursor is None:
            raise Exception('There is no logged data with that study id')

        plt.clf()
        experiments_name = []
        iterations_fig = 1
        fitness_fig = 2
        learning_iterations_fig = 3
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
        handles = []
        for color, setup in zip(self.plot_colors, cursor):
            if len(experiments_name)<len(setup['experiments_name']):
                experiments_name = setup['experiments_name']

            # Plotting iterations statistics for setup
            handles.append(mpatches.Patch(color=color, label=setup['_id'], alpha=0.5))
            plt.figure(iterations_fig)
            plt.subplot(gs[-1])
            plt.boxplot(setup['iterations'],
                        positions=setup['experiments_name'] if not isinstance(setup['experiments_name'][0],str) else [5],
                        showmeans=True,
                        meanline=True,
                        bootstrap=5000,
                        notch=True,
                        patch_artist=True,
                        showfliers=False,
                        boxprops={'facecolor': color, 'alpha':0.5},
                        whiskerprops={'color': color, 'alpha':0.5},
                        meanprops={'color': color, 'alpha':1},
                        capprops={'color': color, 'alpha':0.5},
                        widths=np.full(len(setup['experiments_name']),len(experiments_name))
                        )

            # Plotting fitness statistics for setup
            plt.figure(fitness_fig)
            plt.subplot(gs[-1])
            plt.boxplot(setup['fitness'],
                        positions=setup['experiments_name'] if not isinstance(setup['experiments_name'][0],str) else [5],
                        showmeans=True,
                        meanline=True,
                        bootstrap=5000,
                        notch=True,
                        patch_artist=True,
                        showfliers=False,
                        boxprops={'facecolor': color, 'alpha':0.5},
                        whiskerprops={'color': color, 'alpha':0.5},
                        meanprops={'color': color, 'alpha':1},
                        capprops={'color': color, 'alpha':0.5},
                        widths=np.full(len(setup['experiments_name']),len(experiments_name))
                        )
            if isinstance(setup['experiments_name'][0],str):
                # Plotting learning_iterations statistics for setup
                plt.figure(learning_iterations_fig)
                plt.subplot(gs[-1])
                plt.boxplot(setup['learning_iterations'],
                            positions=setup['experiments_name'] if not isinstance(setup['experiments_name'][0],str) else [5],
                            showmeans=True,
                            meanline=True,
                            bootstrap=5000,
                            notch=True,
                            patch_artist=True,
                            showfliers=False,
                            boxprops={'facecolor': color, 'alpha':0.5},
                            whiskerprops={'color': color, 'alpha':0.5},
                            meanprops={'color': color, 'alpha':1},
                            capprops={'color': color, 'alpha':0.5},
                            widths=np.full(len(setup['experiments_name']),len(experiments_name))
                            )

        # Setup iteration statistics figure
        plt.figure(iterations_fig)
        ax = plt.subplot(gs[-1])
        plt.xlabel("Recursions")
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), visible=False) 
        if not isinstance(setup['experiments_name'][0],str):
            plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
            #PETABA
            #plt.yscale('log')
        else:
            plt.xlim(0,10)            
        plt.ylabel("Iterations")
        plt.legend(handles=handles, loc=2, bbox_to_anchor=(1, 1))
        plt.grid()

        # Setup fitness statistics figure
        plt.figure(fitness_fig)
        ax = plt.subplot(gs[-1])
        plt.xlabel("Recursions")
        ax.xaxis.tick_top()
        plt.setp(ax.get_xticklabels(), visible=False)
        if not isinstance(setup['experiments_name'][0],str):
            plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
            plt.yscale('log')
        else:
            plt.xlim(0,10)
        plt.ylabel("Best fitness")
        plt.legend(handles=handles, loc=2, bbox_to_anchor=(1, 1))
        plt.grid()

        if isinstance(setup['experiments_name'][0],str):
            # Setup learning iterations statistics figure
            plt.figure(learning_iterations_fig)
            ax = plt.subplot(gs[-1])
            plt.xlabel("Recursions")
            ax.xaxis.tick_top()
            plt.setp(ax.get_xticklabels(), visible=False)
            if not isinstance(setup['experiments_name'][0],str):
                plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
                plt.yscale('log')
            else:
                plt.xlim(0,10)
            plt.ylabel("Learning iterations")
            plt.legend(handles=handles, loc=2, bbox_to_anchor=(1, 1))
            plt.grid()

        # PLOTTING ANOVA COMPARISON PER EXPERIMENT #######################################
        anovas_fig_iteration_data = {}
        anovas_fig_fitness_data = {}
        comparisons = self.db.anova.aggregate([{'$match': {'study_id': self.study_id}},
                                          {'$group': {'_id': '$experiments.setup_name',
                                                      'experiments_name': {'$push': '$experiment'},
                                                      'iterations_p': {'$push': '$iterations.p'},
                                                      'fitness_p': {'$push': '$fitness.p'},
                                                      'learning_iterations_p': {'$push': '$learning_iterations.p'}}},
                                          {'$sort': {'_id': 1}}], allowDiskUse=True)

        # Bound for significance difference
        plt.figure(iterations_fig)
        plt.subplot(gs[0])
        if not isinstance(setup['experiments_name'][0],str):
            x_limit = [0-len(experiments_name), experiments_name[-1]+len(experiments_name)]
        else:
            x_limit = [0,10]
        plt.plot(x_limit,
                 np.full(len(x_limit), 0.05),
                 linestyle='dashed', color='k')

        plt.figure(fitness_fig)
        plt.subplot(gs[0])
        plt.plot(x_limit,
                 np.full(len(x_limit), 0.05),
                 linestyle='dashed', color='k')

        if isinstance(setup['experiments_name'][0],str):
            plt.figure(learning_iterations_fig)
            plt.subplot(gs[0])
            plt.plot(x_limit,
                    np.full(len(x_limit), 0.05),
                    linestyle='dashed', color='k')

        # Plotting p value for comparison and experiment
        for color, marker, comparison in zip(reversed(self.plot_colors), self.plot_markers, comparisons):
            plt.figure(iterations_fig)
            plt.subplot(gs[0])
            plt.scatter(comparison['experiments_name'] if not isinstance(comparison['experiments_name'][0],str) else [5],
                     comparison['iterations_p'],
                     edgecolor=color,
                     facecolors='none',
                     marker=marker,
                     alpha=0.5,
                     label=comparison['_id'][0]+' vs '+comparison['_id'][1])

            plt.figure(fitness_fig)
            plt.subplot(gs[0])
            plt.scatter(comparison['experiments_name'] if not isinstance(comparison['experiments_name'][0],str) else [5],
                        comparison['fitness_p'],
                        edgecolor=color,
                        facecolors='none',
                        marker=marker,
                        alpha=0.5,
                        label=comparison['_id'][0]+' vs '+comparison['_id'][1])
            
            if isinstance(setup['experiments_name'][0],str):
                plt.figure(learning_iterations_fig)
                plt.subplot(gs[0])
                plt.scatter(comparison['experiments_name'] if not isinstance(comparison['experiments_name'][0],str) else [5],
                            comparison['learning_iterations_p'],
                            edgecolor=color,
                            facecolors='none',
                            marker=marker,
                            alpha=0.5,
                            label=comparison['_id'][0]+' vs '+comparison['_id'][1])

        # Setup iteration anova figure and save
        plt.figure(iterations_fig)
        plt.subplot(gs[0])
        plt.title(r'Statistical comparison of the iterations for $G_{'+ self.study_name + r'}$')
        plt.ylim(ymin=-0.01)
        plt.ylabel("Anova p")
        lgd = plt.legend(loc=2, bbox_to_anchor=(1, 1))            
        plt.grid()
        if not isinstance(setup['experiments_name'][0],str):
            plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
            plt.xticks(experiments_name)
            plt.yscale('symlog', linthreshy=0.01)
        else:
            plt.xlim(0,10)
            plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom='off',      # ticks along the bottom edge are off
                            top='off',         # ticks along the top edge are off
                            labelbottom='off')
        plt.savefig('./' + self.study_name + '_iterations_evolution.png', dpi=150, additional_artists=lgd, bbox_inches="tight")

        # Setup fitness anova figure and save
        plt.figure(fitness_fig)
        plt.subplot(gs[0])
        plt.title(r'Statistical comparison of the fitness for $G_{'+ self.study_name + r'}$')
        plt.ylim(ymin=-0.01)
        plt.ylabel("Anova p")
        lgd = plt.legend(loc=2, bbox_to_anchor=(1, 1))            
        plt.grid()
        if not isinstance(setup['experiments_name'][0],str):
            plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
            plt.xticks(experiments_name)
            plt.yscale('symlog', linthreshy=0.01)
        else:
            plt.xlim(0,10)
            plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom='off',      # ticks along the bottom edge are off
                            top='off',         # ticks along the top edge are off
                            labelbottom='off')
            
        plt.savefig('./' + self.study_name + '_fitness_evolution.png', dpi=150, additional_artists=lgd, bbox_inches="tight")

        if isinstance(setup['experiments_name'][0],str):
            # Setup learning iteration anova figure and save
            plt.figure(learning_iterations_fig)
            plt.subplot(gs[0])
            plt.title(r'Statistical comparison of the learning iterations for $G_{'+ self.study_name + r'}$')
            plt.ylim(ymin=-0.01)
            plt.ylabel("Anova p")
            lgd = plt.legend(loc=2, bbox_to_anchor=(1, 1))                
            plt.grid()
            if not isinstance(setup['experiments_name'][0],str):
                plt.xlim(xmin=0-len(experiments_name), xmax=experiments_name[-1]+len(experiments_name))
                plt.xticks(experiments_name)
                plt.yscale('symlog', linthreshy=0.01)
            else:
                plt.xlim(0,10)
                plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom='off',      # ticks along the bottom edge are off
                            top='off',         # ticks along the top edge are off
                            labelbottom='off')
            plt.savefig('./' + self.study_name + '_learning_iterations_evolution.png', dpi=150, additional_artists=lgd, bbox_inches="tight")

    def plot_experiments_evolution(self):
        if self.study_path is None:
            setup = self.db.study.find_one({'_id': self.study_id}, {'_id': 0, 'study': 1})
            if setup is None:
                raise Exception('There is no experiment with that ObjectID')

            self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
            self.study_name = setup['study']

        if os.path.basename(os.getcwd()) != self.study_path:
            if not os.path.isdir('./experiments/' + self.study_path):
                os.mkdir('./experiments/' + self.study_path)
            os.chdir('./experiments/' + self.study_path)

        cursor = self.db.experiment.aggregate([
            {'$match': {'study_id': self.study_id}},
            {'$project': {'_id': 1, 'name': 1, 'setup_name': 1, 'initialization': 1, 'crossover': 1}},
            {'$lookup': {
                'from': 'sample',
                'localField': '_id',
                'foreignField': 'experiment_id',
                'as': 'samples'}},
            {'$sort': {'setup_name': 1}},
            {'$group': {'_id': '$name',
                        'setups': {'$push': {'experiment_id': '$_id',
                                             'setup_name': '$setup_name',
                                             'initialization': '$initialization',
                                             'crossover': '$crossover',
                                             'samples': '$samples'}}}},
            {'$sort': {'_id': 1}}], allowDiskUse=True)

        if cursor is None:
            raise Exception('There is no sample logged data with that study id and experiments ids')
        for experiment in cursor:
            plt.clf()
            plt.figure()

            for color, setup in zip(self.plot_colors, experiment['setups']):
                first_sample = True
                for sample in setup['samples']:
                    # Plotting average fitness evolution for setup [best_fitness o average_fitness]
                    if first_sample:
                        plt.plot(np.arange(0, len(sample['average_fitness'])),
                                 np.array(sample['average_fitness']),
                                 linestyle='solid', color=color, alpha=0.5,
                                 label=setup['setup_name'])
                        first_sample = False
                    else:
                        if 'average_fitness' in sample:
                            plt.plot(np.arange(0, len(sample['average_fitness'])),
                                     np.array(sample['average_fitness']),
                                     linestyle='solid', color=color, alpha=0.5)
                plt.ylabel("Avg. population fitness")
                #plt.ylabel("Best fitness")
                #plt.grid()

            plt.xlabel("Generations")
            lgd = plt.legend(loc=1)#, bbox_to_anchor=(0.5, -0.1))
            #plt.title("Target derivation tree with "+ str(experiment['_id']) +" recursions", y=1.08)
            #if isinstance(experiment['_id'], int):
            #    plt.title(r'Fitness evolution for $G_{'+ self.study_name +r'}$. Recursions of target tree: '+ str(experiment['_id']) + r'')
            #else:
            #    plt.title(r'Fitness evolution for $G_{'+ self.study_name +r'}$. '+ experiment['_id'] + r' problem')
            #plt.tight_layout()
            plt.savefig('./' + self.study_name + '_' + str(experiment['_id']) + '_experiment_evolution.png', dpi=150)#,additional_artists=lgd, bbox_inches="tight")

    def createDeleteStudyRoutine(self):
        if self.study_path is None:
            setup = self.db.study.find_one({'_id': self.study_id}, {'_id': 0, 'study': 1})
            if setup is None:
                raise Exception('There is no exception with that ObjectID')

        #     self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
        #     self.study_name = setup['study']
        #
        # if os.path.basename(os.getcwd()) != self.study_path:
        #     if not os.path.isdir('./' + self.study_path):
        #         os.mkdir(self.study_path)
        #     os.chdir(self.study_path)

            self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
            self.study_name = setup['study']

        if os.path.basename(os.getcwd()) != self.study_path:
            if not os.path.isdir('./experiments/' + self.study_path):
                os.mkdir('./experiments/' + self.study_path)
            os.chdir('./experiments/' + self.study_path)

        with open('./delete_study_' + str(self.study_id) + '.sh', 'w+') as f:
            f.write('#!/bin/bash\n')
            if self.user:
                f.write('mongo ' + self.server+'/log -u ' + self.user + ' -p ' + self.password + ' --eval \'db.study.remove({_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log -u ' + self.user + ' -p ' + self.password + ' --eval \'db.experiment.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log -u ' + self.user + ' -p ' + self.password + ' --eval \'db.sample.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log -u ' + self.user + ' -p ' + self.password + ' --eval \'db.anova.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
            else:
                f.write('mongo ' + self.server+'/log --eval \'db.study.remove({_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log --eval \'db.experiment.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log --eval \'db.sample.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
                f.write('mongo ' + self.server+'/log --eval \'db.anova.remove({study_id: ObjectId("' + str(self.study_id) + '")})\'\n')
            f.write('rm -rf ../' + os.getcwd().split('/')[-1] + '\n')
            f.close()

        os.chmod('./delete_study_' + str(self.study_id) + '.sh', 0o777)

    def createPlotRoutine(self):
        if self.study_path is None:
            setup = self.db.study.find_one({'_id': self.study_id}, {'_id': 0, 'study': 1})
            if setup is None:
                raise Exception('There is no study with that ObjectID')

            self.study_path = setup['study'].replace(' ', '_') + '_' + str(self.study_id)
            self.study_name = setup['study']

        if os.path.basename(os.getcwd()) != self.study_path:
            if not os.path.isdir('./experiments/' + self.study_path):
                os.mkdir('./experiments/' + self.study_path)
            os.chdir('./experiments/' + self.study_path)

        with open('./plot_experiments_' + str(self.study_id), 'w+') as f:
            import sys
            py_path = sys.executable.rsplit('/',1)
            py_path_str = py_path[0]+' '+py_path[1] 
            f.write('#!'+py_path_str+'\n')
            f.write('from evo.eda import Log\n')
            f.write('from bson.objectid import ObjectId\n')
            f.write('import os\n')
            f.write('os.chdir("../..")\n')
            if self.user:
                f.write('log = Log(server="' + self.server + '", user="' + self.user + '", password="' + self.password + '")\n')
            else:
                f.write('log = Log(server="' + self.server + '")\n')
            f.write('log.study_id = ObjectId("' + str(self.study_id) + '")\n')
            f.write('log.grammar ="' + self.grammar + '"\n')
            f.write('log.plot_experiments_evolution() \n')
            f.write('log.plot_range_study() \n')
            f.close()

        os.chmod('./plot_experiments_' + str(self.study_id), 0o777)


__author__ = "Pablo Ramos"
__license__ = "Apache License 2.0"
__version__ = "1.1.0"
__maintainer__ = "Pablo Ramos"
__email__ = "pablo.ramos@aturing.com"
__status__ = "Production"