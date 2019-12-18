""" Report results of a finished experiment """

import sys
from gplpy.evo.log import DBLogger
from bson.objectid import ObjectId

if __name__ == "__main__":
    logger = DBLogger()
    logger.study_id = ObjectId("5a00360dd140a51466d25933") #Change study_id 
    logger.grammar = "DFFNN"
    logger.obtain_statistics()
    logger.plot_experiments_evolution()
    logger.plot_range_study()
    logger.report_statistics()
    logger.createDeleteStudyRoutine()


