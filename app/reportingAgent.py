
"""
reportAgent.py

main app script

"""
import sys
from conf.logManager import Logger
import logging

from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent)
sys.path.append(root_path) # import root project to env

logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)
log.info(f"Added ENV = {root_path}")


from projectSetup import Setup
from mods.metricsEvaluator import MetricsEvaluator
from mods.dataHandler import DataHandler, Report
from mods.testBench import TestBench
from mods.reportGenerator import ReportGenerator
from mods.modelLoader import ModelLoader


def main():
    print("main script")
    

if __name__ == "__main__":
    main()
