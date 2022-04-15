import weka.core.jvm as jvm
from weka.core.converters import Loader
import weka.core.converters as converters
from weka.filters import Filter
from weka.classifiers import Evaluation
from weka.core.classes import Random
from weka.classifiers import FilteredClassifier
from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
import weka.plot.dataset as pld
import javabridge
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_correlations(search_class_name: str, evaluation_class_name: str):
    search = ASSearch(classname=search_class_name)
    evaluator = ASEvaluation(classname=evaluation_class_name)
    attribute_selection = AttributeSelection()
    attribute_selection.search(search)
    attribute_selection.evaluator(evaluator)
    attribute_selection.select_attributes(data)

    print("------------------------")
    # df_data = pd.DataFrame(data=principal_evaluator.jwrapper.methods['getCorrelationMatrix'][0]())
    print("# attributes: " + str(attribute_selection.number_attributes_selected))
    print("attributes: " + str(attribute_selection.selected_attributes))
    print("result string:\n" + attribute_selection.results_string)


if __name__ == '__main__':
    jvm.start()
    jvm.start(system_cp=True, packages=True)
    # jvm.start(packages="C:\Users\saada\wekafiles\packages")
    jvm.start(max_heap_size="512m")
    data = converters.load_any_file("heart_failure_clinical_records_dataset.csv")
    data.class_is_last()

    # search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])

    print_correlations("weka.attributeSelection.Ranker", "weka.attributeSelection.CorrelationAttributeEval")

    print_correlations("weka.attributeSelection.Ranker", "weka.attributeSelection.PrincipalComponents")

    df_data = pd.DataFrame(data=[[1,      0.09,  -0.08,  -0.1,    0.06,   0.09,  -0.05,   0.16,  -0.05,   0.07,   0.02,  -0.22],
                [0.09,   1,     -0.19,  -0.01,   0.03,   0.04,  -0.04,   0.05,   0.04,  -0.09,  -0.11,  -0.14],
                [-0.08,  -0.19,   1,     -0.01,  -0.04,  -0.07,   0.02,  -0.02,   0.06,   0.08,   0,     -0.01],
                [-0.1,   -0.01,  -0.01,   1,     -0,     -0.01,   0.09,  -0.05,  -0.09,  -0.16,  -0.15,   0.03],
                [0.06,   0.03,  -0.04,  -0,      1,      0.02,   0.07,  -0.01,   0.18,  -0.15,  -0.07,   0.04],
                [0.09,   0.04,  -0.07,  -0.01,   0.02,   1,      0.05,  -0,      0.04,  -0.1,   -0.06,  -0.2],
                [-0.05,  -0.04,   0.02,   0.09,   0.07,   0.05,   1,     -0.04,   0.06,  -0.13,   0.03,   0.01],
                 [ 0.16,   0.05,  -0.02,  -0.05,  -0.01,  -0,     -0.04,   1,     -0.19,   0.01,  -0.03,  -0.15],
                 [-0.05,   0.04,   0.06,  -0.09,   0.18,   0.04,   0.06,  -0.19,   1,     -0.03,   0,      0.09],
                  [0.07,  -0.09,   0.08,  -0.16,  -0.15,  -0.1,   -0.13,   0.01,  -0.03,   1,      0.45,  -0.02],
                  [0.02,  -0.11,   0,     -0.15,  -0.07,  -0.06,   0.03,  -0.03,   0,      0.45,   1,     -0.02],
                 [-0.22,  -0.14,  -0.01,   0.03,   0.04,  -0.2,    0.01,  -0.15,   0.09,  -0.02,  -0.02,   1]],
                 columns=['age','anaemia', 'creatinine_phosphokinase','diabetes','ejection_fraction'
                     , 'high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time'],
                           index=['age','anaemia', 'creatinine_phosphokinase','diabetes','ejection_fraction'
                     , 'high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time'])
    a4_dims = (22, 15)
    fig, ax = plt.subplots(figsize=a4_dims)
    res = sns.scatterplot(ax=ax, data=df_data)
    plt.savefig('output/plots/data2.png')
    plt.clf()

    sns.heatmap(df_data, annot=True, fmt="10.2f", cmap="YlGnBu")
    plt.savefig('output/plots/data2_heatmap.png')
    plt.clf()

    jvm.stop()
