import math
from pickle import NONE
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from numpy import array
from random import random
import pickle
from tqdm import tqdm

# data struct to handle computing and holding data 
# given to the LivePlotter class by addEstimateData
class PlotData:
    def __init__(self, initVal = None):
        self.initVal = initVal
        self.currentEpochEstimation = initVal
        self.currentEpochData = []
        self.permData = [initVal]

    def addEstimate(self, estimate):
        
        #if estimate is not None:
        self.currentEpochData.append(estimate)

        self.currentEpochEstimation =                               \
            sum(self.currentEpochData) / len(self.currentEpochData) \
            if len(self.currentEpochData) != 0 else self.initVal
    


    def getError(self):
        if len(self.currentEpochData) < 2:
            return 0
        else:
            return max(self.currentEpochData)-min(self.currentEpochData)

    def epochDone(self):

        self.permData.append(self.currentEpochEstimation)
        self.initVal=self.currentEpochEstimation
        self.currentEpochData.clear()
        # self.currentEpochEstimation = self.initVal

        # after plotting, points stay plotted
        # so no need to store points after plotting

        # each call of this function adds at most 1 value to 
        # permdata, so only need to remove one per call to 
        # keep max length = 2
        if len(self.permData) > 2:
            self.permData.pop(0)


    def setInitVal(self, val):
        self.initVal = val
        self.currentEpochEstimation = val
        self.permData = [x for x in self.permData if x is not None] + [val]
        while len(self.permData) < 2:
            self.permData.append(val)





# class to plot loss estimates live
class LivePlotter:
    def __init__(self,
                 lineNames, axiLimits = None, actualPointOpacity = .2,
                 loadLatestPkl=False, errorBarOpacity=0):

        self.N = lineN = len(lineNames)
        # initialise data objects
        self.data = [PlotData(None) for _ in range(lineN)]
        self.idle_draw_time = 1 / 9999

        self.axiLimit = [0, 1] if axiLimits is None else axiLimits
        self.axiFixed = axiLimits is None
        self.actualPointOpacity = actualPointOpacity
        self.lineNames = lineNames
        self.doErrorBars = errorBarOpacity > 0
        self.errorBarOpacity = errorBarOpacity

        self.hasLegend = False
        self.epoch = 0
        self.points_to_be_removed = []
        self.maxDataPoint = 1
        self.hasGottenFirstDatapoint = False
        self.savedEpochData = []

        self.colours = \
            ["red", "blue", "green", "olive", 
             "orange", "black", "purple", "chocolate",
             "cyan", "violet", "yellow"]

        

        if loadLatestPkl:
            self.loadData()




    def xlabel(self, label):
        plt.xlabel(label)
    def ylabel(self, label="epoch"):
        plt.ylabel(label)

    def title(self, title):
        plt.title(title)

    def addEstimateData(self, data, draw = True, log = False):
        assert len(data) == self.N

        # quick iter to check if axi needs to be enlarged
        #for item in data:
        #    if item is not None and item > self.maxDataPoint:
        #        self.maxDataPoint = item
        if (largest := max([x for x in data if x is not None])) > self.maxDataPoint:
            self.maxDataPoint = largest



        if not self.axiFixed:
            plt.axis(
                # x
                [0, self.epoch + 1] +
                # y
                self.axiLimit
            )


        for point in self.points_to_be_removed:
            point.remove()
        self.points_to_be_removed.clear()

        for i, item in enumerate(data):

            item = math.log(max(10e-10, item)) if log else item

            if not self.hasGottenFirstDatapoint:
                self.data[i].setInitVal(item)
                

            self.data[i].addEstimate(item)

            # scatter plot to show current loss estimation

            # actual point being added
            if self.actualPointOpacity > 0 and item is not None:
                self.points_to_be_removed.append(
                    plt.scatter(self.epoch + 1, item, 
                                c=self.colours[i], marker="x", alpha = self.actualPointOpacity)
                )
                self.points_to_be_removed.append(
                    plt.plot(
                        array([self.epoch, self.epoch + 1]),
                        array([self.data[i].permData[-1], item]),
                        c = self.colours[i], alpha = self.actualPointOpacity
                    )[0]
                )
                self.points_to_be_removed.append(
                    plt.annotate(
                        str(item),
                        (self.epoch + 1, item),
                        c= self.colours[i], alpha = self.actualPointOpacity
                    )
                )

            # current average
            self.points_to_be_removed.append(
                plt.plot(
                    array([self.epoch, self.epoch + 1]),
                    [self.data[i].permData[-1], self.data[i].currentEpochEstimation],
                    c = self.colours[i],
                    label = self.lineNames[i] if self.lineNames is not None else ""
                )[0]    
            )
            self.points_to_be_removed.append(
                plt.scatter(self.epoch + 1, self.data[i].currentEpochEstimation, c=self.colours[i])
            )
            # self.points_to_be_removed.append(
            #     plt.annotate(
            #         str(self.data[i].currentEpochEstimation),
            #         (self.epoch + 1, self.data[i].currentEpochEstimation),
            #         (0, 10),
            #         c = self.colours[i]
            #     ) 
            # )


            
        if not self.hasLegend:
            plt.legend()
            self.hasLegend = True

        if draw:
            self.draw()

        self.hasGottenFirstDatapoint = True


    def concludeEpoch(self, draw = True, x_inc = 1, save_img=True, save_data=True):
        pass;
        for i, datapoint in enumerate(self.data):
            
            err = None if (not self.doErrorBars) else datapoint.getError()
            datapoint.epochDone()

            if self.doErrorBars:
                #print(err);quit()
                x, y = array([self.epoch, self.epoch + x_inc]), array(datapoint.permData)
                #print(x, y); quit()
                data = plt.errorbar(
                    x,y,
                    #array(datapoint.permData),
                    yerr=err,
                    xerr=0,
                    color = self.colours[i],
                    ecolor=(self.colours[i], self.errorBarOpacity)
                )
                
            else:
                data = plt.plot(
                    array([self.epoch, self.epoch + x_inc]),
                    array(datapoint.permData),
                    color = self.colours[i])
            
            self.savedEpochData.append(data)

            #datapoint.epochDone()
            #print(self.savedEpochData)
        
        if draw:
            self.draw()
            
        if save_data:
            self.saveData()

        self.epoch += x_inc

        if save_img:
            self.save_img()

    def save_img(self, name="latest.png"):
        plt.savefig(name)

    def draw(self):
        # plt.show(block=False)
        # plt.draw()
        plt.pause(self.idle_draw_time)

    def saveData(self, saveDir="PLOTDATA.pkl"):
        with open(saveDir, 'wb') as f:
            pickle.dump(self.savedEpochData, f)

    def loadData(self, saveDir="PLOTDATA.pkl", drawUpdate=True):
        with open(saveDir, "rb") as f:
            data = pickle.load(f)

        #print(data); quit()

        if len(self.savedEpochData) > 0:
            print(f"overwriting existing data by loading {saveDir}")
            self.saveData("PLOTDATA_overwrite.pkl")

        self.savedEpochData=data

        #prev_y=0
        #for x, y in tqdm(self.savedEpochData):

        #print(self.savedEpochData)

        plotData=[]
        prevX=0
        for i, data in enumerate(tqdm(self.savedEpochData[100:])):
            x, y = data[0].get_data()
            #print(x)
            plotData.append(float(y[0]))
            if len(plotData) == 3:
                #print(plotData); quit()
                self.addEstimateData(plotData, False)
                self.concludeEpoch(x_inc=4, save_data=False)
                #prevX=x[0]
                plotData.clear()
                


        self.draw()
        quit()

    def __del__(self):
        
        try:
            self.saveData()
        except Exception as e:
            print(f"plot on exit data save failure: {str(e)}")

        try:
            if plt is not None:
                plt.savefig("latest.png")
            if plt is not None:
                plt.show()
        except Exception as e:
            print(f"plot on exit image save failure: {str(e)}")




from math import log10, log2


def logb(x, b=.5):
    return log2(x)/log2(b)

import matplotlib.colors as mcolors


if False:

        #d = [[0.0618743896484375, 0.03326225280761719],
        #[0.057611703872680664, 0.026390552520751953],
        #[0.33951711654663086, 0.19370031356811523],
        #[0.18335866928100586, 0.04704594612121582],
        #[0.1275920867919922, 0.005788087844848633],
        #[24.583935022354126, 2.114828586578369]
        #]

    d = [[0.0618743896484375, 0.03326225280761719],
        [0.057611703872680664, 0.026390552520751953],
        #[0.33951711654663086, 0.19370031356811523],
        [0.18335866928100586, 0.04704594612121582],
        [0.1275920867919922, 0.005788087844848633],
        #[24.583935022354126, 2.114828586578369],
        #[24.583935022354126, 2.114828586578369]
        ]


    #colours = \
    #        ["red", "blue", "green", "yellow", 
    #         "orange", "black", "purple", "chocolate",
    #         "cyan", "violet", "olive"][1:]

    colours = \
            ["red", "blue", "green", 
             "orange", "black", "purple", "chocolate",
             "cyan", "violet", "olive"][1:]

    #ns = ['FGSM', 'FFGSM', 'BIM', 'UPGD', 'PGD', 'APGD']
    ns = ['FGSM', 'FFGSM',  'UPGD', 'PGD']
    #ns = [ns[i // 2] + ' avg' if i % 2 == 0 else ' max' for i in range(len(ns))]
    ns = [[a + '\navg', a + '\nmax'] for a in ns]
    
    d = [[a / 23, b] for a, b in d]

    for i, (xs, ys) in enumerate(zip(ns, d)):
        plt.bar(xs, ys, width=1, color=[colours[i], (colours[i], .8)])
    plt.show()
    quit()