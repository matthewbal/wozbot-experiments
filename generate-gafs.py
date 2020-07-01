#########################
# generate-gafs.py
# Author: Matt Balshaw
##########################
# Create our labelled image data for AI training
#########################

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from pathlib import Path
import datetime
import shutil
import os
import random
from os import path
import copy

#############


def save_gaf(periodData, simpleFileName):
    # Clear any previous graphs
    plt.clf()

    # Need to reshape to numpy for plotting in gasf
    periodnp = periodData.to_numpy().reshape(1, -1)

    # Create GAF from data
    gaf = GramianAngularField(
        image_size=len(periodData),
        method='summation')
    gafData = gaf.transform(periodnp)

    # Plot the gaf to an image
    plt.imshow(gafData[0],
               cmap='rainbow',
               origin='lower',
               interpolation='nearest')
    plt.axis('off')

    # Now we save this gaf as a file

    dest = "gafs/" + simpleFileName
    plt.savefig(dest,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)
    plt.close()


if __name__ == "__main__":
    # First, load in the raw data
    kData = pd.read_csv('rawdata/cleanAUSData.csv')
    print(kData)

    # Let's just deal with one location, Darwin

    kData = kData[kData['Location'] == 'Darwin']

    print(kData)

    # Now, our highest correlating field to rain tomorrow was Humididy 3pm.
    # so, let's make gafs of this value over time.

    usingCols = [
        "Date",
        "Location",
        "Humidity3pm",
        "RainNext4days",
    ]

    kData = kData[usingCols]

    print(kData)

    # Ensure the gafs folder exists

    Path(
        "gafs"
    ).mkdir(parents=True, exist_ok=True)

    # So, we'll start by using 14 days of humidity data to try predict
    # if it will rain tomorrow

    totalDays = len(kData)
    daysPerGraph = 32

    #  As we don't have the final data
    # We stop dayrange + 1 days before end of the data
    maxDay = totalDays - (daysPerGraph + 1)

    print("About to create %s gafs" % maxDay)

    labelDF = pd.DataFrame()

    # Now we loop over every day in our whole dataset
    for i in range(0, maxDay):

        # Print every 5% progress
        if i % (maxDay // 20) == 0:
            pct = round((i / maxDay) * 100.0, 3)
            prog = ("%s" % pct) + "%..."
            print(prog, end='', flush=True)

        # i is the day number
        # at day 0, we need to be x days ahead
        # since we need past data

        startRow = i
        endRow = startRow + daysPerGraph

        #########################
        # First we generate the GAF
        #########################

        # Get the date of the day we are predicting for
        currentDate = kData.iloc[[i + daysPerGraph]]['Date'].item()

        # Get our period of data
        humidityOnly = kData[["Humidity3pm"]]
        periodDF = humidityOnly[startRow:endRow]
        simpleFileName = "day-%s-%s.png" % (
            str(i + 1).rjust(5, "0"),
            currentDate
        )

        save_gaf(periodDF, simpleFileName)

        #########################
        # Now we determine the gaf data
        #########################

        # We have a 14 days of data
        # Now we want just the following day's rain label
        # Get our period of data
        rainnext = kData.iloc[[i + daysPerGraph]]['RainNext4days'].item()

        if rainnext == 1.:
            dayLabel = "Rain"
        else:
            dayLabel = "Dry"

        #########################
        # Now we add all gaf data to the csv
        #########################

        gafDataRow = {
            "date": [currentDate],
            "tomorrowResult": [dayLabel],
            "filename": [simpleFileName]
        }

        newRow = pd.DataFrame(gafDataRow)
        labelDF = labelDF.append(newRow)

    print("All GAF's baked")

    # Output the gafdata to a labels file
    dataDest = "gaf_labels.csv"
    labelDF.to_csv(dataDest, index=False)
