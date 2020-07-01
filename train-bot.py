#########################
# train-bot.py
# Author: Matt Balshaw
##########################
# Train our nn to see if it can get decent accuracy
#########################

from fastai.vision import *
import datetime
from fastai.callbacks import *
import warnings

#############

if __name__ == "__main__":

    # Load in our label data
    allData = pd.read_csv("gaf_labels.csv")
    dataPath = 'gafs/'

    # We want 70% train, 20% valid data, 10% test data
    # Since data is time based, we don't want a random split
    # So, we will take the first 70% of data as train and the next 20% as valid

    trainPct = 0.7
    validPct = 0.2
    testPct = 0.1

    maxTrainLoc = int(len(allData) * 0.7)
    maxValidLoc = maxTrainLoc + int(len(allData) * 0.2)
    # Set the column we want
    filenameCol = 'filename'

    # Create the training data frame
    trainDF = allData[:maxTrainLoc]
    trainData = pd.DataFrame()
    trainData["filename"] = trainDF[filenameCol]
    trainData["label"] = trainDF["tomorrowResult"]

    # Create the valid data frame
    validDF = allData[maxTrainLoc:maxValidLoc]
    validData = pd.DataFrame()
    validData["filename"] = validDF[filenameCol]
    validData["label"] = validDF["tomorrowResult"]

    # Ignore warnings for image size
    warnings.filterwarnings("ignore", category=UserWarning,
                            module="torch.nn.functional")

    print("Creating Training set")

    # Hardcoding the datapath for now
    dataSet = ImageDataBunch.from_df(
        df=trainData,
        path=dataPath,
        valid_pct=0.,
        size=32,
        num_workers=4,
        bs=128,
        ds_tfms=get_transforms(
            max_rotate=None,
            do_flip=False,
            max_lighting=0.1,
            max_zoom=1.0,
            max_warp=0.),
    ).normalize(imagenet_stats)

    print("Training set created")

    valid = ImageDataBunch.from_df(
        df=validData,
        path=dataPath,
        valid_pct=0.,
        size=32,
        num_workers=4,
        bs=128,
        ds_tfms=get_transforms(
            max_rotate=None,
            do_flip=False,
            max_lighting=0.1,
            max_zoom=1.0,
            max_warp=0.),
    ).normalize(imagenet_stats)

    dataSet.valid_dl = valid.train_dl

    print("Validation Set created")

    print("Databunch complete")

    # Create the model to be trained...

    model = cnn_learner(
        dataSet,
        models.resnet34,
        metrics=[accuracy],
    )

    print("Model created")

    # Get lr

    # model.lr_find()
    # model.recorder.plot(suggestion=True)
    # plt.title('Magic Learn Rate Graph')
    # plt.savefig("results/lr-magic.png",
    #             bbox_inches='tight', pad_inches=0.2, transparent=False)
    # plt.close()

    # Only save models that are more accurate

    minLearn = 2e-5
    maxLearn = 2e-3
    epochs = 10

    model.unfreeze()
    model.freeze_to(-20)
    model.fit_one_cycle(
        epochs,
        slice(minLearn, maxLearn),
        callbacks=[
            SaveModelCallback(
                model,
                every='improvement',
                monitor='accuracy',
                name='title')])
    print("done training")

    # Gets loss over time
    model.recorder.plot_losses()
    plt.title('Model Losses')
    plt.savefig("results/losses.png",
                bbox_inches='tight', pad_inches=0.2, transparent=False)
    plt.close()

    # Gets metrics over time
    model.recorder.plot_metrics()
    plt.title('Metrics')
    plt.savefig("results/metrics.png",
                bbox_inches='tight', pad_inches=0.2, transparent=False)
    plt.close()

    # Gets learning rate used
    model.recorder.plot_lr(show_moms=True)
    plt.title('Learning Rate')
    plt.savefig("results/learning-stage.png",
                bbox_inches='tight', pad_inches=0.2, transparent=False)
    plt.close()

    interp = ClassificationInterpretation.from_learner(model)

    interp.plot_confusion_matrix()
    plt.title('Confusion for validation data')
    plt.savefig("results/confusion.png",
                bbox_inches='tight', pad_inches=0.2, transparent=False)
    plt.close()
