from valohai import Pipeline
def main(config)-> Pipeline:
    #Create a pipeline called utilispipeline
    pipe = Pipeline(name="utilspipeline", config=config)
    #Define a pipeline nodes
    preprocess = pipe.execution("preprocess-dataset")
    train = pipe.execution("train")
    #Configure the pipeline / Define edges
    preprocess.output('train').to(train.input('train'))
    return pipe