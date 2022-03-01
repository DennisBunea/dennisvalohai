from valohai import Pipeline
def main(config)-> Pipeline:
    #Create a pipeline called utilispipeline
    pipe = Pipeline(name="utilspipeline", config=config)
    #Define a pipeline nodes
    preprocess = pipe.execution("preprocess-dataset")
    train = pipe.execution("train")
    test = pipe.execution("test")
    #Configure the pipeline / Define edges
    preprocess.output('train').to(train.input('train'))
    preprocess.output('test').to(test.input('test'))
    return pipe