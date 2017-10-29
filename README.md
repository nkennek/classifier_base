# classifier_base

## Important Notes

### How to use

```
$ pip install --user path/to/classifier_base
```

if you want to do training, let's

```
$ pip install cupy
or
$ pip install --user -e .[dev] /path/to/classifier_base
```

also, setup for cuda environment will be required.

### Training Scripts

/scripts/models

- model_config.py:
    - all about training. (hyper parameters, model inilialization, path schemes...)
    - you can see setting with ```$ python model_config.py```
- train_models.py
    - train a model according to model_config.py with simply
    - ```$ python train_models.py```
- predict.py
    - classify unlabelled data.
```
  $ python predict.py --help
      optional arguments:
      -h, --help       show this help message and exit
      --model [MODEL]  path to model for judge. if not given, most recent one will
                       be automatically used
      --file [FILE]    if provided, an image is loaded. else, images are loaded
                       from
                       /classifier_base/data/to_predict
      --dir [DIR]      if provided, image are loaded from fed directory
      --out [OUT]      path to output result. by default, the same as model's
```

- cross_validation.py
    - cross validation with ```$ python cross_validation.py```


### Web Application

/scripts/webapp

- app.py
    - implemented with flask
    - ```python app.py --port 8080```
