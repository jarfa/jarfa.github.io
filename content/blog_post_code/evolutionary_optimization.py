from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

def cutoff(x, min_c, max_c):
    return np.minimum(np.maximum(x, min_c), max_c)

def score_function(mod, ix):
    def _scorer(x):
        return mod.predict_proba(x.reshape(1, -1)).reshape((10,))[ix]
    
    return _scorer

def find_best_img(score_fn, epochs, n_children, sd, lr, max_score=1):
    img = np.random.random(64) * 16
    for _ in range(epochs):
        noise = [np.random.normal(0, sd, 64) for _ in range(n_children)]
        child_scores = np.array([score_fn(img + n) for n in noise])

        child_stdev = np.std(child_scores)
        if child_stdev == 0:
            # we've reached a plateau
            break
        # normalize the scores
        child_scores -= np.mean(child_scores)
        child_scores /= child_stdev
        # see the paper and Huszar's blog post for the math behind this
        gradient = np.mean([n * s for n,s in zip(noise, child_scores)],
                           axis=0) / sd
        img = cutoff(img + lr * gradient, 0, 16)
        img_score = score_fn(img)
        if img_score >= max_score:
            # no reason to continue
            break
    
    return img, img_score

digits = datasets.load_digits()
models = dict(
    rf=RandomForestClassifier(
        criterion="entropy",
        n_estimators=100,
        min_samples_leaf=5,
    ),
    gbt=GradientBoostingClassifier(
        loss="deviance",
        n_estimators=100,
        learning_rate=0.1,
        min_samples_leaf=5
    ),
    lr=LogisticRegressionCV(
        Cs=10,
        penalty='l2',
    ),
    mlp=MLPClassifier(
        hidden_layer_sizes=(24,),
        activation='relu',
        learning_rate_init=0.01,
        alpha=1.0,
    ),
)

results = dict((mod, {}) for mod in models.keys())
np.random.seed(5)
for mod_name, model in models.iteritems():
    print("Starting model {mod} at {dt}".format(
        mod=mod_name,
        dt=datetime.now()
    ))
    # train with the entire dataset, we don't need a holdout
    model.fit(digits.data, digits.target)
    for target in range(10):
        image, score = find_best_img(
            score_function(model, target),
            epochs=250,
            n_children=25,
            sd=5,
            lr=0.75,
        )
        results[mod_name][str(target)] = (image, score)

# What if we use an ensemble model?
results["ensemble"] = {}
def ensemble_scorer(t):
    scorers = [score_function(m, t) for m in models.values()]
    def _ensemble_scorer(x):
        return np.mean([s(x) for s in scorers])

    return _ensemble_scorer

print("Starting ensemble at {dt}".format(dt=datetime.now()))
for target in range(10):
    image, score = find_best_img(
        ensemble_scorer(target),
        epochs=300,
        n_children=25,
        sd=5,
        lr=0.75,
    )
    results["ensemble"][str(target)] = (image, score)

print("Done at\t{dt}".format(dt=datetime.now()))

for mod_name in results.keys():
    for t in [str(i) for i in range(10)]:
        # put it in a JSON-able format
        image, score = results[mod_name][t]
        results[mod_name][t] = (list(image), score)

fname = "per_model_data_1.json"

with open(fname, "w") as f:
    json.dump(results, f, sort_keys=True, indent=4, separators=(',', ': '))

# # If you want to skip the data creation, read it back in here
# with open(fname, "r") as f:
#     results = json.load(f)

for mod_name in results.keys():
    for t in [str(i) for i in range(10)]:
        # change the image data back to ndarray
        image, score = results[mod_name][t]
        results[mod_name][t] = (np.array(image), score)


model_full_names = dict(
    rf="Random\nForest",
    gbt="Gradient Boosted\nTrees",
    lr="Logistic\nRegression",
    mlp="Neural Network\n(1 Hidden Layer)",
    ensemble="Ensemble\nof Models",
)
plt.style.use(["ggplot"])
for suffix in ("models", "plus_ensemble"):
    # run it once with only the models, once with models + ensemble
    model_names = ["lr", "mlp", "rf", "gbt",]
    if suffix == "plus_ensemble":
        model_names.append("ensemble")

    fig, axarr = plt.subplots(nrows=10, ncols=len(model_names),
                              figsize=(16, 16))

    for j, mod_name in enumerate(model_names):
        for target in range(10):
            image, score = results[mod_name][str(target)]
            ax = axarr[target, j]
            ax.imshow(
                image.reshape((8, 8)),
                cmap=plt.cm.gray_r,
                interpolation='nearest'
            )
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
            ax.set_ylabel(
                "{t:d}:  {s:.2f}".format(t=target, s=score),
                size=15
            )

            if target == 0:
                ax.set_title(model_full_names[mod_name], size=20)

    plt.tight_layout()
    plt.savefig("best_examples_{0}.png".format(suffix), pad_inches=0.1)
