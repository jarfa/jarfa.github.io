Title: Finding a Confidence Interval for Lift
Date: 2016-06-03 16:33
Category: Blog

Note: this post doesn't describe original mathematical research on my part. Honestly, my motivation is simply to make it easier for the next person who googles "confidence interval for lift" or something along those lines. I was surprised by how little useful information that search found when I was researching this for a project at Magnetic.

# What is Lift?

Imagine that we work at an advertising technology company called [Magnetic](www.magnetic.com). We have two ideas for strategies we will use to decide which internet users sees which ads: **strategy A** and **strategy B**. We already suspect that strategy B is better, and therefore want to know how much better it is.

We use strategy A to show ads to 20,000 people, 20 of whom buy the product we are showing to them (that purchase is called a *conversion* in our industry). Strategy B is used on 1,000 people, 3 of whom buy the product. So strategy A has a 0.1% conversion rate, while strategy B's conversion rate is 0.3%. How much better is strategy B?

We could subtract the conversion rates: 0.3% - 0.1% = 0.2%. But while 0.2% is a big jump starting from 0.1%, it's a tiny change if the conversion rates were 90.2% and 90%. If we divide the conversion rates, however, we get lift. 0.3% / 0.1% =  3. So strategy B's conversion rate is 3x better.

Oftentimes you'll see lift presented as a percentage centered at 0 (so 0% lift would mean no change). If `lift = 100 * (conv_rate_B / conv_rate_A - 1)`, we get 200%. In this blog post I'll mostly use latter way of computing lift, but please remember that 3x lift and 200% lift are the same thing.

One more caveat - there are two ways to conceive of lift. You could either talk about lift as the quotient when comparing two separate rates, or the rate of a subset of a population compared to the rate of the larger population. A bit more formally, the former is `P(event | condition B) / P(event | condition A)` where conditions A and B never occur together, while the latter is `P(event | condition) / P(event)`. In this blog post we'll solely talk about the former, `P(event | condition B) / P(event | condition A)` - but please don't be confused if you hear other people discuss the alternative version of lift.


# Why Do You Care About Confidence Intervals?

For those of you who are not statisticians and therefore don't already trust me that you *should* care about [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval), I owe you an explanation. 

Let's continue with our example from above: strategy A is shown to 20,000 people and results in 20 conversions, while strategy B is shown to 1,000 people for 3 conversions. We have relatively small sample sizes by the standard of online advertising. 200% lift sounds very impressive, but that's based on 3 conversions vs. 20 conversions. We expect that if we ran this experiment again on the same number of people those numbers would not be the same - so can we actually conclude that strategy B is the way to go? If we gave both strategies 100x more users to show ads to, would strategy B still get us a 3x higher conversion rate?

Statisiticans like to imagine that the gods have endowed both strategies A and B with 'true' conversion rates that are generating this data, and that there therefore exists in the heavens a true lift for stategy B with regards to strategy A. A confidence interval would tell us that, given our data and a preference for how confident we'd like to be (let's say 95% just for fun), we can expect that the true lift will be in the computed confidence interval with 95% probability.

More practically, when a product manager asks my team "Is strategy B better than strategy A?", we want to answer either "it's better", "it's worse", or "we don't have enough data to know". If the confidence interval on lift includes 0, that would imply we don't have enough data to know whether B is better than A. If the confidence interval lies entirely on the "better" or "worse" sides, we can give them a more interesting answer

Whew. Now you understand why we want a confindence interval for lift. How do we get it?

# Computing Confidence Intervals via Simulation

Let's start with a simulation-based method. They're far more fun and intuitive than formulas, and once we have a formula we can check it against the results we get from simulations. As an aside - I recently watched Jake Vanderplas's talk at PyCon 2016, [Statisitics for Hackers](https://www.youtube.com/watch?v=-7I7MWTX0gA), which I highly recommend watching. To the extent that I've ever had a philosophy behind statistical tests, he shares it and explains it.

As mentioned, lift is a proportion of two proportions. We already know how proportions of successful attempts / total attempts are distributed - that's the [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution). David Robinson has an [excellent post](http://varianceexplained.org/statistics/beta_distribution_and_baseball/) explaining its interpretation and usefullness, so I'll refrain from elaborating on that. To simulate the distribution of lift we just need to simulate pairs of beta distributions, and take their quotient. Please ignore the fact that in this blog post I'm switching back and forth between frequentist and Bayesian approaches.

```
import numpy as np

def lift_simulations(Na, Pa, Nb, Pb, Nsim=10**4):
    """
    Na is the total events for strategy A,
    Pa is positive events (conversions) for A, 
    etc.
    """
    # add 1 to both alpha and beta for a uniform prior 
    cvr_b = np.random.beta(1 + Pa, 1 + Na - Pa, size=Nsim)
    cvr_b = np.random.beta(1 + Pa, 1 + Na - Pa, size=Nsim)
    return (cvr_b / cvr_a) - 1.0
```

Now we have a function that will give us as many simulated data points as we want. What about a confidence interval? We have an array of simulated data points, so we just need to use the precentile function to find the bounds of our confidence interval.

```
def sim_conf_int(Na, Pa, Nb, Pb, interval, Nsim=10**4,, CI=0.95):
    simulations = lift_simulations(Na, Pa, Nb, Pb, Nsim=Nsim)
    if interval == "upper":
        return (np.percentile(simulations, 100 * (1 - CI)), float("inf"))
    if interval == "lower":
        return (float("-inf"), np.percentile(simulations, 100 * CI))
    if interval == "two-sided":
        return np.percentile(simulations, (100 * (1 - CI)/2, 100*(1 - (1 - CI)/2)))

    raise ValueError("interval must be either 'upper', 'lower', or 'two-sided'")
```

The upper confidence interval is equivalent to saying "We are 95% certain the true value is above _." The lower interval is "We are 95% certain the true value is below _." And the two-sided interval is "We are 95% certain the true value is between _ and _". For symmetry, all three are returned as pairs of numbers - for high and low one of those numbers is positive or negative infinity.


<!-- The short version - the null hypothesis is that the strategies will get us the same outcome, so lift = 0%. That's a fancy way of saying that labeling these impressions and conversions as A vs. B doesn't matter. So let's switch up their labels a bunch of times, and see how impressive this 200% lift looks in context.

```
import numpy as np

def permute_lift(Na, Pa, Nb, Pb, Nsim=10**4):
    """
    Na is the total events for strategy A,
    Pa is positive events (conversions) for A, 
    etc.
    """
    impressions = np.array(
        Pa * [1] + (Na - Pa) * [0] + Pb * [1] + (Nb - Pb) * [0]
    )
    lifts = np.zeros(Nsim)
    for i in xrange(Nsim):
        if i > 0:
            np.random.shuffle(impressions)
        cvr_a = np.mean(impressions[:Na])
        cvr_b = np.mean(impressions[Na:])
        lifts[i] = cvr_b / cvr_a - 1
   
    print "P-value for lift > 0: %.03f" % np.mean(lifts >= lifts[0])

permute_lift(20000, 20, 1000, 3)

# P-value for lift > 0: 0.096
```
That's a p-value of 9.6%. So at the 95% level we are not confident that lift is above 0. -->